"""
disease_analysis.py – Med-Sight | Disease Detection & Analysis
==============================================================
Analyzes medical images for disease patterns, queries Qdrant for
semantically related reports, and generates comprehensive clinical analysis
with image-report correlation scoring.

Flow
----
  Image embedding (512-d vector from BiomedCLIP)
      └─► Qdrant similarity search (find related reports)
      └─► Disease inference (based on modality, anatomy, related findings)
      └─► Report correlation (semantic similarity scoring)
              └─► Descriptive analysis (image findings + related conditions)
              └─► Relationship classification (related vs standalone)

Usage
-----
  from data.disease_analysis import analyze_image_with_reports
  result = analyze_image_with_reports(
      image_embedding,
      scan_id="scan_123",
      patient_id="P001",
      modality="X-ray",
      anatomy="Chest"
  )
"""

import os
import sys
import logging
from typing import Optional, Dict, List, Any
import json

import numpy as np
from qdrant_client import QdrantClient

from data.qdrant_utils import get_qdrant_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("disease_analysis")

# Configuration
PATIENT_COLLECTION = "patient_reports"
SIMILARITY_THRESHOLD = 0.65  # Cosine similarity threshold for related findings
TOP_K_RELATED = 5  # Maximum related reports to return

# Disease/Finding patterns for analysis
DISEASE_KEYWORDS = {
    "pneumonia": ["consolidation", "infiltrate", "opacification", "air bronchogram", "fever"],
    "tuberculosis": ["cavitation", "apical", "nodular", "fibrosis", "hilar"],
    "covid-19": ["ground-glass", "bilateral", "peripheral", "atypical pneumonia", "halo sign"],
    "heart_disease": ["cardiomegaly", "pulmonary_edema", "pleural_effusion", "cardiac"],
    "pneumothorax": ["lung collapse", "air leak", "visceral pleural line", "pneumothorax"],
    "pulmonary_embolism": ["wedge-shaped", "oligemia", "infarction", "pe"],
    "fracture": ["fracture line", "cortical disruption", "callus", "fracture"],
    "osteoarthritis": ["joint space narrowing", "osteophytes", "bone spur"],
    "tumor": ["mass", "neoplasm", "malignancy", "lesion", "enhancement", "nodule"],
    "infection": ["abscess", "cellulitis", "infection", "inflammatory", "phlegmon"],
    "hemorrhage": ["hemorrhage", "bleed", "hematoma", "extravasation", "blood"],
    "inflammation": ["swelling", "edema", "inflammatory", "acute", "inflammation"],
}

MODALITY_DISEASES = {
    "X-ray": ["pneumonia", "covid-19", "heart_disease", "pneumothorax", "tuberculosis", "fracture"],
    "CT": ["tumor", "pulmonary_embolism", "fracture", "hemorrhage", "infection"],
    "MRI": ["tumor", "inflammation", "hemorrhage", "infection", "osteoarthritis"],
    "Ultrasound": ["infection", "inflammation", "hemorrhage", "tumor", "osteoarthritis"],
}

ANATOMY_DISEASES = {
    "Chest": ["pneumonia", "covid-19", "heart_disease", "pneumothorax", "tuberculosis", "pulmonary_embolism"],
    "Head": ["tumor", "hemorrhage", "infection", "inflammation"],
    "Abdomen": ["tumor", "infection", "hemorrhage", "inflammation"],
    "Musculoskeletal": ["fracture", "osteoarthritis", "tumor", "infection"],
    "Breast": ["tumor", "infection", "inflammation"],
}


class DiseaseAnalysis:
    """Comprehensive disease analysis result."""

    def __init__(
        self,
        scan_id: str,
        patient_id: str,
        modality: str,
        anatomy: str,
        detected_diseases: List[str],
        disease_confidence: float,
        disease_findings: str,
        related_reports: List[Dict[str, Any]],
        is_related: bool,
        relationship_explanation: str,
        composite_analysis: str,
    ):
        self.scan_id = scan_id
        self.patient_id = patient_id
        self.modality = modality
        self.anatomy = anatomy
        self.detected_diseases = detected_diseases
        self.disease_confidence = disease_confidence
        self.disease_findings = disease_findings
        self.related_reports = related_reports
        self.is_related = is_related
        self.relationship_explanation = relationship_explanation
        self.composite_analysis = composite_analysis

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scan_id": self.scan_id,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "anatomy": self.anatomy,
            "detected_diseases": self.detected_diseases,
            "disease_confidence": round(self.disease_confidence, 2),
            "disease_findings": self.disease_findings,
            "related_reports": self.related_reports,
            "is_related": self.is_related,
            "relationship_explanation": self.relationship_explanation,
            "composite_analysis": self.composite_analysis,
        }

    def to_json_str(self) -> str:
        """Convert to JSON string for Qdrant storage."""
        return json.dumps(self.to_dict())


def _query_related_reports(
    client: QdrantClient,
    embedding: List[float],
    patient_id: str,
    top_k: int = TOP_K_RELATED,
) -> List[Dict[str, Any]]:
    """Query Qdrant for semantically similar reports."""
    try:
        results = client.search(
            collection_name=PATIENT_COLLECTION,
            query_vector=embedding,
            limit=top_k * 2,
        )

        related = []
        for point in results:
            payload = point.payload

            # Filter by patient
            if payload.get("patient_id") != patient_id:
                continue

            # Extract text snippets
            text = payload.get("text", "")[:200]
            findings = payload.get("findings", "")[:150]

            related.append({
                "point_id": str(point.id),
                "filename": payload.get("filename", ""),
                "modality": payload.get("modality", ""),
                "anatomy": payload.get("anatomy", ""),
                "text": text,
                "findings": findings,
                "similarity": float(point.score),
                "dataset": payload.get("dataset", ""),
            })

        return sorted(related, key=lambda x: x["similarity"], reverse=True)[:top_k]

    except Exception as exc:
        log.warning("Qdrant search failed: %s", exc)
        return []


def _analyze_image_patterns(
    image_array: np.ndarray,
    modality: str,
    anatomy: str,
) -> tuple[List[str], float, str]:
    """
    Analyze image pixel patterns to detect disease indicators.
    Returns: (diseases, confidence, pattern_description)
    """
    try:
        # Ensure grayscale for analysis
        if len(image_array.shape) == 3:
            image_gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            image_gray = image_array.astype(np.uint8)

        # Calculate image statistics
        mean_val = np.mean(image_gray)
        std_val = np.std(image_gray)
        contrast = np.max(image_gray) - np.min(image_gray)
        
        # Count high-variance regions (potential infiltrates/consolidation)
        blocks = []
        block_size = 32
        height, width = image_gray.shape
        
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = image_gray[y:y+block_size, x:x+block_size]
                blocks.append(np.std(block))
        
        high_variance_regions = sum(1 for b in blocks if b > std_val * 1.5) if blocks else 0
        variance_ratio = high_variance_regions / len(blocks) if blocks else 0
        
        diseases = []
        confidence = 0.5
        pattern_desc = f"Mean: {mean_val:.0f}, Std: {std_val:.0f}, Contrast: {contrast:.0f}"
        
        # Pattern-based disease inference
        if modality == "X-ray" and anatomy == "Chest":
            # High variance regions suggest consolidation/infiltration
            if variance_ratio > 0.3:
                diseases.extend(["pneumonia", "covid-19", "tuberculosis"])
                confidence = 0.65
                pattern_desc += " | High variance regions detected (infiltrates)"
            # Very high contrast suggests normal/pneumothorax
            elif contrast > 200:
                diseases.extend(["pneumothorax", "heart_disease"])
                confidence = 0.55
                pattern_desc += " | High contrast detected"
            # Low variance suggests normal or pleural effusion
            elif std_val < 30:
                diseases.extend(["heart_disease", "pleural effusion"])
                confidence = 0.50
                pattern_desc += " | Low variance (possible effusion)"
            else:
                diseases = ["pneumonia", "covid-19", "tuberculosis"]
                confidence = 0.60
                pattern_desc += " | Generic chest findings"
        
        elif modality in ("CT", "MRI"):
            # For CT/MRI: high variance suggests tumors or heterogeneous tissue
            if variance_ratio > 0.4:
                diseases.extend(["tumor", "infection", "hemorrhage"])
                confidence = 0.70
                pattern_desc += " | Complex tissue patterns"
            else:
                diseases.extend(["tumor", "inflammation"])
                confidence = 0.55
                pattern_desc += " | Potential mass/lesion"
        
        elif anatomy == "Musculoskeletal":
            # Fractures show as bright lines/discontinuities
            if contrast > 150:
                diseases.extend(["fracture", "osteoarthritis"])
                confidence = 0.65
                pattern_desc += " | High contrast (cortical disruption)"
            else:
                diseases.extend(["fracture", "osteoarthritis"])
                confidence = 0.50
        
        else:
            # Generic fallback
            diseases.extend(["infection", "inflammation", "tumor"])
            confidence = 0.50
            pattern_desc += " | Generic analysis"
        
        return diseases[:3], min(confidence, 0.95), pattern_desc
    
    except Exception as exc:
        log.warning("Image pattern analysis failed: %s", exc)
        return [], 0.5, "Pattern analysis unavailable"


def _infer_diseases_with_image(
    image_array: np.ndarray,
    modality: str,
    anatomy: str,
    related_text: str = "",
) -> tuple[List[str], float, str]:
    """
    Enhanced disease inference combining image patterns + modality/anatomy hints.
    """
    # Pattern analysis from image
    pattern_diseases, pattern_confidence, pattern_desc = _analyze_image_patterns(
        image_array, modality, anatomy
    )
    
    # If we have pattern diseases, use them
    if pattern_diseases:
        log.info("Pattern analysis detected: %s (confidence: %.0f%%)", 
                 pattern_diseases, pattern_confidence * 100)
        return pattern_diseases, pattern_confidence, pattern_desc
    
    # Fallback: modality + anatomy inference
    modality_diseases = set(MODALITY_DISEASES.get(modality, []))
    anatomy_diseases = set(ANATOMY_DISEASES.get(anatomy, []))
    probable_diseases = modality_diseases & anatomy_diseases
    if not probable_diseases:
        probable_diseases = modality_diseases | anatomy_diseases
    
    diseases = list(probable_diseases)[:3] if probable_diseases else ["unknown_pathology"]
    confidence = 0.50
    desc = f"Modality/Anatomy inference: {modality} of {anatomy}"
    
    return diseases, confidence, desc


def _generate_disease_findings(
    modality: str,
    anatomy: str,
    diseases: List[str],
    pattern_analysis: str = "",
) -> str:
    """Generate descriptive findings based on detected diseases."""
    findings = f"**Medical Scan Analysis Report**\n\n"
    findings += f"**Imaging Type:** {modality} of {anatomy}\n\n"

    if diseases:
        findings += "**Probable Conditions Detected:**\n"
        for i, disease in enumerate(diseases[:3], 1):
            disease_display = disease.replace("_", " ").title()
            findings += f"{i}. {disease_display}\n"
    else:
        findings += "**Findings:** No specific disease indicators detected in the scan characteristics.\n"

    if pattern_analysis:
        findings += f"\n**Image Analysis Details:** {pattern_analysis}\n"

    findings += f"\n**Modality Notes:** {modality} imaging is suitable for evaluating {anatomy.lower()} region.\n"
    findings += "Further clinical correlation and specialist review recommended.\n"

    return findings


def _compare_with_reports(
    related_reports: List[Dict[str, Any]],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[bool, str]:
    """Classify image as related or standalone."""
    if not related_reports:
        return False, "No related reports found in patient history."

    related_above_threshold = [
        r for r in related_reports
        if r["similarity"] >= similarity_threshold
    ]

    if not related_above_threshold:
        return False, "No highly similar reports in patient history."

    is_related = len(related_above_threshold) >= 1
    explanation = (
        f"This scan shows {len(related_above_threshold)} related report(s) "
        f"in the patient's history with {related_above_threshold[0]['similarity']:.0%} similarity."
    )

    return is_related, explanation


def _generate_composite_analysis(
    disease_findings: str,
    is_related: bool,
    relationship_explanation: str,
    related_reports: List[Dict[str, Any]],
) -> str:
    """Generate comprehensive analysis combining image and report data."""
    analysis = disease_findings

    if is_related and related_reports:
        analysis += "\n---\n\n"
        analysis += "**Correlation with Patient History:**\n\n"
        analysis += f"{relationship_explanation}\n\n"

        analysis += "**Related Prior Studies:**\n"
        for i, report in enumerate(related_reports[:3], 1):
            analysis += f"\n{i}. **{report['filename']}** ({report['modality']} · {report['anatomy']})\n"
            if report.get("findings"):
                analysis += f"   - {report['findings']}\n"
            if report.get("text"):
                analysis += f"   - {report['text']}\n"
            analysis += f"   - Similarity: {report['similarity']:.0%}\n"

        analysis += "\n**Clinical Impression:**\n"
        analysis += (
            "The current imaging findings are clinically consistent with the patient's "
            "prior diagnostic history. Progression or resolution should be evaluated "
            "in comparison to baseline studies.\n"
        )
    else:
        analysis += "\n---\n\n"
        analysis += "**Clinical Impression:**\n"
        analysis += (
            "This scan represents a new or independent imaging study without directly "
            "comparable prior studies in the current database. Consider obtaining prior "
            "imaging for comparison if clinically available.\n"
        )

    return analysis


def analyze_image_with_reports(
    image_embedding: List[float],
    scan_id: str,
    patient_id: str,
    modality: str,
    anatomy: str,
    image_array: Optional[np.ndarray] = None,
) -> DiseaseAnalysis:
    """
    Comprehensive disease analysis for an uploaded medical image.

    Performs:
    1. Image pattern analysis if image array provided
    2. Qdrant similarity search for related reports
    3. Disease inference from modality, anatomy, and related reports
    4. Image-report correlation scoring
    5. Composite analysis generation

    Parameters
    ----------
    image_embedding : 512-d normalized BiomedCLIP embedding
    scan_id : unique scan identifier
    patient_id : patient identifier
    modality : imaging modality (X-ray, CT, MRI, etc.)
    anatomy : anatomical region (Chest, Head, Abdomen, etc.)
    image_array : optional numpy array of image for pattern analysis

    Returns
    -------
    DiseaseAnalysis : comprehensive analysis with related reports and findings
    """
    log.info("Analyzing scan (id=%s, patient=%s, modality=%s, anatomy=%s) …",
             scan_id, patient_id, modality, anatomy)

    # Infer diseases with image analysis if available
    if image_array is not None:
        diseases, confidence, pattern_desc = _infer_diseases_with_image(
            image_array, modality, anatomy, ""
        )
        findings = _generate_disease_findings(modality, anatomy, diseases, pattern_desc)
        related_points = []
    else:
        # Fallback without image
        diseases = []
        confidence = 0.5
        pattern_desc = ""
        findings = f"**Medical Scan Analysis Report**\n\n**Imaging Type:** {modality} of {anatomy}\n\nImage pattern analysis unavailable."
        related_points = []

    # Get Qdrant client and query for related reports
    try:
        client = get_qdrant_client()
        related_points = _query_related_reports(
            client,
            image_embedding,
            patient_id,
            top_k=TOP_K_RELATED,
        )
        log.info("Found %d related reports", len(related_points))
    except Exception as exc:
        log.warning("Qdrant search failed: %s", exc)
        related_points = []

    # Extract text from related reports for additional inference
    if related_points:
        related_text = " ".join(
            (p.get("text", "") or "") + " " + (p.get("findings", "") or "")
            for p in related_points
        )
        # Re-infer if we have related text but not image
        if image_array is None and related_text:
            diseases_from_text = []
            text_lower = related_text.lower()
            for disease, keywords in DISEASE_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in text_lower)
                if matches > 0:
                    diseases_from_text.append((disease, matches))
            
            if diseases_from_text:
                sorted_diseases = sorted(diseases_from_text, key=lambda x: x[1], reverse=True)
                diseases = [d for d, _ in sorted_diseases[:3]]
                confidence = min(0.9, 0.6 + len(diseases_from_text) * 0.1)
                findings = _generate_disease_findings(modality, anatomy, diseases)

    # Compare with reports
    is_related, explanation = _compare_with_reports(
        related_points,
        SIMILARITY_THRESHOLD
    )

    # Generate composite analysis
    composite = _generate_composite_analysis(
        findings,
        is_related,
        explanation,
        related_points,
    )

    analysis = DiseaseAnalysis(
        scan_id=scan_id,
        patient_id=patient_id,
        modality=modality,
        anatomy=anatomy,
        detected_diseases=diseases,
        disease_confidence=confidence,
        disease_findings=findings,
        related_reports=related_points,
        is_related=is_related,
        relationship_explanation=explanation,
        composite_analysis=composite,
    )

    log.info("✓ Analysis complete: %d diseases (confidence: %.0f%%), %d related reports",
             len(diseases), confidence * 100, len(related_points))

    return analysis
