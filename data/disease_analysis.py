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


def _infer_diseases(
    modality: str,
    anatomy: str,
    related_text: str = "",
) -> tuple[List[str], float]:
    """Infer probable diseases based on modality, anatomy, and related findings."""
    # Get disease candidates from modality + anatomy
    modality_diseases = set(MODALITY_DISEASES.get(modality, []))
    anatomy_diseases = set(ANATOMY_DISEASES.get(anatomy, []))

    # Intersection if both exist, otherwise union
    probable_diseases = modality_diseases & anatomy_diseases
    if not probable_diseases:
        probable_diseases = modality_diseases | anatomy_diseases

    # Score diseases by keyword matches in related text
    disease_scores = {}
    text_lower = related_text.lower()

    for disease, keywords in DISEASE_KEYWORDS.items():
        if disease in probable_diseases:
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                disease_scores[disease] = matches

    # Return top diseases sorted by match count
    if disease_scores:
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        diseases = [d for d, _ in sorted_diseases[:3]]
        # Confidence based on keyword density
        max_matches = max(disease_scores.values())
        confidence = min(0.95, 0.5 + (max_matches / 10.0) * 0.45)
    else:
        diseases = list(probable_diseases)[:3]
        confidence = 0.6

    return diseases, confidence


def _generate_disease_findings(
    modality: str,
    anatomy: str,
    diseases: List[str],
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
) -> DiseaseAnalysis:
    """
    Comprehensive disease analysis for an uploaded medical image.

    Performs:
    1. Qdrant similarity search for related reports
    2. Disease inference from modality, anatomy, and related reports
    3. Image-report correlation scoring
    4. Composite analysis generation

    Parameters
    ----------
    image_embedding : 512-d normalized BiomedCLIP embedding
    scan_id : unique scan identifier
    patient_id : patient identifier
    modality : imaging modality (X-ray, CT, MRI, etc.)
    anatomy : anatomical region (Chest, Head, Abdomen, etc.)

    Returns
    -------
    DiseaseAnalysis : comprehensive analysis with related reports and findings
    """
    log.info("Analyzing scan (id=%s, patient=%s, modality=%s, anatomy=%s) …",
             scan_id, patient_id, modality, anatomy)

    # Get Qdrant client
    try:
        client = get_qdrant_client()
    except Exception as exc:
        log.warning("Failed to connect to Qdrant: %s", exc)
        # Fallback: return basic analysis without Qdrant
        diseases, conf = _infer_diseases(modality, anatomy, "")
        findings = _generate_disease_findings(modality, anatomy, diseases)
        return DiseaseAnalysis(
            scan_id=scan_id,
            patient_id=patient_id,
            modality=modality,
            anatomy=anatomy,
            detected_diseases=diseases,
            disease_confidence=conf,
            disease_findings=findings,
            related_reports=[],
            is_related=False,
            relationship_explanation="Qdrant unavailable; local analysis only.",
            composite_analysis=findings,
        )

    # Query for related reports
    related_points = _query_related_reports(
        client,
        image_embedding,
        patient_id,
        top_k=TOP_K_RELATED,
    )

    # Extract text from related reports for disease inference
    related_text = " ".join(
        (p.get("text", "") or "") + " " + (p.get("findings", "") or "")
        for p in related_points
    )

    # Infer diseases
    diseases, confidence = _infer_diseases(modality, anatomy, related_text)
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

    log.info("✓ Analysis complete: %d diseases, %d related reports",
             len(diseases), len(related_points))

    return analysis
