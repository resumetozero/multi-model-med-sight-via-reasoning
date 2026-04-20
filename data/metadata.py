from typing import Any
from PIL import Image

_MODALITY_KEYWORDS = {
    "CT": ["computed tomography", " ct ", "ct scan", "ct:", "ct", "computed"],
    "MRI": ["magnetic resonance", " mri ", "mri:", "nmr", "mri"],
    "X-ray": ["x-ray", "radiograph", "chest pa", "ap view", "xray", "cxr"],
    "Ultrasound": ["ultrasound", "sonography", "echography", " us ", "us_", "echo"],
    "PET": ["pet scan", "positron emission", "nuclear medicine", "pet"],
    "Pathology": ["histopathology", "biopsy", "cytology", "pathology"],
}

_ANATOMY_KEYWORDS = {
    "Chest": ["chest", "lung", "pulmonary", "pleural", "thorax", "bronchi", "trachea", "cardiac", "heart"],
    "Head": ["brain", "cerebral", "cranial", "skull", "head", "intracranial", "meninges", "ventricle"],
    "Abdomen": ["abdomen", "liver", "hepatic", "renal", "kidney", "spleen", "pancreas", "gallbladder", "pelvis", "bowel", "colon", "rectum"],
    "Musculoskeletal": ["spine", "vertebr", "disc", "lumbar", "cervical", "bone", "fracture", "joint", "femur", "tibia", "humer", "shoulder", "knee", "hip"],
    "Breast": ["breast", "mammograph", "mammo"],
    "Pelvis": ["prostate", "uterus", "ovarian", "testicular"],
}


def _contains_keywords(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_modality(text: str, default: str = "Unknown") -> str:
    normalized = text.lower()
    for label, keywords in _MODALITY_KEYWORDS.items():
        if _contains_keywords(normalized, keywords):
            return label
    return default


def infer_anatomy(text: str, default: str = "General") -> str:
    normalized = text.lower()
    for label, keywords in _ANATOMY_KEYWORDS.items():
        if _contains_keywords(normalized, keywords):
            return label
    return default


def summarize_text(text: str, min_length: int = 20) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= min_length:
            return stripped[:200]
    return text.strip()[:200]


def extract_report_metadata(text: str) -> dict[str, str]:
    return {
        "modality": infer_modality(text),
        "anatomy": infer_anatomy(text),
        "summary": summarize_text(text),
    }


def extract_scan_metadata(
    image: Image.Image,
    filename: str,
    caption: str = "",
) -> dict[str, Any]:
    text = (caption + " " + filename).lower()
    meta = {
        "modality": infer_modality(text),
        "anatomy": infer_anatomy(text),
        "color_mode": image.mode,
        "width": image.width,
        "height": image.height,
    }

    if meta["modality"] == "Unknown" and image.mode in ("L", "I"):
        meta["modality"] = "X-ray"

    return meta


def extract_clinical_metadata(text: str) -> dict[str, str]:
    return {
        "modality": infer_modality(text, default="Other"),
        "anatomy": infer_anatomy(text, default="General"),
    }
