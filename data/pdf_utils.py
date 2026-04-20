from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def _build_converter() -> DocumentConverter:
    # Use default configuration for now to avoid API compatibility issues
    return DocumentConverter()


def parse_pdf(pdf_path: str) -> dict:
    converter = _build_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    full_text = doc.export_to_markdown()
    page_count = len(doc.pages) if hasattr(doc, "pages") else 0

    tables = []
    for t_idx, table in enumerate(doc.tables):
        tables.append({
            "page": getattr(table, "page_no", 0),
            "index": t_idx,
            "text": table.export_to_markdown(),
        })

    sections, current = [], []
    for line in full_text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    return {
        "full_text": full_text,
        "page_count": page_count,
        "tables": tables,
        "sections": sections,
    }


def chunk_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [chunk for chunk in chunks if len(chunk) > 30]
