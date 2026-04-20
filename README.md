# Med-Sight | Multimodal Medical AI Platform

A comprehensive medical AI platform for processing clinical reports and diagnostic scans using multimodal embeddings and vector search.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Qdrant vector database (local or cloud)
- UV package manager (recommended)

### Environment Setup

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup project**:
   ```bash
   cd /path/to/project
   uv sync  # Installs all dependencies from pyproject.toml
   ```

3. **Configure Qdrant**:
   Create a `.env` file in the project root:
   ```env
   QDRANT_URL=http://localhost:6333  # or your cloud Qdrant URL
   QDRANT_API_KEY=your_api_key_here  # if using cloud Qdrant
   MEDSIGHT_DB_PATH=data/database/medsight_personal.db
   ```

## 🎯 Running the Project

### Option 1: Streamlit Web Interface (Recommended)

**Start the clinical upload portal:**
```bash
uv run streamlit run frontend/app.py
```

This launches a web interface where you can:
- Upload PDF clinical reports
- Upload medical scan images (PNG/JPG/JPEG)
- Process files with automatic metadata extraction
- View analysis results with modality/anatomy detection

**Features:**
- Patient-specific data organization
- Duplicate detection (skips already processed files)
- Real-time processing with progress indicators
- HIPAA-compliant local storage

### Option 2: Command Line Interface

**Ingest individual files:**
```bash
# Process PDF reports
uv run python main.py report path/to/report.pdf --patient-id patient123

# Process medical scans
uv run python main.py scan path/to/scan.png --patient-id patient123 --caption "Chest X-ray"

# Process multiple files at once
uv run python main.py report report1.pdf report2.pdf --patient-id patient123
uv run python main.py scan scan1.png scan2.jpg scan3.jpeg --patient-id patient123
```

**Build public medical corpus:**
```bash
# Index Indiana University and ROCOv2 datasets into Qdrant
uv run python main.py index
```

**Manage patient data:**
```bash
# Delete all data for a specific patient
uv run python main.py delete-patient patient123
```

### Option 3: Programmatic Usage

**Import and use in Python:**
```python
from data.raw_input import ingest_report, ingest_scan
from data_pipeline.qdrant_vdata import ingest_data, search_collection

# Ingest a report
result = ingest_report("path/to/report.pdf", patient_id="patient123")
print(f"Report ingested: {result}")

# Ingest a scan
result = ingest_scan("path/to/scan.png", patient_id="patient123")
print(f"Scan ingested: {result}")

# Search the medical corpus
results = search_collection("chest pain symptoms", patient_id="patient123")
```

## 🏗️ Project Structure

```
├── frontend/app.py              # Streamlit web interface
├── main.py                      # CLI entrypoint
├── data/
│   ├── raw_input/               # Patient data ingestion
│   │   ├── docs_upload.py       # PDF report processing
│   │   └── images_upload.py     # Medical scan processing
│   ├── pipeline/                # Data processing pipelines
│   │   └── qdrant_vdata.py      # Vector database operations
│   ├── database.py              # SQLite utilities
│   ├── qdrant_utils.py          # Qdrant client management
│   ├── metadata.py              # Clinical metadata extraction
│   ├── pdf_utils.py             # PDF parsing utilities
│   └── image_utils.py           # Medical image preprocessing
├── data_pipeline/
│   ├── qdrant_vdata.py          # Public corpus management
│   └── research_papers.py       # Research paper processing
└── agents/                      # AI reasoning agents
    ├── reflector.py
    ├── router.py
    └── vision.py
```

## 🔧 Configuration

### Qdrant Setup

**Local Qdrant (Docker):**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

**Cloud Qdrant:**
- Get URL and API key from [Qdrant Cloud](https://cloud.qdrant.io)
- Update `.env` file with your credentials

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key | None |
| `MEDSIGHT_DB_PATH` | SQLite database path | `data/database/medsight_personal.db` |

## 📊 Data Processing Pipeline

1. **File Upload**: PDF reports or medical images
2. **Preprocessing**: Text extraction, image normalization
3. **Metadata Extraction**: Clinical information parsing
4. **Embedding Generation**: BiomedCLIP multimodal embeddings
5. **Vector Storage**: Qdrant for similarity search
6. **Local Persistence**: SQLite for patient-specific data

## 🛠️ Development

**Run tests:**
```bash
uv run python -m pytest
```

**Code formatting:**
```bash
uv run black .
uv run isort .
```

**Type checking:**
```bash
uv run mypy .
```

## 📈 Features

- **Multimodal Processing**: Text + image medical data
- **Patient Privacy**: Local SQLite storage with encryption
- **Duplicate Detection**: File hash-based deduplication
- **Clinical Metadata**: Automatic modality/anatomy extraction
- **Vector Search**: Semantic similarity across medical content
- **Batch Processing**: Efficient bulk ingestion
- **Web Interface**: User-friendly Streamlit application

## 🔒 Security & Compliance

- End-to-end encrypted data handling
- HIPAA-compliant local storage
- Patient-specific data isolation
- Secure API key management
- No external data transmission without explicit consent</content>
<parameter name="filePath">/home/resumetozero/Documents/Projects/BTP/README.md