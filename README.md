# Multi-Model Med-Sight via Reasoning

A comprehensive multimodal medical AI platform that leverages advanced reasoning agents to process clinical reports and diagnostic scans using multimodal embeddings and vector search. The system integrates vision, reflection, and routing agents to provide intelligent analysis of medical data with semantic correlation and disease detection.

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

## 🤖 AI Agents Architecture

The platform employs a multi-agent reasoning system:

- **Vision Agent** (`agents/vision.py`): Processes medical images using BiomedCLIP for feature extraction and initial analysis
- **Reflector Agent** (`agents/reflector.py`): Performs reflective reasoning on analysis results for deeper insights
- **Router Agent** (`agents/router.py`): Coordinates between agents and manages workflow orchestration

## 📊 Quantitative Metrics

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
3. **Metadata Extraction**: Clinical information parsing (modality, anatomy)
4. **Embedding Generation**: BiomedCLIP multimodal embeddings (512-d vectors)
5. **Vector Storage**: Qdrant for similarity search and patient correlation
6. **AI Analysis**: Multi-agent reasoning for disease detection and clinical insights
7. **Local Persistence**: SQLite for patient-specific data with encryption

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

- **Multimodal Processing**: Text + image medical data integration
- **Patient Privacy**: Local SQLite storage with encryption
- **Duplicate Detection**: File hash-based deduplication
- **Clinical Metadata**: Automatic modality/anatomy extraction
- **Vector Search**: Semantic similarity across medical content
- **AI Reasoning Agents**: Advanced analysis with reflection and routing
- **Disease Detection**: Pattern-based disease identification with confidence scoring
- **Batch Processing**: Efficient bulk ingestion
- **Web Interface**: User-friendly Streamlit application
- **Research Integration**: Latest medical research incorporation via web search

## 🔒 Security & Compliance

- End-to-end encrypted data handling
- HIPAA-compliant local storage
- Patient-specific data isolation
- Secure API key management
- No external data transmission without explicit consent

## 📦 Dependencies

### Key Libraries
- **Streamlit 1.55.0+**: Web interface framework
- **Docling 2.90.0+**: Advanced PDF parsing with OCR
- **Qdrant Client 1.17.1+**: Vector database client
- **Open-CLIP-Torch 3.3.0+**: Multimodal embeddings (BiomedCLIP)
- **PyTorch 2.0+**: Deep learning framework
- **LangChain 1.2+**: LLM orchestration and agent framework
- **LangGraph 1.1+**: Multi-agent workflow management

### Installation Methods

**Recommended - Using UV:**
```bash
uv sync
```

**Alternative - Using pip with requirements.txt:**
```bash
pip install -r requirements.txt
```

## 🐛 Troubleshooting

### Import Errors
If you encounter `ModuleNotFoundError`, ensure:
- You're using `uv run` to execute commands
- The virtual environment has all dependencies: `uv sync`
- The project root is in your Python path

### Qdrant Connection Issues
- Verify Qdrant is running: `docker ps` for local instances
- Check credentials in `.env` for cloud deployments
- Ensure firewall allows ports 6333-6334

### PDF Parsing Failures
- Docling requires system libraries: `apt-get install poppler-utils`
- Scanned PDFs may require OCR processing (automatic)
- Large PDFs stream in batches; check available memory

### Database Locked
- If `*.db-wal` or `*.db-shm` files exist, SQLite is in use
- Close all open connections: `lsof data/database/medsight_personal.db`
- Clear stale locks if needed: `rm data/database/*.db-wal data/database/*.db-shm`

## 📝 License

This project supports clinical AI research and HIPAA-compliant medical data processing.</content>
<parameter name="filePath">/home/resumetozero/Documents/Projects/BTP/README.md