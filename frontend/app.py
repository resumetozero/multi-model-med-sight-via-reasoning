import time
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Add project root to Python path to resolve imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ── Project upload pipelines ──────────────────────────────────────────────────
from data.raw_input import ingest_report, ingest_scan

st.set_page_config(
    page_title="Med-Sight | Clinical Upload Portal",
    page_icon="⚕️",
    layout="wide",
)

st.markdown("""
<style>
.stApp                              { background-color: #f3f6f7; }
h1, h2, h3                         { color: #004d40; }
.stApp p, .stApp label, .stApp strong, .stApp li { color: #1f2933; }
[data-testid="stFileUploader"] label{ color: #1a5276; font-weight: 600; }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] div[role="button"],
[data-testid="stFileUploader"] span {
    color: white !important;
}
[data-testid="stFileUploader"] button {
    background-color: #00695c !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] textarea {
    color: #0f2f2f !important;
    background-color: #ffffff !important;
    border: 1px solid #cfd8dc !important;
}
[data-testid="stSidebar"]           { background-color: #eef3f3; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #0f2f2f !important;
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] div[role="button"] {
    color: white !important;
    background-color: #00695c !important;
    border-radius: 8px !important;
}
.stButton > button                  { background-color: #00695c; color: white; border-radius: 8px; }
.upload-box                         { border: 2px dashed #cfd8dc; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://www.gstatic.com/lamda/images/medical_shield_logo.png",
        width=100,
    )
    st.title("Patient Portal")
    st.info(
        "Upload diagnostic scans or clinical reports.\n\n"
        "Supported formats:\n- PNG\n- JPG / JPEG\n- PDF"
    )
    st.divider()

    # Patient ID input — used as the personalisation key in SQLite + Qdrant
    patient_id = st.text_input(
        "Patient ID",
        value="anonymous",
        help="Identifier stored with every uploaded file for personalised retrieval.",
    )

    st.divider()
    st.caption("🔒 End-to-end encrypted & HIPAA compliant environment.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚕️ Med-Sight | Clinical Upload Portal")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    label="Drop patient records or scans here",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Select multiple files using Ctrl / Cmd.",
)


# ─────────────────────────────────────────────────────────────────────────────
# FILE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) staged for analysis.")

    tab1, tab2 = st.tabs(["🖼️ Diagnostic Scans", "📄 Clinical Reports"])

    # ──────────────────────────────────────────────────────────────────────
    # TAB 1 – IMAGES
    # ──────────────────────────────────────────────────────────────────────
    with tab1:

        image_files = [
            f for f in uploaded_files
            if f.type.startswith("image/")
        ]

        if not image_files:
            st.info("No image files detected.")
        else:
            cols      = st.columns(3)
            img_index = 0

            for file in image_files:
                img = Image.open(file)
                with cols[img_index % 3]:
                    st.image(img, caption=file.name, use_container_width=True)

                    # Per-scan analyse button
                    if st.button(f"Analyse {file.name}", key=f"scan_{file.name}"):
                        with st.spinner(f"Processing {file.name} …"):
                            # Save upload to a temp path so the pipeline can read it
                            import tempfile, os
                            suffix = os.path.splitext(file.name)[1]
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=suffix
                            ) as tmp:
                                tmp.write(file.getvalue())
                                tmp_path = tmp.name

                            result = ingest_scan(
                                tmp_path,
                                patient_id=patient_id,
                                caption=file.name,
                            )
                            os.unlink(tmp_path)

                        if result.get("status") == "success":
                            st.success(
                                f"✓ Scan stored  |  "
                                f"Modality: **{result['modality']}**  |  "
                                f"Anatomy: **{result['anatomy']}**"
                            )
                        elif result.get("status") == "already_ingested":
                            st.info("This scan was already ingested.")
                        else:
                            st.error("Ingestion failed – check logs.")

                img_index += 1

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2 – PDFs
    # ──────────────────────────────────────────────────────────────────────
    with tab2:

        pdf_files = [f for f in uploaded_files if f.type == "application/pdf"]

        if not pdf_files:
            st.info("No PDF reports detected.")
        else:
            for file in pdf_files:
                with st.expander(f"📄 {file.name}"):

                    st.write(f"File size: {file.size / 1024:.2f} KB")

                    if st.button(f"Analyse {file.name}", key=f"pdf_{file.name}"):
                        with st.spinner(
                            f"Parsing {file.name} with Docling …"
                        ):
                            import tempfile, os
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                            ) as tmp:
                                tmp.write(file.getvalue())
                                tmp_path = tmp.name

                            result = ingest_report(
                                tmp_path,
                                patient_id=patient_id,
                            )
                            os.unlink(tmp_path)

                        if result.get("status") == "success":
                            st.success(
                                f"✓ Report stored  |  "
                                f"Pages: **{result['page_count']}**  |  "
                                f"Modality: **{result['modality']}**  |  "
                                f"Anatomy: **{result['anatomy']}**  |  "
                                f"Chunks indexed: **{result['chunk_count']}**"
                            )
                        elif result.get("status") == "already_ingested":
                            st.info("This report was already ingested.")
                        else:
                            st.error("Ingestion failed – check logs.")

    # ──────────────────────────────────────────────────────────────────────
    # BULK ACTION – Proceed to Analysis
    # ──────────────────────────────────────────────────────────────────────
    st.divider()

    if st.button("Proceed to Analysis", type="primary", use_container_width=True):
        image_files = [f for f in uploaded_files if f.type.startswith("image/")]
        pdf_files   = [f for f in uploaded_files if f.type == "application/pdf"]

        progress = st.progress(0, text="Starting …")
        total    = len(uploaded_files)
        done     = 0
        errors   = []

        import tempfile, os

        # Ingest all scans
        for file in image_files:
            progress.progress(done / total, text=f"Embedding scan: {file.name}")
            suffix = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            try:
                ingest_scan(tmp_path, patient_id=patient_id, caption=file.name)
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")
            finally:
                os.unlink(tmp_path)
            done += 1

        # Ingest all reports
        for file in pdf_files:
            progress.progress(done / total, text=f"Parsing report: {file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            try:
                ingest_report(tmp_path, patient_id=patient_id)
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")
            finally:
                os.unlink(tmp_path)
            done += 1

        progress.progress(1.0, text="Done")

        if errors:
            st.warning(
                f"Analysis pipeline complete with {len(errors)} error(s):\n"
                + "\n".join(errors)
            )
        else:
            st.success(
                f"✅ All {total} file(s) ingested for patient **{patient_id}**. "
                "Records are now searchable in the Med-Sight AI pipeline."
            )

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.info("Awaiting file upload.\n\nSupported formats: **PDF, JPG, PNG, JPEG**")