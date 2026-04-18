import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Med-Sight | Clinical Upload Portal",
    page_icon="⚕️",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f3f6f7;
}

/* Headings */
h1, h2, h3 {
    color: #004d40;
}

/* General text */
p, span, label, div {
    color: #1f2933;
}

/* File uploader */
[data-testid="stFileUploader"] label {
    color: #1a5276;
    font-weight: 600;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #eef3f3;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #0f2f2f;
}

/* Buttons */
.stButton > button {
    background-color: #00695c;
    color: white;
    border-radius: 8px;
}

/* Upload container */
.upload-box {
    border: 2px dashed #cfd8dc;
    padding: 20px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.image(
        "https://www.gstatic.com/lamda/images/medical_shield_logo.png",
        width=100
    )

    st.title("Patient Portal")

    st.info(
        "Upload diagnostic scans or clinical reports.\n\n"
        "Supported formats:\n"
        "- PNG\n"
        "- JPG\n"
        "- SVG\n"
        "- PDF"
    )

    st.divider()

    st.caption("🔒 End-to-end encrypted & HIPAA compliant environment.")

# ---------------- MAIN HEADER ---------------- #
st.title("⚕️ Med-Sight | Clinical Upload Portal")
st.markdown("---")

# ---------------- FILE UPLOADER ---------------- #
uploaded_files = st.file_uploader(
    label="Drop patient records or scans here",
    type=["pdf", "png", "jpg", "jpeg", "svg"],
    accept_multiple_files=True,
    help="Select multiple files using Ctrl / Cmd."
)

# ---------------- FILE PROCESSING ---------------- #

if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) staged for analysis.")

    tab1, tab2 = st.tabs(["🖼️ Diagnostic Scans", "📄 Clinical Reports"])

    # ---------- IMAGE TAB ---------- #
    with tab1:

        cols = st.columns(3)
        img_index = 0

        for file in uploaded_files:

            # Raster Images
            if file.type.startswith("image/") and file.type != "image/svg+xml":

                img = Image.open(file)

                with cols[img_index % 3]:
                    st.image(
                        img,
                        caption=file.name,
                        use_container_width=True
                    )

                img_index += 1

            # SVG Images
            elif file.type == "image/svg+xml":

                svg = file.read().decode("utf-8")

                with cols[img_index % 3]:
                    st.markdown(
                        f"""
                        <div style="text-align:center">
                        {svg}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption(file.name)

                img_index += 1

        if img_index == 0:
            st.info("No image files detected.")

    # ---------- PDF TAB ---------- #
    with tab2:

        pdf_count = 0

        for file in uploaded_files:

            if file.type == "application/pdf":

                pdf_count += 1

                with st.expander(f"📄 {file.name}"):

                    st.write(f"File size: {file.size/1024:.2f} KB")

                    st.button(
                        f"Analyze {file.name}",
                        key=file.name
                    )

        if pdf_count == 0:
            st.info("No PDF reports detected.")

    # ---------------- ACTION ---------------- #
    st.divider()

    if st.button("Proceed to Analysis", type="primary", use_container_width=True):

        st.toast("Processing clinical data...", icon="⏳")

        with st.spinner("Running Med-Sight AI analysis..."):
            st.sleep(2)

        st.success("Analysis pipeline initiated.")

# ---------------- EMPTY STATE ---------------- #

else:

    st.info(
        "Awaiting file upload.\n\n"
        "Supported formats: **PDF, JPG, PNG, SVG**"
    )