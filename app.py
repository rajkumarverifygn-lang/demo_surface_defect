import os
# Prevent ultralytics/cv2 from trying to use GUI display libs
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["DISPLAY"] = ""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.express as plotly_express

# ==========================================
# 1. PAGE CONFIGURATION & INDUSTRIAL THEME
# ==========================================

st.set_page_config(
    page_title="Defect Detection System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Hide all Streamlit chrome ── */
#MainMenu                        { display: none !important; }
footer                           { display: none !important; }
header                           { display: none !important; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="stStatusWidget"]   { display: none !important; }
[data-testid="manage-app-button"]{ display: none !important; }
.viewerBadge_container__1QSob   { display: none !important; }
.styles_viewerBadge__1yB5_      { display: none !important; }
#stDecoration                    { display: none !important; }
[data-testid="stSidebarNav"]     { display: none !important; }
.viewerBadge_link__qRIco        { display: none !important; }
#badge-button                    { display: none !important; }
[data-testid="baseButton-badge"] { display: none !important; }
a[href="https://streamlit.io"]   { display: none !important; }
.st-emotion-cache-1wbqy5l        { display: none !important; }
div[class*="viewerBadge"]        { display: none !important; }
div[class*="badge"]              { display: none !important; }

/* ── App theme ── */
.main { background-color: #F8F9FA; color: #212529; }
h1, h2, h3, h4 { color: #212529; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600; }
.stButton>button { background-color: #FF8C00; color: white; font-weight: bold; border-radius: 4px; border: none; padding: 10px 24px; }
.stButton>button:hover { background-color: #CC7000; color: white; }
.status-ok  { background-color: #E8F5E9; border: 2px solid #4CAF50; color: #2E7D32; padding: 20px; text-align: center; border-radius: 8px; font-size: 48px; font-weight: bold; }
.status-nok { background-color: #FFEBEE; border: 2px solid #F44336; color: #C62828; padding: 20px; text-align: center; border-radius: 8px; font-size: 48px; font-weight: bold; }
div[data-testid="stMetricValue"] { font-size: 32px; color: #FF8C00; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================

for key, default in [
    ('total_parts', 0),
    ('ok_count', 0),
    ('nok_count', 0),
    ('last_processed_id', None),
    ('current_processed_image', None),
    ('current_raw_image', None),
    ('current_is_nok', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 3. HELPER FUNCTIONS (PIL-ONLY, NO CV2)
# ==========================================

@st.cache_resource
def load_model(model_path):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def draw_boxes_pil(image_pil, boxes_data):
    """Draw bounding boxes on a PIL image without using cv2."""
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18
        )
    except Exception:
        font = ImageFont.load_default()

    color_map = {"blowhole": "#F44336"}
    default_color = "#FF8C00"

    for box in boxes_data:
        x1, y1, x2, y2 = box["xyxy"]
        label = box["label"]
        conf = box["conf"]
        color = color_map.get(label.lower(), default_color)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill=color)
        draw.text((x1, y1), text, fill="white", font=font)

    return image_pil


def process_image(image_pil, model):
    """Run YOLO inference and draw results using PIL only — no cv2."""
    original_w, original_h = image_pil.size

    # Resize to 640x640 using PIL (no cv2)
    img_640 = image_pil.resize((640, 640), Image.Resampling.BILINEAR)
    img_array = np.array(img_640)  # RGB numpy array — ultralytics accepts this

    results = model(img_array)

    is_nok = False
    names = model.names
    boxes_data = []

    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            label = names[cls_idx]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Scale coords from 640x640 back to original size
            scale_x = original_w / 640
            scale_y = original_h / 640
            boxes_data.append({
                "xyxy": (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                "label": label,
                "conf": conf,
            })

            if label.lower() == "blowhole":
                is_nok = True

    annotated = image_pil.copy()
    if boxes_data:
        annotated = draw_boxes_pil(annotated, boxes_data)

    return annotated, is_nok

# ==========================================
# 4. HEADER LAYOUT
# ==========================================

header_col1, header_col2, header_col3 = st.columns([1, 4, 1])

with header_col1:
    if os.path.exists("VFN_logo.png"):
        st.image("VFN_logo.png", width=150)
    else:
        st.write("VFN Logo")

with header_col2:
    st.markdown(
        "<h1 style='text-align: center;'>🏭 SURFACE DEFECT DETECTION</h1>",
        unsafe_allow_html=True
    )

with header_col3:
    if os.path.exists("JAYAHIND_logo.png"):
        st.image("JAYAHIND_logo.png", width=150)
    else:
        st.write("Jayahind Logo")

st.divider()

# ==========================================
# 5. LOAD MODEL
# ==========================================

MODEL_PATH = "weights.pt"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.warning(f"Model file '{MODEL_PATH}' not found. Using standard yolov8n.pt.")
    model = load_model("yolov8n.pt")

if model is None:
    st.stop()

# ==========================================
# 6. CORE LOGIC
# ==========================================

col_control, col_stats = st.columns([1, 1])

with col_control:
    st.subheader("Image Acquisition")
    source_type = st.radio("Select Input Source:", ("Live Camera", "Local Disk"))

    raw_image_pil = None
    current_image_id = None

    if source_type == "Live Camera":
        camera_img = st.camera_input("Capture Part")
        if camera_img is not None:
            raw_image_pil = Image.open(camera_img).convert("RGB")
            current_image_id = camera_img.file_id

    elif source_type == "Local Disk":
        uploaded_file = st.file_uploader(
            "Load Image from Disk", type=['jpg', 'jpeg', 'png', 'bmp']
        )
        if uploaded_file is not None:
            raw_image_pil = Image.open(uploaded_file).convert("RGB")
            current_image_id = uploaded_file.file_id

    if raw_image_pil is not None and current_image_id is not None:
        if st.session_state.last_processed_id != current_image_id:
            with st.spinner('Analyzing Part...'):
                processed_image_pil, is_nok = process_image(raw_image_pil, model)

            st.session_state.total_parts += 1
            if is_nok:
                st.session_state.nok_count += 1
            else:
                st.session_state.ok_count += 1

            st.session_state.last_processed_id = current_image_id
            st.session_state.current_raw_image = raw_image_pil
            st.session_state.current_processed_image = processed_image_pil
            st.session_state.current_is_nok = is_nok

with col_stats:
    st.subheader("Yield Analysis")
    labels = ['OK Parts', 'NOK Parts']
    values = [st.session_state.ok_count, st.session_state.nok_count]

    if st.session_state.total_parts > 0:
        fig = plotly_express.pie(
            names=labels,
            values=values,
            color=labels,
            color_discrete_map={'OK Parts': '#4CAF50', 'NOK Parts': '#F44336'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#000000"),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting first part inspection to generate yield chart.")

    if st.button("Reset Shift Data"):
        st.session_state.total_parts = 0
        st.session_state.ok_count = 0
        st.session_state.nok_count = 0
        st.session_state.last_processed_id = None
        st.session_state.current_raw_image = None
        st.session_state.current_processed_image = None
        st.rerun()

# ==========================================
# 7. RESULTS DISPLAY
# ==========================================

if st.session_state.current_processed_image is not None:
    st.divider()
    st.subheader(f"Inspection Result - Part #{st.session_state.total_parts}")

    if st.session_state.current_is_nok:
        st.markdown(
            '<div class="status-nok">❌ NOK (BLOWHOLE DETECTED)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-ok">✅ OK (PART ACCEPTED)</div>',
            unsafe_allow_html=True
        )

    st.write("")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown(
            "<h4 style='text-align: center;'>Raw Image</h4>",
            unsafe_allow_html=True
        )
        st.image(st.session_state.current_raw_image, use_column_width=True)

    with img_col2:
        st.markdown(
            "<h4 style='text-align: center;'>Detected Image</h4>",
            unsafe_allow_html=True
        )
        st.image(st.session_state.current_processed_image, use_column_width=True)
