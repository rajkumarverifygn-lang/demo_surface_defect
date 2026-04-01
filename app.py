import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import plotly.express as plotly_express
import os

# ==========================================
# 1. PAGE CONFIGURATION & INDUSTRIAL THEME
# ==========================================
st.set_page_config(
    page_title="Defect Detection System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Light Industrial UI Look
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; color: #212529; }
    h1, h2, h3, h4 { color: #212529; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600; }
    .stButton>button { background-color: #FF8C00; color: white; font-weight: bold; border-radius: 4px; border: none; padding: 10px 24px; }
    .stButton>button:hover { background-color: #CC7000; color: white; }
    .status-ok { background-color: #E8F5E9; border: 2px solid #4CAF50; color: #2E7D32; padding: 20px; text-align: center; border-radius: 8px; font-size: 48px; font-weight: bold; }
    .status-nok { background-color: #FFEBEE; border: 2px solid #F44336; color: #C62828; padding: 20px; text-align: center; border-radius: 8px; font-size: 48px; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 32px; color: #FF8C00; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST SESSION STATE INITIALIZATION
# ==========================================
if 'total_parts' not in st.session_state:
    st.session_state.total_parts = 0
if 'ok_count' not in st.session_state:
    st.session_state.ok_count = 0
if 'nok_count' not in st.session_state:
    st.session_state.nok_count = 0
if 'last_processed_id' not in st.session_state:
    st.session_state.last_processed_id = None
if 'current_processed_image' not in st.session_state:
    st.session_state.current_processed_image = None
if 'current_raw_image' not in st.session_state:
    st.session_state.current_raw_image = None
if 'current_is_nok' not in st.session_state:
    st.session_state.current_is_nok = False

# ==========================================
# 3. HELPER FUNCTIONS (NO OPENCV)
# ==========================================
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(image_pil, model):
    # PIL size returns (width, height)
    original_w, original_h = image_pil.size 
    
    # Resize using PIL instead of cv2
    img_640 = image_pil.resize((640, 640), Image.Resampling.BILINEAR)
    
    # YOLO can accept PIL images directly
    results = model(img_640)
    
    is_nok = False
    blowhole_class_name = "blowhole" 
    
    names = model.names
    for r in results:
        for c in r.boxes.cls:
            if names[int(c)].lower() == blowhole_class_name:
                is_nok = True
                break
                
    # results[0].plot() returns a NumPy array in BGR format (YOLO default)
    annotated_bgr_np = results[0].plot()
    
    # Convert BGR to RGB using pure NumPy slicing (no cv2 required!)
    annotated_rgb_np = annotated_bgr_np[..., ::-1]
    
    # Convert back to PIL Image and resize to original dimensions
    annotated_pil = Image.fromarray(annotated_rgb_np)
    final_img = annotated_pil.resize((original_w, original_h), Image.Resampling.BILINEAR)
    
    return final_img, is_nok

# ==========================================
# 4. HEADER LAYOUT
# ==========================================
header_col1, header_col2, header_col3 = st.columns([1, 4, 1])

with header_col1:
    st.image("VFN_logo.png", width=150)

with header_col2:
    st.markdown("<h1 style='text-align: center;'>🏭 SURFACE DEFECT DETECTION</h1>", unsafe_allow_html=True)

with header_col3:
    st.image("JAYAHIND_logo.png", width=150)

st.divider()

# Load the Model
MODEL_PATH = "weights.pt" 
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.warning(f"Model file '{MODEL_PATH}' not found. Using yolov8n.pt for demonstration.")
    model = load_model("yolov8n.pt")

# ==========================================
# 5. CORE LOGIC: ACQUIRE -> PROCESS -> DISPLAY
# ==========================================
col_control, col_stats = st.columns([1, 1])

# Step A: Image Acquisition
with col_control:
    st.subheader("Image Acquisition")
    source_type = st.radio("Select Input Source:", ("Live Camera", "Local Disk"))
    
    raw_image_pil = None
    current_image_id = None
    
    if source_type == "Live Camera":
        camera_img = st.camera_input("Capture Part")
        if camera_img is not None:
            raw_image_pil = Image.open(camera_img).convert("RGB")
            current_image_id = camera_img.file_id # Get unique ID
            
    elif source_type == "Local Disk":
        uploaded_file = st.file_uploader("Load Image from Disk", type=['jpg', 'jpeg', 'png', 'bmp'])
        if uploaded_file is not None:
            raw_image_pil = Image.open(uploaded_file).convert("RGB")
            current_image_id = uploaded_file.file_id # Get unique ID

# Step B: Immediate Image Processing & State Update
if raw_image_pil is not None and current_image_id is not None:
    # ONLY process if this is a brand new image we haven't seen yet
    if st.session_state.last_processed_id != current_image_id:
        
        with st.spinner('Analyzing Part...'):
            # Pass PIL image directly to the function
            processed_image_pil, is_nok = process_image(raw_image_pil, model)
            
        # Update Counts immediately
        st.session_state.total_parts += 1
        if is_nok:
            st.session_state.nok_count += 1
        else:
            st.session_state.ok_count += 1
            
        # Save results to session state
        st.session_state.last_processed_id = current_image_id
        st.session_state.current_raw_image = raw_image_pil
        st.session_state.current_processed_image = processed_image_pil
        st.session_state.current_is_nok = is_nok

# Step C: Yield Analysis
with col_stats:
    st.subheader("Yield Analysis")
    labels = ['OK Parts', 'NOK Parts']
    values = [st.session_state.ok_count, st.session_state.nok_count]
    
    if st.session_state.total_parts > 0:
        fig = plotly_express.pie(
            names=labels, 
            values=values, 
            color=labels,
            color_discrete_map={'OK Parts':'#4CAF50', 'NOK Parts':'#F44336'},
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
# 6. RESULTS DISPLAY
# ==========================================
if st.session_state.current_processed_image is not None:
    st.divider()
    
    st.subheader(f"Inspection Result - Part #{st.session_state.total_parts}")
    if st.session_state.current_is_nok:
        st.markdown('<div class="status-nok">❌ NOK (BLOWHOLE DETECTED)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-ok">✅ OK (PART ACCEPTED)</div>', unsafe_allow_html=True)
        
    st.write("")
    
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.markdown("<h4 style='text-align: center;'>Raw Image</h4>", unsafe_allow_html=True)
        st.image(st.session_state.current_raw_image, use_column_width=True)
        
    with img_col2:
        st.markdown("<h4 style='text-align: center;'>Detected Image</h4>", unsafe_allow_html=True)
        st.image(st.session_state.current_processed_image, use_column_width=True)