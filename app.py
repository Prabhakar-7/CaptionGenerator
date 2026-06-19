# ==============================================================================
# STREAMLIT PRODUCTION APPLICATION (app.py)
# ==============================================================================
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Page Setup & Configurations ---
st.set_page_config(
    page_title="AI Caption Generator",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Automated Image Caption Generator")
st.markdown("Upload any image to automatically generate structured captions using OpenCV and BLIP transformers.")

# --- Cached Model Loader (Prevents reloading on every UI interaction) ---
@st.cache_resource
def load_vision_models():
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    # Force CPU for stable, low-cost cloud deployment (e.g., Streamlit Community Cloud)
    device = torch.device("cpu")
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model, device

try:
    with st.spinner("Loading Vision-Language AI Models into memory (this may take a minute on first boot)..."):
        processor, model, device = load_vision_models()
    st.sidebar.success("🤖 Neural Engine Core: Connected")
except Exception as e:
    st.error(f"Failed to allocate model structures: {e}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Configuration Panel")
style_mode = st.sidebar.selectbox(
    "Caption Refinement Style",
    options=["Minimalist", "Verbose", "Social Media", "Journalistic"]
)

# --- Post-Processing / Formatting Rules ---
def apply_style(raw_caption: str, style: str) -> str:
    clean = raw_caption.capitalize()
    if style == "Verbose":
        return f"A comprehensive scene analysis captures {clean[0].lower()}{clean[1:]}."
    elif style == "Social Media":
        return f"✨ Capture the moment: {clean} #AIVision #DeepLearning"
    elif style == "Journalistic":
        return f"Observed Environment Log: {clean}."
    return clean

# --- File Ingestion UI Component ---
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # 1. Read bytes directly from memory into a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_mat = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 2. Display Image Preview
    rgb_mat = cv2.cvtColor(bgr_mat, cv2.COLOR_BGR2RGB)
    st.image(rgb_mat, caption="Uploaded Image Source Frame", use_container_width=True)
    
    # 3. Trigger Caption Generation Call
    if st.button("⚡ Generate AI Caption", type="primary"):
        with st.spinner("Processing image pixels and decoding semantic tokens..."):
            # Normalize and resize using OpenCV for model format compatibility (384x384)
            resized_rgb = cv2.resize(rgb_mat, (384, 384), interpolation=cv2.INTER_CUBIC)
            pil_image = Image.fromarray(resized_rgb)
            
            # Formulate inputs and move tensors to device memory
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_length=45,
                    min_length=10,
                    num_beams=5,
                    repetition_penalty=1.2
                )
            
            raw_caption = processor.decode(output_tokens[0], skip_special_tokens=True).strip()
            final_caption = apply_style(raw_caption, style_mode)
            
        # --- UI Output Metrics Display ---
        st.success("Analysis Complete!")
        
        st.subheader("Results Dashboard")
        st.markdown(f"**Raw AI Prediction:** *\"{raw_caption}\"*")
        st.info(f"**Refined Output ({style_mode} Style):** \n\n {final_caption}")