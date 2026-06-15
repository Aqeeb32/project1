import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="Precision Vision AI", page_icon="🎯", layout="centered")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    # YOLOv8 Nano model (yolov8n.pt) is incredibly light (~6MB) and fast
    # It will automatically download the first time you run it
    return YOLO('yolov8n.pt')

st.title("🎯 Precision AI Object Detector")
st.write("Yeh engine poori image ko scan kar ke exact objects locate karta hai aur unke gird boxes banata hai.")

with st.spinner("Initializing YOLOv8 Engine..."):
    model = load_model()

st.divider()

# --- INPUT UI ---
input_mode = st.radio("Choose Input Method:", ["📁 Upload Image", "📷 Live Camera"])

img_source = None

if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_source = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a snapshot")
    if camera_file:
        img_source = Image.open(camera_file)

# --- PROCESSING ---
if img_source is not None:
    st.subheader("Original Image:")
    st.image(img_source, use_container_width=True)
    
    # Confidence Slider
    conf_threshold = st.slider("Detection Confidence Threshold (%)", 5, 100, 25) / 100.0
    
    if st.button("🚀 Run Precision Detection", use_container_width=True):
        with st.spinner("Detecting and drawing bounding boxes..."):
            
            # YOLO expects RGB images (PIL is RGB by default)
            # Run inference directly
            results = model.predict(source=img_source, conf=conf_threshold)
            
            # Get the plotted image array from results (Draws boxes & labels automatically)
            annotated_img = results[0].plot()
            
            st.divider()
            st.success("Detection Complete!")
            st.subheader("🔍 AI Detection Results:")
            
            # Display the annotated image (YOLO's plot outputs BGR, so we specify channels)
            st.image(annotated_img, channels="BGR", use_container_width=True)
            
            # Show detected items in a clean list
            detected_items = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_items.append(class_name.title())
            
            if detected_items:
                st.write("**Objects Found:**")
                # Count frequencies of detected items
                counts = {i:detected_items.count(i) for i in set(detected_items)}
                for item, count in counts.items():
                    st.write(f"✅ {count}x {item}")
            else:
                st.warning("Koi specific object detect nahi hua. Try lowering the threshold or use a clearer image.")

st.divider()
st.caption("Powered by Ultralytics YOLOv8 & Streamlit")
