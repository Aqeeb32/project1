import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(page_title="AI Image Detective Pro", page_icon="📸", layout="centered")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

st.title("📸 AI Image Detective Pro")
st.write("Upload a file or use your live camera to let the Computer Vision engine detect objects accurately.")

with st.spinner("Initializing Upgraded Vision Engine..."):
    model = load_model()

st.divider()

# --- SIDEBAR: Configuration & Controls ---
st.sidebar.header("⚙️ App Settings")

# Feature 1: Confidence Threshold Slider
threshold = st.sidebar.slider(
    "Minimum Confidence Threshold (%)", 
    min_value=5, 
    max_value=100, 
    value=20, 
    step=5
)
st.sidebar.caption("Is percentage se kam confidence wali predictions filter out ho jayengi.")

# --- MAIN UI: Input Selection ---
input_mode = st.radio("Choose Input Method:", ["📁 Upload Image File", "📷 Use Live Camera"])

img_source = None

if input_mode == "📁 Upload Image File":
    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_source = Image.open(uploaded_file)
else:
    # Feature 2: Live Camera Integration
    camera_file = st.camera_input("Take a snapshot")
    if camera_file:
        img_source = Image.open(camera_file)

# --- Processing Pipeline ---
if img_source is not None:
    # Display selected/captured image
    st.image(img_source, caption="Target Image", use_container_width=True)
    
    if st.button("🔍 Run AI Analysis", use_container_width=True):
        with st.spinner("Analyzing pixels and calculating probabilities..."):
            
            # 1. Image Preprocessing (MobileNetV2 requirements)
            img_resized = img_source.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # 2. Model Prediction
            predictions = model.predict(img_array)
            decoded_results = decode_predictions(predictions, top=5)[0] # Get top 5 candidates
            
            # 3. Filtering via Threshold Slider
            filtered_results = []
            for res in decoded_results:
                conf_percentage = float(res[2]) * 100
                if conf_percentage >= threshold:
                    filtered_results.append({
                        "Object": res[1].replace("_", " ").title(),
                        "Confidence (%)": round(conf_percentage, 2)
                    })
            
            # 4. Display Results
            st.divider()
            if len(filtered_results) == 0:
                st.warning(f"Model ko is image me {threshold}% se zyada confidence ka koi object nahi mila. Settings se threshold kam kar ke dekhein.")
            else:
                st.success("Analysis Complete!")
                st.subheader("📊 Detected Entities:")
                
                # Display individual progress bars
                for item in filtered_results:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**{item['Object']}**")
                    with col2:
                        st.progress(item["Confidence (%)"] / 100, text=f"{item['Confidence (%)']}%")
                
                # Feature 3: Downloadable Data Report
                st.write("")
                df = pd.DataFrame(filtered_results)
                csv_data = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Prediction Report (CSV)",
                    data=csv_data,
                    file_name="ai_detection_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

st.divider()
st.caption("Engine: TensorFlow MobileNetV2 | UI: Streamlit Pro Layout")
