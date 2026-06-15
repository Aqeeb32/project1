import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Image Detective", page_icon="📸", layout="centered")

# --- Model Loading (Cached to save RAM and time) ---
@st.cache_resource
def load_model():
    # Loading the pre-trained MobileNetV2 model (Lightweight and highly accurate)
    model = MobileNetV2(weights='imagenet')
    return model

st.title("📸 AI Image Detective")
st.write("Koi bhi picture upload karein, aur AI accurately bata dega ke usme kya hai!")

# Load model with spinner
with st.spinner("Initializing Computer Vision Engine..."):
    model = load_model()

st.divider()

# --- Image Upload UI ---
uploaded_file = st.file_uploader("Upload an Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Aapki Upload ki gayi Image", use_column_width=True)
    
    # Process Button
    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("Analyzing pixels..."):
            # 1. Resize image to 224x224 (Required by MobileNetV2)
            img_resized = img.resize((224, 224))
            
            # 2. Convert to Array and Preprocess
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            img_array = preprocess_input(img_array)       # Scale pixels
            
            # 3. Make Prediction
            predictions = model.predict(img_array)
            
            # 4. Decode the top 3 results
            results = decode_predictions(predictions, top=3)[0]
            
            st.success("Analysis Complete!")
            st.subheader("📊 Top Predictions:")
            
            # Display results beautifully with progress bars
            for result in results:
                object_name = result[1].replace("_", " ").title() # Clean up name
                confidence = float(result[2])
                
                # Layout: Name on left, confidence bar on right
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{object_name}**")
                with col2:
                    st.progress(confidence, text=f"{confidence*100:.1f}% Confidence")

st.divider()
st.caption("Built with TensorFlow (MobileNetV2) & Streamlit. 100% Offline & API-Free.")
