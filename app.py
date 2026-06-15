import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Page Config
st.set_page_config(page_title="AI Prompt Enhancer", page_icon="✨", layout="centered")

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_ai_model():
    # google/flan-t5-small is free, lightweight, and great for text instructions
    model_name = "google/flan-t5-small" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

st.title("✨ AI Prompt Enhancer & Tester")
st.write("Apna simple prompt yahan likhein, aur AI usko mazeed professional aur detailed bana dega!")

# Load model in the background with a spinner
with st.spinner("Loading AI Model into memory... (First time only)"):
    tokenizer, model = load_ai_model()

st.divider()

# --- User Interface ---
st.subheader("📝 Draft Your Prompt")
user_prompt = st.text_area(
    "Enter your basic idea:", 
    placeholder="e.g., write an email to my boss for 2 days leave...",
    height=100
)

# Enhancement Options
enhancement_style = st.selectbox(
    "How do you want to enhance it?", 
    ["Make it more professional and detailed", "Make it creative and engaging", "Make it concise and clear"]
)

if st.button("🚀 Enhance Prompt", use_container_width=True):
    if user_prompt.strip() == "":
        st.warning("Pehle koi prompt to likhein!")
    else:
        with st.spinner("AI is working its magic..."):
            # Prepare the instruction for the model
            instruction = f"{enhancement_style}: {user_prompt}"
            
            # Tokenize and Generate
            inputs = tokenizer(instruction, return_tensors="tf")
            
            # Generate output (adjust max_new_tokens for longer responses)
            outputs = model.generate(
                inputs["input_ids"], 
                max_new_tokens=150, 
                temperature=0.7, 
                do_sample=True
            )
            
            # Decode the AI's response
            enhanced_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Display Results
            st.success("Enhancement Complete!")
            st.subheader("💡 Enhanced Prompt:")
            
            # Display inside a code block so the user can easily copy it
            st.code(enhanced_text, language="text")
            
st.divider()
st.caption("Powered by TensorFlow, Hugging Face (Flan-T5), and Streamlit.")
