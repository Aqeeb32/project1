import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Page Config
st.set_page_config(page_title="AI Prompt Enhancer", page_icon="✨", layout="centered")

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_ai_model():
    # Updated to 'base' model for better reasoning while staying within free tier limits
    model_name = "google/flan-t5-base" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

st.title("✨ AI Prompt Enhancer & Tester")
st.write("Apna simple prompt yahan likhein, aur AI usko mazeed professional bana dega!")

# Load model in the background with a spinner
with st.spinner("Loading AI Model into memory... (This takes a moment on startup)"):
    tokenizer, model = load_ai_model()

st.divider()

# --- User Interface ---
st.subheader("📝 Draft Your Prompt")
user_prompt = st.text_area(
    "Enter your basic idea:", 
    placeholder="e.g., I am applying in an IT company for an internship and I need a cover letter.",
    height=100
)

# Enhancement Options
enhancement_style = st.selectbox(
    "How do you want to enhance it?", 
    [
        "Make it more professional and detailed", 
        "Make it concise and clear",
        "Improve grammar and structure"
    ]
)

if st.button("🚀 Enhance Prompt", use_container_width=True):
    if user_prompt.strip() == "":
        st.warning("Pehle koi prompt to likhein!")
    else:
        with st.spinner("AI is working its magic..."):
            # Strict formatting to prevent the model from hallucinating
            instruction = f"Task: {enhancement_style} this text.\nText: {user_prompt}\nRewritten Text:"
            
            # Tokenize Input
            inputs = tokenizer(instruction, return_tensors="tf")
            
            # Generate output with stricter parameters to stay on track
            outputs = model.generate(
                inputs["input_ids"], 
                max_new_tokens=150, 
                temperature=0.3,          # Lower temperature = less hallucination
                do_sample=True,
                repetition_penalty=1.2    # Prevents repeating the same phrases
            )
            
            # Decode the AI's response
            enhanced_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Display Results
            st.success("Enhancement Complete!")
            st.subheader("💡 Enhanced Prompt:")
            
            # Display inside a code block so the user can easily copy it
            st.code(enhanced_text, language="text")
            
st.divider()
st.caption("Powered by TensorFlow, Hugging Face (Flan-T5-Base), and Streamlit.")
