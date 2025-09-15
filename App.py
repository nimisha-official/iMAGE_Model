import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load model once (cached)
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cpu")  # CPU-only mode
    return pipe

st.title("🖼️ Stable Diffusion 1.5 - CPU (Streamlit)")

# User input
prompt = st.text_area("Enter your prompt:", "a fantasy castle on a mountain at sunset")

if st.button("Generate"):
    pipe = load_model()
    with st.spinner("Generating... this may take 2-5 minutes on CPU ⏳"):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_container_width=True)
        image.save("output.png")
        st.success("✅ Done! Saved as output.png")
