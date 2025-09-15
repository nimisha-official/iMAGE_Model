# 🖼️ Stable Diffusion 1.5
This project is a simple Streamlit app that lets you generate images using **Stable Diffusion 1.5** locally on your CPU.


## 🔹 1. Import Libraries

```python
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
```

- **Streamlit** → creates the simple web UI.  
- **diffusers** → Hugging Face library that loads and runs Stable Diffusion models.  
- **torch** → backend (PyTorch) that actually runs the neural network.  



## 🔹 2. Load the Model (only once, cached)

```python
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to("cpu")  # CPU-only mode
    return pipe
```

- Downloads **Stable Diffusion 1.5 weights** from Hugging Face (≈4GB) the first time you run it.  
- Saves them locally (next time it just reuses the cached model).  
- Moves the pipeline to **CPU** (since you don’t have a GPU).  
- 💡 `@st.cache_resource` ensures the model is loaded **only once** → not every time you click the button.  


## 🔹 3. Streamlit UI

```python
st.title("🖼️ Stable Diffusion 1.5 - CPU (Streamlit)")
prompt = st.text_area("Enter your prompt:", "a fantasy castle on a mountain at sunset")
```

- Shows a page title.  
- Adds a text box where you type your prompt (default is `"a fantasy castle on a mountain at sunset"`).  


## 🔹 4. Generate Button

```python
if st.button("Generate"):
    pipe = load_model()
    with st.spinner("Generating... this may take 2-5 minutes on CPU ⏳"):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("output.png")
        st.success("✅ Done! Saved as output.png")
```

When you press **Generate**, it:  
- Loads the model (from cache).  
- Runs Stable Diffusion on CPU with your prompt.  
- Shows a spinner ⏳ while generating (takes ~2–5 min on CPU).  
- Displays the result in the app.  
- Saves it locally as **output.png**.  
- Shows ✅ success message.  



## 🔹 What actually happens in the background

1. Your text prompt → converted into embeddings (vector form).  
2. The model starts with random noise → gradually denoises into an image guided by your prompt.  
3. This process is slow on CPU because it runs **thousands of matrix multiplications**.  



## ✅ Summary

You get a **mini local Stable Diffusion app** where you:  
- Enter a text prompt  
- Click a button  
- Wait a few minutes  
- Get an image both on screen and saved to disk 🎉  

## 🧞‍♂️ Screenshort

<img width="1920" height="884" alt="image" src="https://github.com/user-attachments/assets/58187aed-45aa-4d76-b709-63e2690c0d7d" />

