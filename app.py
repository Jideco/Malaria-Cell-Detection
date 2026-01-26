import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="Malaria Cell Classifier", page_icon="🔬")
st.title("🔬 Malaria Cell Classifier")
st.write("Upload a blood cell image (PNG/JPG) to detect if it is 'Parasitized' or 'Uninfected'.Get the data set from the [LHNCBC Malaria Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets) OR (https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data).")

# 1. Load the ONNX Model (cached so it only loads once)
@st.cache_resource
def load_model():
    return ort.InferenceSession("malaria_model.onnx")

session = load_model()

# 2. Image Preprocessing 
def preprocess(image):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    # ImageNet Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1) # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0) # Add Batch dimension

    return np.array(img_array, dtype=np.float32)

# 3. File Uploader
uploaded_file = st.file_uploader("Choose a cell image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Cell Image', width='stretch')
    
    if st.button('Run Prediction'):
        with st.spinner('Analyzing...'):
            # Preprocess
            input_tensor = preprocess(image)
            
            # Run Inference
            input_name = session.get_inputs()[0].name
            
            # Double check the feed dictionary
            try:
                outputs = session.run(None, {input_name: input_tensor})
                
                # Get Result
                prediction = int(np.argmax(outputs))
                classes = ['Uninfected', 'Parasitized']
                result = classes[prediction]
                
                # Show Result
                if result == 'Parasitized':
                    st.error(f"Prediction: {result}")
                else:
                    st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Inference Error: {e}")

st.divider()
st.caption("2026 Jideco | Malaria Cell Detection using Transfer Learning")
