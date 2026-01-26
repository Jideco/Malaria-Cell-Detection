# 🦠 Malaria Cell Detection - MobileNetV3 Transfer Learning Project

## Problem Description
**Context**: Malaria is a life-threatening disease caused by parasites transmitted through infected mosquito bites. Early detection through blood smear microscopy is critical for effective treatment, but manual examination by trained technicians is time-consuming, subjective, and requires specialized expertise that's often unavailable in resource-limited regions.

**Problem**: Can we build an automated deep learning system to detect malaria parasites in blood cell images with accuracy comparable to human experts, enabling faster diagnosis and treatment in underserved communities?

**Solution**: This project develops a Convolutional Neural Network (CNN) model using MobileNetV3 Large architecture that classifies red blood cell images as either "Parasitized" (infected with malaria) or "Uninfected", providing:
- **Automated Screening**: Rapid analysis of blood smear images
- **Mobile-First Design**: Lightweight model suitable for deployment on mobile devices
- **Resource Efficiency**: Optimized for low-power environments
- **Accessibility**: Deployable in remote areas with limited medical infrastructure

**Social Value**:
- **🏥 Healthcare Systems**: Reduce diagnostic time from hours to seconds
- **🌍 Global Health**: Enable malaria screening in rural and underserved communities
- **📱 Mobile Health**: Run diagnostic tools on smartphones in field settings
- **⚕️ Medical Professionals**: Provide AI-assisted second opinions for diagnosis

---
## Dataset
**Source**: National Institutes of Health (NIH) - Malaria Cell Images Dataset:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/  OR https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data  

**Description**:
- **Total Images**: 27,558 microscopic images of red blood cells
- **Classes**: Binary classification (Parasitized vs Uninfected)
- **Balance**: Perfectly balanced - 13,779 images per class
- **Quality**: High-resolution, expert-labeled cell images
- **Format**: JPEG images with varying resolutions (typically 130x130 to 150x150 pixels)

**Dataset Structure**:
```plaintext
data/
└── cell_images/
    ├── Parasitized/    # 13,779 images of infected cells
    └── Uninfected/     # 13,779 images of healthy cells
```

---
## Project Structure
```plaintext
malaria-cell-detection/
│
├── data/
│   └── cell_images/           # Dataset (gitignored - download instructions provided)
│       ├── Parasitized/
│       └── Uninfected/
├── notebook.ipynb                     # Complete EDA, training, Model evaluation, metrics generation, Model export to ONNX
│
├── malaria_mobilenet_v1_05.pth   # Trained MobileNetV3 Large model weights
├── malaria_model.onnx            # ONNX format model for deployment
│
├── metrics/                    # Evaluation metrics and visualizations      
│       ├── confusion_matrix.png
│       ├── roc_curve.png
|       ├── train_report.png
│       └── metrics_report.png
│
├── app.py       # Streamlit web application for inference
│
├── requirements.txt           # Python dependencies
└── README.md                   # This documentation file
```
---
## Key Findings from EDA
**Data Overview**:
- No missing values ✓
- Perfect class balance (50% Parasitized, 50% Uninfected) ✓
- Images show clear visual differences between infected and healthy cells
- Parasitized cells typically show ring-shaped parasites and altered cell morphology

**Image Characteristics**:
- **Color Distribution**: Parasitized cells often show blue-stained parasite regions
- **Texture Patterns**: Infected cells have distinctive texture patterns
- **Size Variations**: Cell sizes vary but parasites maintain consistent visual features
- **Background Noise**: Some images contain background artifacts that need preprocessing

**Preprocessing Steps**:
- **Resizing**: Standardized to 224x224 pixels (MobileNetV3 input requirement)
- **Normalization**: Pixel values normalized using ImageNet mean/std
- **Augmentation**: Rotation, flipping, zoom to improve model robustness
- **Data Split**: 80% training, 20% validation and testing

---
## Model Performance
**Best Model**: MobileNetV3 Large (5 epochs)

**Performace Metrics**:
- **Accuracy:** 93.4%
- **Precision:** 93.0%
- **Recall:** 93.0%
- **F1-Score:** 93.0%
  
 - <img width="515" height="272" alt="metrics_report" src="https://github.com/user-attachments/assets/df24b39a-5af4-447e-b0b4-b79d9d1b33dc" />

- **AUC-ROC:** 0.97
  
 - <img width="613" height="545" alt="roc_curve" src="https://github.com/user-attachments/assets/d53f3700-c3e8-4a6c-b33e-26dc79674da9" />

- **Confusion Matrix:**
  
 - <img width="514" height="468" alt="confusion_matrix" src="https://github.com/user-attachments/assets/50283291-1344-4995-85c5-c023798106e8" />

- **Training Details:**
  
 - <img width="1389" height="490" alt="10c20572-c2c5-4ccc-8292-7faff027f515" src="https://github.com/user-attachments/assets/c779eb2d-fd1e-4b62-b50e-f04c5c27712e" />

---
## ONNX Export
For efficient deployment, the trained model was exported to ONNX format:
   ```bash
   # Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model, 
    dummy_input, 
    "models/mobilenetv3_malaria.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
   ```
**Benefits of ONNX format:**
- Cross-platform compatibility
- Optimized inference speed
- Reduced model size (from 27.3MB to 13.8MB)
- Framework independence

---
## Installation & Setup
**Prerequisites**
- Python 3.8+
- PyTorch with CUDA support (optional)
- ONNX Runtime (for inference)

**Local Installation**
```bash
# Clone the repository
git clone https://github.com/Jideco/Malaria-Cell-Detection.git
cd malaria-cell-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (follow instructions in data/README.md)
# Dataset available at: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data

# Run the Streamlit app
streamlit run app.py
   ```
The Streamlit app will be available at http://localhost:8501

---
## Streamlit Web Application
The project includes a user-friendly Streamlit interface for malaria cell detection:

**Features:**
- Upload blood cell images via drag-and-drop interface
- Real-time prediction with probability scores
- Visual display of uploaded image with prediction overlay
- Confidence meter showing prediction certainty
- Performance metrics dashboard
- Educational information about malaria detection

**Usage:**
- Run the Streamlit app: streamlit run app/streamlit_app.py
- Upload a blood cell image using the file uploader
- View prediction result (Parasitized or Uninfected)
- See confidence score and probability distribution
- Compare with example images in the gallery

**The Streamlit app is available at** https://malaria-cell-detection-transfer-learning-opabode.streamlit.app/

---
## Project Limitations & Future Work

**Current Limitations**
- Training Duration: Only trained for 5 epochs (No GPU availabe)
- Single-cell focus: Only analyzes individual cells, not whole slide context
- Limited parasite stages: May not detect all malaria parasite development stages
- Image quality dependence: Performance varies with image resolution and staining quality
  
**Future Improvements**
- Extended Training: Train for more epochs with learning rate scheduling
- Model Quantization: Further reduce model size for mobile deployment
- Grad-CAM Visualizations: Show which regions of the image influenced the prediction
- Federated Learning: Train on diverse datasets while preserving privacy
- Multi-stage detection: Identify different parasite development stages
- Ensemble Methods: Combine multiple lightweight models for improved accuracy

---
## Tech Stack
- **Language:** Python 3.8
- **ML Framework:** PyTorch 2.1, torchvision
- **Model Format:** ONNX 1.14
- **Inference Engine:** ONNX Runtime
- **Computer Vision:** OpenCV, Pillow
- **Frontend:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Docker
- **Version Control:** Git

---
## Author
Mujeeb Olajide Opabode

GitHub: @Jideco
LinkedIn: https://www.linkedin.com/in/mujeeb-opabode-96b981189/

Email: jideopabode@gmail.com

---
## Acknowledgments
- Dataset Provider: National Institutes of Health (NIH)
- Research Team: Malaria Screener research group at Lister Hill National Center for Biomedical Communications
- Educational Resource: ML Zoomcamp by DataTalks.Club
- Open Source Community: PyTorch, ONNX, and Streamlit development teams

---
## License
This project is for educational and research purposes only. The model should NOT be used for clinical diagnosis without proper medical validation and regulatory approval.

