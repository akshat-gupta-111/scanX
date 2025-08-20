# Streamlit Cloud Deployment Guide

## Common Issues and Solutions

### 1. Model Loading Issues

**Problem:** Models not loading on Streamlit Cloud but working locally.

**Solutions:**

#### A. Check File Sizes
- Streamlit Cloud has limitations on file sizes
- Your model files:
  - `best_detection.pt`: ~22.5 MB
  - `best_classifier.pt`: ~10.3 MB  
  - `best_fracture_yolov8.pt`: ~6.2 MB

#### B. Memory Optimization
Add this to your app to load models lazily:

```python
@st.cache_resource
def load_model_on_demand(model_path, model_type):
    """Load individual model on demand"""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading {model_type}: {e}")
        return None
```

#### C. Requirements.txt Optimization
Update your requirements.txt:

```
streamlit>=1.28.0
torch>=2.0.0,<3.0.0
torchvision>=0.15.0,<1.0.0
opencv-python-headless==4.8.1.78
ultralytics>=8.0.0
google-generativeai>=0.3.0
requests>=2.31.0
Pillow>=9.5.0
numpy>=1.24.0,<2.0.0
python-dotenv>=1.0.0
```

### 2. Streamlit Cloud Configuration

#### A. Secrets Configuration
In your Streamlit Cloud app dashboard, go to "Settings" > "Secrets" and add:

```toml
GEMINI_API_KEY = "your_actual_gemini_api_key"
FRACTURE_MODEL_PATH = "best_fracture_yolov8.pt"
PNEUMONIA_CLASSIFIER_PATH = "best_classifier.pt"
PNEUMONIA_DET_MODEL_PATH = "best_detection.pt"
ENABLE_VISION_IN_PROMPT = true
APP_LOG_LEVEL = "INFO"
```

#### B. Advanced Settings
In "Settings" > "Advanced":
- Python version: 3.11
- Enable: "Always rerun"

### 3. Model File Upload

#### A. Git LFS (Recommended)
If your models are >25MB, use Git LFS:

```bash
# Install git-lfs
git lfs install

# Track model files
git lfs track "*.pt"

# Add and commit
git add .gitattributes
git add *.pt
git commit -m "Add model files with LFS"
git push
```

#### B. Alternative: External Storage
Host models on external storage and download them:

```python
import requests
import os

@st.cache_resource
def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
    return filename
```

### 4. Debugging Steps

#### A. Enable Debug Mode
Add to your app:

```python
# Add after imports
import sys
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir('.')}")
```

#### B. Check Resource Usage
Monitor your app's resource usage in Streamlit Cloud dashboard.

#### C. Test Locally First
Before deploying, test with similar conditions:

```bash
# Install exact same versions
pip install -r requirements.txt

# Test with limited memory
ulimit -v 1000000  # Limit virtual memory to ~1GB
streamlit run app.py
```

### 5. Performance Optimization

#### A. Lazy Loading
Only load models when needed:

```python
def get_fracture_model():
    if 'fracture_model' not in st.session_state:
        st.session_state.fracture_model = load_model_on_demand(
            FRACTURE_MODEL_PATH, "fracture"
        )
    return st.session_state.fracture_model
```

#### B. Model Compression
Consider using model quantization or compression:

```python
# Example with torch quantization
model = YOLO(model_path)
model.model = torch.quantization.quantize_dynamic(
    model.model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 6. Troubleshooting Checklist

- [ ] Model files are in repository root
- [ ] File sizes are under limits
- [ ] Requirements.txt has correct versions
- [ ] Secrets are properly configured
- [ ] Git LFS is used for large files
- [ ] App works locally with same requirements
- [ ] Debug information shows correct file paths
- [ ] Memory usage is within limits

### 7. Error Messages and Solutions

| Error | Solution |
|-------|----------|
| "File not found" | Check file paths and Git LFS |
| "Out of memory" | Use lazy loading or model compression |
| "Module not found" | Update requirements.txt |
| "Model loading failed" | Check model file integrity |
| "Secrets not found" | Configure secrets in dashboard |

### 8. Contact Support

If issues persist:
1. Check Streamlit Cloud status page
2. Review app logs in dashboard
3. Contact Streamlit support with:
   - App URL
   - Error messages
   - Requirements.txt
   - Model file sizes
