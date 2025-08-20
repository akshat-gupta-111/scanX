# ScanX Deployment Guide for Streamlit Cloud

This guide provides step-by-step instructions for deploying the ScanX application to Streamlit Cloud with proper Git LFS support for large model files.

## 🚀 Quick Deployment

### Prerequisites
- Git repository with LFS-tracked model files
- Streamlit Cloud account
- GEMINI_API_KEY (optional, for AI explanations)

### Step 1: Git LFS Setup (Already Done)
The repository is already configured with Git LFS for model files:
- ✅ `best_classifier.pt` (9.78 MB) - Pneumonia Classification Model
- ✅ `best_detection.pt` (21.47 MB) - Pneumonia Detection Model  
- ✅ `best_fracture_yolov8.pt` (5.96 MB) - Fracture Detection Model

### Step 2: Deploy to Streamlit Cloud

1. **Login to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Select repository: `akshat-gupta-111/scanX`
   - Main file path: `app.py`
   - App URL: Choose your preferred URL

3. **Environment Configuration**
   - Add environment variables in "Advanced settings":
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ENABLE_VISION_IN_PROMPT=true
     APP_LOG_LEVEL=INFO
     ```

4. **Deploy**
   - Click "Deploy!"
   - Wait for build and deployment (may take 5-10 minutes)

### Step 3: Verify Deployment

1. **Check Model Loading**
   - App should display model loading status
   - Enable "Show Model Debug Info" in sidebar
   - Verify all 3 models show as loaded

2. **Test Functionality**
   - Upload a test X-ray image
   - Try both Fracture Detection and Pneumonia Analysis
   - Verify AI explanations work (if Gemini API key provided)

## 🔧 Troubleshooting

### OpenCV Import Errors

**Symptoms:**
- ImportError related to cv2 module
- Bootstrap errors in OpenCV initialization
- Missing system dependencies

**Solutions:**

1. **System Dependencies (Already Configured):**
   ```
   packages.txt includes:
   - libgl1-mesa-glx
   - libglib2.0-0  
   - libsm6
   - libxext6
   - libxrender-dev
   - libgomp1
   ```

2. **OpenCV Version Issues:**
   - Current: `opencv-contrib-python-headless==4.8.0.76`
   - Alternative: `opencv-python-headless==4.8.1.78`
   - Fallback: App has built-in OpenCV fallbacks

3. **Force Rebuild:**
   - Delete and recreate Streamlit Cloud app
   - Clear build cache in Streamlit Cloud settings

### Model Files Not Loading

**Symptoms:**
- "Model not found" errors
- LFS pointer files instead of actual models
- App crashes during model loading

**Solutions:**

1. **Verify LFS Setup Locally:**
   ```bash
   git lfs ls-files
   git lfs pull
   python test_models.py
   ```

2. **Check Streamlit Cloud Logs:**
   - Look for LFS-related errors
   - Verify `packages.txt` includes `git-lfs`

3. **Force LFS Pull on Streamlit Cloud:**
   - Add this to your app temporarily:
   ```python
   import subprocess
   subprocess.run(['git', 'lfs', 'pull'], cwd='.')
   ```

### Performance Issues

**Symptoms:**
- Slow model loading
- App timeouts
- Memory errors

**Solutions:**

1. **Optimize Model Loading:**
   - Models are cached using `@st.cache_resource`
   - First load may be slow (normal)
   - Subsequent loads should be fast

2. **Monitor Resource Usage:**
   - Streamlit Cloud has memory limits
   - Models total ~37MB, well within limits

### API Issues

**Symptoms:**
- AI explanations not working
- Gemini API errors

**Solutions:**

1. **Check API Key:**
   - Verify `GEMINI_API_KEY` is set correctly
   - Test API key separately
   - App works without API key (basic mode)

2. **Rate Limiting:**
   - Gemini has usage limits
   - Consider adding retry logic for production

## 📁 File Structure

```
app_streamlit/
├── app.py                    # Main Streamlit application
├── model_loader.py          # Robust model loading with LFS support
├── test_models.py           # Diagnostics and testing
├── requirements.txt         # Python dependencies
├── packages.txt            # System packages (git-lfs)
├── .gitattributes          # LFS file tracking rules
├── best_classifier.pt      # Pneumonia classification model (LFS)
├── best_detection.pt       # Pneumonia detection model (LFS)
├── best_fracture_yolov8.pt # Fracture detection model (LFS)
└── .env                    # Environment variables (local only)
```

## 🔐 Security

### Environment Variables
- **Never commit API keys to repository**
- Use Streamlit Cloud's environment variable feature
- Local development: use `.env` file (gitignored)

### Model Security
- Models are public in LFS (consider private repo for sensitive models)
- No user data is stored
- All processing happens in-memory

## 🚀 Production Considerations

### Scaling
- Streamlit Cloud auto-scales to some extent
- For high traffic, consider:
  - Streamlit Cloud Teams/Enterprise
  - Self-hosted deployment
  - Model optimization/quantization

### Monitoring
- Use built-in model status debugging
- Monitor Streamlit Cloud metrics
- Set up error tracking if needed

### Updates
```bash
# To update models:
git add new_model.pt
git commit -m "Update model"
git push origin main

# Streamlit Cloud will auto-redeploy
```

## 🧪 Development Workflow

### Local Development
1. **Setup:**
   ```bash
   git clone https://github.com/akshat-gupta-111/scanX.git
   cd scanX
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **LFS Setup:**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Test:**
   ```bash
   python test_models.py
   streamlit run app.py
   ```

### Model Updates
1. Replace model files
2. Test locally: `python test_models.py`
3. Commit and push (LFS handles large files)
4. Streamlit Cloud auto-redeploys

---

## 📞 Support

If you encounter issues:

1. **Check Logs:** Streamlit Cloud app logs
2. **Run Diagnostics:** `python test_models.py` locally
3. **Verify LFS:** `git lfs ls-files` and `git lfs status`
4. **Test Locally:** Ensure app works locally first

For more help, check the troubleshooting section or create an issue on GitHub.