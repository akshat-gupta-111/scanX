"""
Model loader module with fallback mechanisms for Streamlit Cloud deployment
Handles large model files with Git LFS support and provides error handling
"""

import os
import time
import streamlit as st
from ultralytics import YOLO
import tempfile
import requests
from pathlib import Path

# Model configuration
MODEL_CONFIG = {
    "fracture": {
        "path": "best_fracture_yolov8.pt",
        "description": "Fracture Detection Model",
        "fallback_url": None  # Add fallback URL if needed
    },
    "pneumonia_cls": {
        "path": "best_classifier.pt", 
        "description": "Pneumonia Classification Model",
        "fallback_url": None  # Add fallback URL if needed
    },
    "pneumonia_det": {
        "path": "best_detection.pt",
        "description": "Pneumonia Detection Model", 
        "fallback_url": None  # Add fallback URL if needed
    }
}

def check_file_integrity(file_path: str) -> bool:
    """Check if file exists and has reasonable size"""
    try:
        if not os.path.exists(file_path):
            return False
        
        file_size = os.path.getsize(file_path)
        # Check if file is too small (likely LFS pointer file)
        if file_size < 1000:  # Less than 1KB suggests LFS pointer
            with open(file_path, 'r') as f:
                content = f.read()
                if 'version https://git-lfs.github.com/spec/v1' in content:
                    return False
        
        return file_size > 1000  # Model should be at least 1KB
    except Exception:
        return False

def download_lfs_file(file_path: str) -> bool:
    """Attempt to download LFS file using git lfs pull"""
    try:
        import subprocess
        result = subprocess.run(['git', 'lfs', 'pull', '--include', file_path], 
                              capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False

def load_model_with_retry(model_path: str, model_name: str, max_retries: int = 3) -> YOLO:
    """Load YOLO model with retry mechanism"""
    
    for attempt in range(max_retries):
        try:
            # Check file integrity first
            if not check_file_integrity(model_path):
                st.warning(f"⚠️ {model_name}: File integrity check failed (attempt {attempt + 1})")
                
                # Try to pull LFS file
                if download_lfs_file(model_path):
                    st.info(f"✅ {model_name}: Successfully pulled LFS file")
                else:
                    st.error(f"❌ {model_name}: Failed to pull LFS file")
                    
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            
            # Attempt to load model
            with st.spinner(f"Loading {model_name}..."):
                model = YOLO(model_path)
                
                # Test model by getting basic info
                _ = model.names
                
                st.success(f"✅ {model_name} loaded successfully")
                return model
                
        except Exception as e:
            st.warning(f"⚠️ {model_name} loading failed (attempt {attempt + 1}): {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error(f"❌ {model_name}: All loading attempts failed")
    
    return None

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all AI models with comprehensive error handling"""
    models = {}
    loading_status = {}
    
    st.info("🔄 Initializing AI models...")
    
    for model_key, config in MODEL_CONFIG.items():
        model_path = config["path"]
        model_desc = config["description"]
        
        st.write(f"Loading {model_desc}...")
        
        # Check if file exists
        if not os.path.exists(model_path):
            st.error(f"❌ {model_desc}: File not found at {model_path}")
            models[model_key] = None
            loading_status[model_key] = "missing"
            continue
        
        # Load model with retries
        model = load_model_with_retry(model_path, model_desc)
        models[model_key] = model
        loading_status[model_key] = "loaded" if model else "failed"
    
    # Display loading summary
    loaded_count = sum(1 for status in loading_status.values() if status == "loaded")
    total_count = len(MODEL_CONFIG)
    
    if loaded_count == total_count:
        st.success(f"🎉 All {total_count} models loaded successfully!")
    elif loaded_count > 0:
        st.warning(f"⚠️ {loaded_count}/{total_count} models loaded. Some features may be limited.")
    else:
        st.error("❌ No models could be loaded. Please check model files.")
    
    return models, loading_status

def get_model_info():
    """Get information about model files"""
    info = {}
    
    for model_key, config in MODEL_CONFIG.items():
        model_path = config["path"]
        file_info = {
            "exists": os.path.exists(model_path),
            "size": 0,
            "description": config["description"]
        }
        
        if file_info["exists"]:
            try:
                file_info["size"] = os.path.getsize(model_path)
                file_info["size_mb"] = round(file_info["size"] / (1024 * 1024), 2)
                
                # Check if it's an LFS pointer file
                if file_info["size"] < 1000:
                    with open(model_path, 'r') as f:
                        content = f.read()
                        if 'version https://git-lfs.github.com/spec/v1' in content:
                            file_info["is_lfs_pointer"] = True
                        else:
                            file_info["is_lfs_pointer"] = False
                else:
                    file_info["is_lfs_pointer"] = False
                    
            except Exception as e:
                file_info["error"] = str(e)
        
        info[model_key] = file_info
    
    return info

def display_model_status():
    """Display current model loading status"""
    st.subheader("🔧 Model Status")
    
    model_info = get_model_info()
    
    for model_key, info in model_info.items():
        with st.expander(f"{info['description']} ({model_key})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**File:** {MODEL_CONFIG[model_key]['path']}")
                st.write(f"**Exists:** {'✅' if info['exists'] else '❌'}")
                
            with col2:
                if info['exists']:
                    st.write(f"**Size:** {info.get('size_mb', 0)} MB")
                    if info.get('is_lfs_pointer'):
                        st.warning("⚠️ LFS pointer file detected")
                    else:
                        st.success("✅ Actual model file")
                
                if 'error' in info:
                    st.error(f"Error: {info['error']}")
