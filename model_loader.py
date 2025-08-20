"""
Alternative lightweight model loading for Streamlit Cloud
Use this if main app.py has memory issues
"""

import streamlit as st
import os
from ultralytics import YOLO

@st.cache_resource
def load_model_on_demand(model_name):
    """Load individual model only when needed"""
    model_paths = {
        'fracture': 'best_fracture_yolov8.pt',
        'pneumonia_cls': 'best_classifier.pt', 
        'pneumonia_det': 'best_detection.pt'
    }
    
    model_path = model_paths.get(model_name)
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model {model_name} not found at {model_path}")
        return None
    
    try:
        with st.spinner(f"Loading {model_name} model..."):
            model = YOLO(model_path)
        st.success(f"✅ {model_name} model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load {model_name}: {str(e)}")
        return None

def get_fracture_model():
    """Get fracture detection model"""
    if 'fracture_model' not in st.session_state:
        st.session_state.fracture_model = load_model_on_demand('fracture')
    return st.session_state.fracture_model

def get_pneumonia_classifier():
    """Get pneumonia classification model"""
    if 'pneumonia_cls_model' not in st.session_state:
        st.session_state.pneumonia_cls_model = load_model_on_demand('pneumonia_cls')
    return st.session_state.pneumonia_cls_model

def get_pneumonia_detector():
    """Get pneumonia detection model"""
    if 'pneumonia_det_model' not in st.session_state:
        st.session_state.pneumonia_det_model = load_model_on_demand('pneumonia_det')
    return st.session_state.pneumonia_det_model

def clear_models():
    """Clear all models from session state to free memory"""
    keys_to_remove = ['fracture_model', 'pneumonia_cls_model', 'pneumonia_det_model']
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_resource.clear()
    st.success("🗑️ Models cleared from memory")
