"""
Streamlit X-ray AI Analysis App
- Real-time fracture detection and pneumonia classification
- In-memory image processing with AI explanations
- Optimized for Streamlit Cloud deployment
"""

import os
import re
import json
import io
import base64
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO
from dotenv import load_dotenv
from model_loader import load_all_models, display_model_status

# Load environment variables
load_dotenv()

# Auto-pull LFS files on Streamlit Cloud
def ensure_lfs_files():
    """Ensure LFS files are available, particularly for Streamlit Cloud"""
    try:
        import subprocess
        
        # Check if this looks like Streamlit Cloud environment
        is_cloud = os.environ.get('STREAMLIT_CLOUD', False) or 'streamlit' in os.environ.get('USER', '').lower()
        
        # Always try to pull LFS files if they seem to be pointer files
        model_files = ['best_fracture_yolov8.pt', 'best_classifier.pt', 'best_detection.pt']
        needs_pull = False
        
        for file_path in model_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size < 1000:  # Likely an LFS pointer file
                    needs_pull = True
                    break
            else:
                needs_pull = True
                break
        
        if needs_pull:
            st.info("🔄 Downloading model files...")
            result = subprocess.run(['git', 'lfs', 'pull'], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                st.success("✅ Model files downloaded successfully")
            else:
                st.warning(f"⚠️ LFS pull warning: {result.stderr}")
                
    except Exception as e:
        st.warning(f"⚠️ Could not auto-pull LFS files: {e}")

# Ensure LFS files are available
ensure_lfs_files()

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FRACTURE_MODEL_PATH = os.environ.get("FRACTURE_MODEL_PATH", "best_fracture_yolov8.pt")
PNEUMONIA_CLASSIFIER_PATH = os.environ.get("PNEUMONIA_CLASSIFIER_PATH", "best_classifier.pt")
PNEUMONIA_DET_MODEL_PATH = os.environ.get("PNEUMONIA_DET_MODEL_PATH", "best_detection.pt")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ENABLE_VISION_IN_PROMPT = os.environ.get("ENABLE_VISION_IN_PROMPT", "true").lower() == "true"
LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()

# Configure Streamlit page
st.set_page_config(
    page_title="X-ray AI Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def log(msg: str, level: str = "INFO"):
    """Logging function"""
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        st.write(f"[{level}] {msg}")

@st.cache_resource
def load_models():
    """Load all AI models with caching - delegated to model_loader"""
    return load_all_models()

@st.cache_resource
def initialize_gemini():
    """Initialize Gemini client with caching"""
    if not GEMINI_API_KEY:
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel("gemini-2.0-flash-exp")
        return client
    except Exception as e:
        return None

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def detect_fractures(image_array: np.ndarray, model) -> Tuple[np.ndarray, List[Dict]]:
    """Detect fractures in the image"""
    if model is None:
        raise RuntimeError("Fracture model not loaded")
    
    results = model(image_array)
    annotated_img = image_array.copy()
    detections = []
    
    if len(results[0].boxes) == 0:
        return annotated_img, detections
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names.get(cls, f"class_{cls}")
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (220, 20, 60), 2)
        cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 20, 60), 1)
        
        detections.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1)
        })
    
    return annotated_img, detections

def classify_pneumonia(image_array: np.ndarray, model) -> Tuple[str, float, List[float], List[str]]:
    """Classify pneumonia in the image"""
    if model is None:
        raise RuntimeError("Pneumonia classification model not loaded")
    
    results = model.predict(image_array, verbose=False)
    res = results[0]
    
    if not hasattr(res, "probs"):
        raise RuntimeError("No probabilities in classification result")
    
    pred_idx = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    names = getattr(res, "names", getattr(model, "names", {}))
    label = names.get(pred_idx, f"class_{pred_idx}")
    probs_list = res.probs.data.tolist() if hasattr(res.probs, "data") else []
    class_names = [names[i] for i in range(len(probs_list))] if probs_list else []
    
    return label, conf, probs_list, class_names

def detect_pneumonia_regions(image_array: np.ndarray, classification_label: str, model) -> Tuple[np.ndarray, List[Dict]]:
    """Detect pneumonia regions in the image"""
    if classification_label.lower() != "pneumonia":
        return image_array, []
    
    if model is None:
        return image_array, []
    
    results = model(image_array)
    annotated_img = image_array.copy()
    regions = []
    
    if len(results[0].boxes) == 0:
        return annotated_img, regions
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        raw_label = model.names.get(cls, f"class_{cls}")
        
        if raw_label.lower() == "normal":
            continue
        
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 140, 0), 2)
        cv2.putText(annotated_img, f"{raw_label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)
        
        regions.append({
            "label": raw_label,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1)
        })
    
    return annotated_img if regions else image_array, regions

def gemini_visual_inspect(image_bytes: bytes, client) -> Dict[str, Any]:
    """Visual inspection using Gemini"""
    if client is None:
        return {
            "raw_description": "Gemini unavailable.",
            "detected_type": "uncertain",
            "confidence_hint": 0.0,
            "rationale": "Skipped due to missing Gemini client."
        }
    
    try:
        prompt = (
            "Describe this image briefly (<=20 words). Then state if it is a medical radiographic X-ray (Yes/No) "
            "and your confidence 0-1. Format strictly as JSON with keys: description, is_xray (Yes|No|Uncertain), confidence."
        )
        
        response = client.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        
        text = (response.text or "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON in visual inspection response")
        
        vi = json.loads(match.group(0))
        is_xray_field = vi.get("is_xray", "").lower()
        
        if is_xray_field.startswith("y"):
            dtype = "xray"
        elif is_xray_field.startswith("n"):
            dtype = "non_xray"
        else:
            dtype = "uncertain"
        
        return {
            "raw_description": vi.get("description", "").strip(),
            "detected_type": dtype,
            "confidence_hint": float(vi.get("confidence", 0.0)),
            "rationale": f"Model said is_xray={vi.get('is_xray')} conf={vi.get('confidence')}"
        }
    except Exception as e:
        st.warning(f"Visual inspection failed: {e}")
        return {
            "raw_description": "Inspection error.",
            "detected_type": "uncertain",
            "confidence_hint": 0.0,
            "rationale": f"Error: {e}"
        }

def gemini_structured_advice(task: str, context: Dict[str, Any], image_bytes: bytes, visual_desc: str, client) -> Dict[str, Any]:
    """Get structured advice from Gemini"""
    if client is None:
        return {
            "summary": f"{task.replace('_', ' ').title()} analysis completed.",
            "risk_level": "unknown",
            "reasoning": "Detailed reasoning unavailable.",
            "detection_analysis": "No detailed region analysis.",
            "recommendations": "Consult a qualified medical professional.",
            "follow_up": "Monitor symptoms; seek care if they worsen.",
            "disclaimer": "AI-generated assistance. Not a diagnosis."
        }
    
    context_lines = []
    if task == "fracture_detection":
        context_lines.append(f"num_detections={context.get('num_detections')}")
        for d in context.get("detections", []):
            context_lines.append(f"{d['label']} conf={d['confidence']:.2f} box={d['box']}")
    elif task == "pneumonia_analysis":
        cls = context.get("classification", {})
        context_lines.append(f"class_label={cls.get('label')} conf={cls.get('confidence')}")
        context_lines.append(f"regions_found={context.get('regions_found')}")
        for r in context.get("regional_detections", []):
            context_lines.append(f"region {r['label']} conf={r['confidence']:.2f} box={r['box']}")
    
    context_lines.append(f"visual_description={visual_desc[:120]}")
    
    prompt = (
        "You are a medical imaging assistant. Produce ONLY valid minified JSON with keys:\n"
        "['summary', 'risk_level', 'reasoning', 'detection_analysis', 'recommendations', 'follow_up', 'disclaimer']\n"
        "No extra keys, no markdown.\n"
        f"Task: {task}\n"
        "Context:\n" + "\n".join(context_lines) + "\n"
        "Rules:\n"
        "- summary: <= 26 words.\n"
        "- risk_level: choose one of none, low, moderate, high.\n"
        "- reasoning: <= 45 words referencing signals.\n"
        "- detection_analysis: <= 50 words or 'None'.\n"
        "- recommendations: 2-4 clauses separated by ';'.\n"
        "- follow_up: single sentence.\n"
        "- disclaimer: single sentence (not diagnostic).\n"
        "Return JSON only."
    )
    
    parts = [prompt]
    if ENABLE_VISION_IN_PROMPT:
        parts.append({"mime_type": "image/jpeg", "data": image_bytes})
    
    try:
        response = client.generate_content(parts)
        raw_text = (response.text or "").strip()
        
        # Extract JSON
        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            raise ValueError("No JSON found in response")
        
        parsed = json.loads(match.group(0))
        
        # Ensure all required keys exist
        required_keys = ["summary", "risk_level", "reasoning", "detection_analysis", "recommendations", "follow_up", "disclaimer"]
        for key in required_keys:
            if key not in parsed:
                parsed[key] = f"No {key} provided"
        
        return parsed
    except Exception as e:
        st.warning(f"Gemini structured advice failed: {e}")
        return {
            "summary": f"{task.replace('_', ' ').title()} analysis completed.",
            "risk_level": "unknown",
            "reasoning": "Detailed reasoning unavailable.",
            "detection_analysis": "No detailed region analysis.",
            "recommendations": "Consult a qualified medical professional.",
            "follow_up": "Monitor symptoms; seek care if they worsen.",
            "disclaimer": "AI-generated assistance. Not a diagnosis."
        }

def get_severity_color(severity: str) -> str:
    """Get color for severity badge"""
    colors = {
        "none": "🟢",
        "low": "🟡", 
        "moderate": "🟠",
        "high": "🔴"
    }
    return colors.get(severity, "⚪")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🩻 X-ray AI Analysis")
    st.markdown("Upload an X-ray image for AI-powered fracture detection and pneumonia classification with real-time explanations.")
    
    # Load models and Gemini (with improved error handling)
    models_result = load_models()
    if isinstance(models_result, tuple):
        models, loading_status = models_result
    else:
        # Fallback for backward compatibility
        models = models_result
        loading_status = {}
    
    gemini_client = initialize_gemini()
    
    # Add debug panel in sidebar for development
    with st.sidebar:
        if st.checkbox("🔧 Show Model Debug Info"):
            display_model_status()
    
    # Task selection in main area
    st.subheader("Select Analysis Type")
    col_task1, col_task2 = st.columns(2)
    
    with col_task1:
        fracture_selected = st.button("🦴 Fracture Detection", use_container_width=True, type="primary" if st.session_state.get('task_mode', 'fracture') == 'fracture' else "secondary")
    with col_task2:
        pneumonia_selected = st.button("🫁 Pneumonia Analysis", use_container_width=True, type="primary" if st.session_state.get('task_mode', 'fracture') == 'pneumonia' else "secondary")
    
    # Set task mode based on button clicks
    if fracture_selected:
        st.session_state.task_mode = 'fracture'
    elif pneumonia_selected:
        st.session_state.task_mode = 'pneumonia'
    
    # Default to fracture if no selection
    task_mode = st.session_state.get('task_mode', 'fracture')
    
    st.divider()
    
    # File uploader in main area
    st.subheader("📁 Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG X-ray image"
    )
    
    if uploaded_file is not None:
        # Check if models are available
        if task_mode == "fracture" and models['fracture'] is None:
            st.error("❌ Fracture detection model not available. Please check model files.")
            return
        elif task_mode == "pneumonia" and models['pneumonia_cls'] is None:
            st.error("❌ Pneumonia classification model not available. Please check model files.")
            return
        
        if gemini_client is None:
            st.warning("⚠️ AI explanations unavailable. Proceeding with basic analysis.")
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original X-ray")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Process the image
        with st.spinner("🔬 Analyzing image with AI models..."):
            try:
                # Convert to OpenCV format
                image_array = pil_to_cv2(image)
                
                # Convert to bytes for Gemini
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                image_bytes = img_byte_arr.getvalue()
                
                # Visual inspection only if Gemini is available
                if gemini_client:
                    visual_inspection = gemini_visual_inspect(image_bytes, gemini_client)
                    
                    # Check if it's a valid X-ray
                    if visual_inspection["detected_type"] == "non_xray":
                        st.error("❌ **Not an X-ray Image**")
                        st.warning("The uploaded image does not appear to be a medical X-ray. Please upload a valid chest radiograph.")
                        return
                else:
                    # Skip visual inspection if Gemini not available
                    visual_inspection = {
                        "raw_description": "Visual inspection skipped",
                        "detected_type": "uncertain",
                        "confidence_hint": 0.0,
                        "rationale": "Gemini API not available"
                    }
                
                # Process based on task mode
                if task_mode == "fracture":
                    # Detect fractures
                    annotated_img, detections = detect_fractures(image_array, models['fracture'])
                    
                    # Display results
                    with col2:
                        st.subheader("🦴 Fracture Detection Results")
                        if len(detections) > 0:
                            st.image(cv2_to_pil(annotated_img), use_container_width=True)
                        else:
                            st.image(image, use_container_width=True)
                            st.success("✅ No fractures detected")
                    
                    # Get AI explanation if available
                    if gemini_client:
                        context = {
                            "num_detections": len(detections),
                            "detections": detections
                        }
                        explanation = gemini_structured_advice(
                            "fracture_detection", context, image_bytes, 
                            visual_inspection.get("raw_description", ""), gemini_client
                        )
                    else:
                        explanation = {
                            "summary": f"Fracture analysis completed. {len(detections)} detections found.",
                            "risk_level": "unknown",
                            "reasoning": "Detailed reasoning unavailable without AI service.",
                            "detection_analysis": f"Model detected {len(detections)} potential fracture regions.",
                            "recommendations": "Consult a qualified medical professional for diagnosis.",
                            "follow_up": "Seek medical attention if symptoms persist.",
                            "disclaimer": "Automated analysis. Not a medical diagnosis."
                        }
                    
                    # Determine severity
                    max_conf = max([d['confidence'] for d in detections], default=0.0)
                    if max_conf >= 0.85:
                        severity = "high"
                    elif max_conf >= 0.7:
                        severity = "moderate"
                    elif max_conf > 0:
                        severity = "low"
                    else:
                        severity = "none"
                    
                    # Display detailed results
                    st.markdown("---")
                    st.subheader(f"📊 Analysis Results {get_severity_color(severity)} **{severity.upper()}**")
                    
                    if detections:
                        st.markdown("### 🎯 Detections Found:")
                        for i, detection in enumerate(detections, 1):
                            st.markdown(f"**{i}.** {detection['label']} - Confidence: {detection['confidence']:.2f}")
                    else:
                        st.success("✅ No fractures detected by the AI model")
                
                elif task_mode == "pneumonia":
                    # Classify pneumonia
                    label, conf, probs, class_names = classify_pneumonia(image_array, models['pneumonia_cls'])
                    
                    # Detect regions if pneumonia
                    region_img, regions = detect_pneumonia_regions(image_array, label, models['pneumonia_det'])
                    
                    # Display results
                    with col2:
                        st.subheader("🫁 Pneumonia Analysis Results")
                        if label.lower() == "pneumonia" and regions:
                            st.image(cv2_to_pil(region_img), use_container_width=True)
                        else:
                            st.image(image, use_container_width=True)
                    
                    # Get AI explanation if available
                    if gemini_client:
                        context = {
                            "classification": {"label": label, "confidence": conf},
                            "regional_detections": regions,
                            "regions_found": len(regions)
                        }
                        explanation = gemini_structured_advice(
                            "pneumonia_analysis", context, image_bytes,
                            visual_inspection.get("raw_description", ""), gemini_client
                        )
                    else:
                        explanation = {
                            "summary": f"Pneumonia analysis completed. Classification: {label}.",
                            "risk_level": "unknown",
                            "reasoning": "Detailed reasoning unavailable without AI service.",
                            "detection_analysis": f"Model classified as {label} with {conf:.3f} confidence.",
                            "recommendations": "Consult a qualified medical professional for diagnosis.",
                            "follow_up": "Seek medical attention if symptoms persist.",
                            "disclaimer": "Automated analysis. Not a medical diagnosis."
                        }
                    
                    # Determine severity
                    if label.lower() != "pneumonia":
                        severity = "none"
                    elif conf >= 0.9:
                        severity = "high"
                    elif conf >= 0.75:
                        severity = "moderate"
                    else:
                        severity = "low"
                    
                    # Display detailed results
                    st.markdown("---")
                    st.subheader(f"📊 Analysis Results {get_severity_color(severity)} **{severity.upper()}**")
                    
                    st.markdown(f"### 🎯 Classification: **{label}** (Confidence: {conf:.3f})")
                    
                    if probs and class_names:
                        st.markdown("### 📈 Class Probabilities:")
                        for name, prob in zip(class_names, probs):
                            st.progress(prob, text=f"{name}: {prob:.3f}")
                    
                    if label.lower() == "pneumonia" and regions:
                        st.markdown("### 🔍 Region Detection:")
                        for i, region in enumerate(regions, 1):
                            st.markdown(f"**{i}.** {region['label']} - Confidence: {region['confidence']:.2f}")
                    elif label.lower() == "pneumonia":
                        st.info("ℹ️ No specific regions localized")
                
                # AI Explanation Section
                st.markdown("---")
                st.subheader("🤖 AI Medical Explanation")
                
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    st.markdown("#### 📝 Summary")
                    st.info(explanation['summary'])
                    
                    st.markdown("#### ⚠️ Risk Level")
                    risk_color = {
                        "none": "🟢 None",
                        "low": "🟡 Low", 
                        "moderate": "🟠 Moderate",
                        "high": "🔴 High"
                    }
                    st.markdown(f"**{risk_color.get(explanation['risk_level'], explanation['risk_level'])}**")
                    
                    st.markdown("#### 🧠 Reasoning")
                    st.write(explanation['reasoning'])
                
                with col_exp2:
                    st.markdown("#### 🎯 Detection Analysis")
                    st.write(explanation['detection_analysis'])
                    
                    st.markdown("#### 💡 Recommendations")
                    if ';' in explanation['recommendations']:
                        recommendations = [r.strip() for r in explanation['recommendations'].split(';') if r.strip()]
                        for rec in recommendations:
                            st.markdown(f"• {rec}")
                    else:
                        st.write(explanation['recommendations'])
                    
                    st.markdown("#### 🔄 Follow Up")
                    st.write(explanation['follow_up'])
                
                # Disclaimer
                st.markdown("---")
                st.warning(f"⚠️ **Disclaimer:** {explanation['disclaimer']}")
                
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
    
    else:
        st.info("👆 Please upload an X-ray image to begin analysis")

if __name__ == "__main__":
    main()
