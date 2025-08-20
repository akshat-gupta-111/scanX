#!/usr/bin/env python3
"""
Test script to verify model loading before deployment
Run this locally to ensure models load correctly
"""

import os
import sys
from ultralytics import YOLO

def test_model_loading():
    """Test if all models can be loaded"""
    models = {
        'fracture': 'best_fracture_yolov8.pt',
        'pneumonia_cls': 'best_classifier.pt',
        'pneumonia_det': 'best_detection.pt'
    }
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    print("-" * 50)
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\nTesting {model_name} model...")
        print(f"Path: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                print(f"✅ {model_name} loaded successfully")
                print(f"   Model type: {type(model)}")
                if hasattr(model, 'names'):
                    print(f"   Classes: {list(model.names.values())}")
                results[model_name] = "SUCCESS"
            except Exception as e:
                print(f"❌ {model_name} failed to load: {e}")
                results[model_name] = f"ERROR: {e}"
        else:
            print(f"❌ {model_name} file not found")
            results[model_name] = "FILE_NOT_FOUND"
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    for model_name, status in results.items():
        print(f"{model_name}: {status}")
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    print(f"\nSuccessfully loaded: {success_count}/{len(models)} models")
    
    if success_count == len(models):
        print("🎉 All models loaded successfully! Ready for deployment.")
    else:
        print("⚠️ Some models failed to load. Check errors above.")
    
    return results

if __name__ == "__main__":
    test_model_loading()
