"""
Test script to validate model loading with Git LFS
This script helps diagnose model loading issues on Streamlit Cloud
"""

import os
import sys
from pathlib import Path
from model_loader import get_model_info, load_all_models, check_file_integrity

def test_file_existence():
    """Test if model files exist and have correct sizes"""
    print("🔍 Testing model file existence and integrity...")
    
    model_files = [
        "best_fracture_yolov8.pt",
        "best_classifier.pt", 
        "best_detection.pt"
    ]
    
    all_good = True
    
    for file_path in model_files:
        exists = os.path.exists(file_path)
        print(f"📁 {file_path}: {'✅ Exists' if exists else '❌ Missing'}")
        
        if exists:
            size = os.path.getsize(file_path)
            size_mb = round(size / (1024 * 1024), 2)
            print(f"   Size: {size_mb} MB")
            
            # Check if it's an LFS pointer file
            if size < 1000:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if 'version https://git-lfs.github.com/spec/v1' in content:
                            print(f"   ⚠️  LFS pointer file detected!")
                            print(f"   Content preview: {content[:100]}...")
                            all_good = False
                        else:
                            print(f"   ✅ Small but valid file")
                except:
                    print(f"   ❌ Cannot read file")
                    all_good = False
            else:
                print(f"   ✅ Good size for model file")
                
            # Test file integrity
            integrity = check_file_integrity(file_path)
            print(f"   Integrity check: {'✅ Pass' if integrity else '❌ Fail'}")
            if not integrity:
                all_good = False
        else:
            all_good = False
        
        print()
    
    return all_good

def test_git_lfs_status():
    """Test Git LFS status"""
    print("🔍 Testing Git LFS status...")
    
    try:
        import subprocess
        
        # Check if git lfs is available
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git LFS is available")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print("❌ Git LFS not available")
            return False
        
        # Check LFS tracked files
        result = subprocess.run(['git', 'lfs', 'ls-files'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lfs_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            print(f"📋 LFS tracked files: {len(lfs_files)}")
            for file_line in lfs_files:
                if file_line.strip():
                    print(f"   {file_line}")
        
        # Check LFS status
        result = subprocess.run(['git', 'lfs', 'status'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("📊 LFS Status:")
            print(result.stdout)
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking Git LFS: {e}")
        return False

def test_model_loading():
    """Test actual model loading"""
    print("🔍 Testing model loading...")
    
    try:
        from ultralytics import YOLO
        
        model_files = [
            ("best_fracture_yolov8.pt", "Fracture Detection"),
            ("best_classifier.pt", "Pneumonia Classification"),
            ("best_detection.pt", "Pneumonia Detection")
        ]
        
        successful_loads = 0
        
        for file_path, description in model_files:
            print(f"🔄 Loading {description}...")
            
            if not os.path.exists(file_path):
                print(f"   ❌ File not found: {file_path}")
                continue
            
            try:
                model = YOLO(file_path)
                # Test basic functionality
                names = model.names
                print(f"   ✅ Successfully loaded {description}")
                print(f"   📋 Model classes: {len(names) if names else 0}")
                successful_loads += 1
                
            except Exception as e:
                print(f"   ❌ Failed to load {description}: {e}")
        
        print(f"\n📊 Summary: {successful_loads}/{len(model_files)} models loaded successfully")
        return successful_loads == len(model_files)
        
    except Exception as e:
        print(f"❌ Error in model loading test: {e}")
        return False

def test_environment():
    """Test environment and dependencies"""
    print("🔍 Testing environment...")
    
    print(f"🐍 Python version: {sys.version}")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"🌍 Environment variables:")
    
    env_vars = [
        'FRACTURE_MODEL_PATH',
        'PNEUMONIA_CLASSIFIER_PATH', 
        'PNEUMONIA_DET_MODEL_PATH',
        'GEMINI_API_KEY'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        # Mask API key for security
        if 'API_KEY' in var and value != 'Not set':
            value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
        print(f"   {var}: {value}")
    
    print(f"\n📦 Key dependencies:")
    try:
        import ultralytics
        print(f"   ultralytics: {ultralytics.__version__}")
    except ImportError:
        print(f"   ultralytics: ❌ Not installed")
    
    try:
        import streamlit
        print(f"   streamlit: {streamlit.__version__}")
    except ImportError:
        print(f"   streamlit: ❌ Not installed")

def main():
    """Run all tests"""
    print("🩺 ScanX Model Loading Diagnostics")
    print("=" * 50)
    
    # Test environment
    test_environment()
    print("\n" + "=" * 50)
    
    # Test file existence
    files_ok = test_file_existence()
    print("=" * 50)
    
    # Test Git LFS
    lfs_ok = test_git_lfs_status()
    print("\n" + "=" * 50)
    
    # Test model loading
    models_ok = test_model_loading()
    print("\n" + "=" * 50)
    
    # Final summary
    print("📊 FINAL SUMMARY:")
    print(f"   Files: {'✅ OK' if files_ok else '❌ ISSUES'}")
    print(f"   Git LFS: {'✅ OK' if lfs_ok else '❌ ISSUES'}")
    print(f"   Models: {'✅ OK' if models_ok else '❌ ISSUES'}")
    
    if files_ok and lfs_ok and models_ok:
        print("\n🎉 All tests passed! Models should work on Streamlit Cloud.")
    else:
        print("\n⚠️  Some issues detected. Check the details above.")
        
        if not lfs_ok:
            print("\n💡 To fix Git LFS issues:")
            print("   1. Install Git LFS: brew install git-lfs")
            print("   2. Initialize: git lfs install") 
            print("   3. Pull LFS files: git lfs pull")
        
        if not files_ok:
            print("\n💡 To fix file issues:")
            print("   1. Ensure model files are tracked by LFS")
            print("   2. Run: git lfs pull")
            print("   3. Verify file sizes are > 1MB")

if __name__ == "__main__":
    main()
