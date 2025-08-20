#!/bin/bash
# Streamlit Cloud startup script to ensure LFS files are available

echo "🔄 Checking Git LFS files..."

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS not found"
    exit 1
fi

# Initialize LFS if needed
git lfs install

# Pull LFS files
echo "📥 Pulling LFS files..."
git lfs pull

# Verify model files exist and have correct sizes
echo "🔍 Verifying model files..."

check_file() {
    local file=$1
    local min_size=$2
    
    if [[ -f "$file" ]]; then
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        if [[ $size -gt $min_size ]]; then
            echo "✅ $file: ${size} bytes"
            return 0
        else
            echo "⚠️  $file: Too small (${size} bytes)"
            return 1
        fi
    else
        echo "❌ $file: Not found"
        return 1
    fi
}

# Check each model file (minimum expected sizes in bytes)
check_file "best_fracture_yolov8.pt" 5000000    # ~5MB
check_file "best_classifier.pt" 9000000         # ~9MB  
check_file "best_detection.pt" 20000000         # ~20MB

echo "🎉 LFS setup complete!"
