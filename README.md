# Streamlit X-ray AI Analysis App

A comprehensive Streamlit application for AI-powered X-ray analysis featuring fracture detection and pneumonia classification with real-time explanations.

## Features

- **🦴 Fracture Detection**: YOLO-based bone fracture detection with bounding boxes
- **🫁 Pneumonia Analysis**: Classification and region detection for pneumonia
- **🤖 AI Explanations**: Structured medical insights via Google Gemini
- **🔍 Visual Inspection**: Automatic X-ray image validation
- **📊 Interactive UI**: Real-time analysis with beautiful visualizations
- **☁️ Cloud Ready**: Optimized for Streamlit Cloud deployment

## Quick Start

### Local Development

1. **Navigate to the Streamlit app directory**
   ```bash
   cd app_streamlit
   ```

2. **Activate the virtual environment**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies (if not already done)**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Edit .env file with your API keys
   nano .env
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Push to GitHub**
   - Ensure your repository is updated on GitHub
   - Make sure the `app_streamlit/` folder is included

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `x-ray_project`
   - Set main file path: `app_streamlit/app.py`
   - Add secrets in the app dashboard:
     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     FRACTURE_MODEL_PATH = "../best_fracture_yolov8.pt"
     PNEUMONIA_CLASSIFIER_PATH = "../best_classifier.pt"
     PNEUMONIA_DET_MODEL_PATH = "../best_detection.pt"
     ```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI explanations | Yes |
| `FRACTURE_MODEL_PATH` | Path to fracture detection model | No (default: ../best_fracture_yolov8.pt) |
| `PNEUMONIA_CLASSIFIER_PATH` | Path to pneumonia classification model | No (default: ../best_classifier.pt) |
| `PNEUMONIA_DET_MODEL_PATH` | Path to pneumonia detection model | No (default: ../best_detection.pt) |
| `APP_LOG_LEVEL` | Logging level | No (default: INFO) |
| `ENABLE_VISION_IN_PROMPT` | Enable vision in Gemini prompts | No (default: true) |

## Usage

1. **Select Analysis Type**: Choose between fracture detection or pneumonia analysis
2. **Upload X-ray Image**: Drag and drop or browse for PNG/JPG/JPEG files
3. **View Results**: See original and annotated images side by side
4. **Review AI Analysis**: Get detailed explanations and recommendations
5. **Explore Details**: Expand sections for raw data and technical details

## Architecture

- **Frontend**: Streamlit with interactive widgets and visualizations
- **Backend**: Python with caching for model performance
- **AI Models**: YOLO (Ultralytics) + Google Gemini
- **Deployment**: Streamlit Cloud (recommended) or local hosting

## Performance Features

- **Model Caching**: `@st.cache_resource` for fast model loading
- **Optimized Dependencies**: Headless OpenCV for better performance
- **Real-time Processing**: In-memory image processing
- **Smart UI**: Progressive disclosure of information

## File Structure

```
app_streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── .streamlit/           # Streamlit configuration
│   └── config.toml       # UI theme and settings
├── venv/                 # Virtual environment
└── README.md            # This file
```

## Development Tips

1. **Model Loading**: Models are cached and loaded once per session
2. **Environment Variables**: Use `.env` for local development, secrets for cloud
3. **Image Processing**: All processing is done in memory for efficiency
4. **Error Handling**: Comprehensive error handling with user-friendly messages

## Troubleshooting

### Common Issues

1. **Models not loading**: Check file paths in environment variables
2. **Gemini API errors**: Verify API key is correctly set
3. **Memory issues**: Reduce image size or restart the app
4. **Slow performance**: Models are cached after first load

### Model Requirements

Place your trained YOLO model files in the parent directory:
- `../best_fracture_yolov8.pt` - Fracture detection model
- `../best_classifier.pt` - Pneumonia classification model
- `../best_detection.pt` - Pneumonia detection model

## Deployment Options

### Streamlit Cloud (Recommended)
- ✅ Free hosting
- ✅ Automatic updates from GitHub
- ✅ Built-in secrets management
- ✅ Easy sharing and collaboration

### Local Hosting
- ✅ Full control over environment
- ✅ No upload limits
- ✅ Private/internal use
- ❌ Requires manual maintenance

### Other Platforms
- **Heroku**: Possible but requires additional configuration
- **AWS/GCP**: More complex but highly scalable
- **Docker**: For containerized deployment

## Contributing

1. Fork the repository
2. Create a feature branch in `app_streamlit/`
3. Test thoroughly with sample images
4. Submit a pull request

## License

This project is licensed under the MIT License.
