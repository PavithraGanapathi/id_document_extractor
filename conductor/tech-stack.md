# Technology Stack

## 1. Core Language & Runtime
- **Python**: `3.10+` (`3.12` in local virtual environment, `3.10+` on Google Colab GPU runtime)

## 2. Vision-Language Model (VLM) & Deep Learning
- **Model**: `Qwen/Qwen2-VL-2B-Instruct` (loaded from Google Drive `/content/drive/MyDrive/model_ocr/qwen2_vl_2b`, local Drive folder, or Hugging Face Hub)
- **PyTorch (`torch`)**: GPU-accelerated tensor computation (`torch.float16` on Tesla T4, `torch.bfloat16` on Ampere+ GPUs)
- **Hugging Face Transformers (`transformers>=4.45.0`)**: `Qwen2VLForConditionalGeneration` and `AutoProcessor`
- **Accelerate (`accelerate>=0.26.0`)**: Automatic device mapping (`device_map="auto"`) and `low_cpu_mem_usage=True`
- **Qwen-VL Utilities (`qwen-vl-utils>=0.0.8`)**: `process_vision_info` for multi-modal image resolution & patch preprocessing
- **Hugging Face Hub (`huggingface_hub`)**: Checkpoint shard verification and automated `snapshot_download` repair

## 3. Document Processing, Computer Vision & OCR
- **PyMuPDF (`fitz>=1.23.0`)**: Direct digital PDF text vector stream extraction and native 300 DPI page rasterization
- **pdf2image (`pdf2image>=1.17.0`)**: Fallback PDF rasterization
- **OpenCV (`opencv-python-headless>=4.8.0`)**: Non-destructive bicubic upscaling, auto-rotation (90°/180°/270°), and CLAHE contrast enhancement
- **EasyOCR (`easyocr>=1.7.0`)**: Multi-lingual (`en`, `ta`, `hi`) text detection and bounding-box extraction
- **Pillow (`Pillow>=10.0.0`)**: Image manipulation for VLM and PDF pipelines

## 4. Web Interfaces, Ingestion & Data Export
- **Streamlit (`streamlit>=1.30.0`)**: Local interactive web application (`app.py`)
- **Gradio (`gradio>=4.40.0`) & `nest-asyncio`**: Cloud/Colab interactive web UI (`ModelFYP (2).ipynb`)
- **gdown (`gdown>=5.0.0`)**: Resilient Google Drive file and folder downloader
- **Pandas (`pandas>=2.0.0`) & OpenPyXL (`openpyxl>=3.1.0`)**: Structured tabular data representation and styled `.xlsx` spreadsheet export
- **Matplotlib (`matplotlib`)**: FYP performance evaluation and accuracy comparison charts
