# Product Definition: Hybrid Aadhaar & Identity Document Field Extractor

## 1. Product Vision & Overview

**Hybrid Aadhaar & Identity Document Field Extractor (Qwen2-VL-2B + OCR)** is a multi-modal document extraction platform designed to accurately extract structured personal identity fields from Indian Aadhaar cards and national identity documents. It combines the deep visual-semantic comprehension of the **Qwen2-VL-2B-Instruct** Vision-Language Model (loaded from Google Drive or local checkpoint storage) with a deterministic, noise-tolerant **PyMuPDF + OpenCV + EasyOCR + Regex** validation pipeline.

## 2. Target Identity Fields

1. **12-Digit Document ID (Aadhaar Number)**: Supports continuous, spaced (`4-4-4`), hyphenated, and pre-masked formats (`XXXX XXXX 1234`), with toggleable privacy masking.
2. **Holder Name**: Full name of the cardholder, accurately distinguished from organizational headers, bilingual regional labels, and OCR noise.
3. **Date of Birth (DOB) / Year of Birth (YOB)**: Standardized to `DD/MM/YYYY` or `YYYY`, resilient to separator noise and font ligature merges.
4. **Gender**: Standardized to `Male`, `Female`, or `Other` across English and regional scripts.
5. **Father / Guardian / Relation Name**: Extracted from explicit labels or embedded address prefixes (`S/O`, `D/O`, `C/O`, `W/O`, `Father's Name`) on either the front or back of the Aadhaar card, cleanly stripped of downstream address tokens.
6. **Mobile Number**: 10-digit Indian mobile number starting with `6–9`, strictly avoiding collisions with 12-digit Aadhaar numbers or dates.
7. **Residential Address**: Full address extracted from back-side cards or full-page e-Aadhaar letters.

## 3. Core Architecture & Capabilities

- **Google Drive Qwen2-VL-2B Model Loader & Validator**: Verifies `.safetensors` shard integrity in Google Drive (`/content/drive/MyDrive/model_ocr/qwen2_vl_2b` or local Drive path) and automatically repairs/downloads missing shards from Hugging Face Hub (`Qwen/Qwen2-VL-2B-Instruct`) while optimizing GPU dtype (`float16` on T4, `bfloat16` on Ampere+).
- **Multi-Modal Document Ingestion**: Accepts single/multi-page PDFs (300 DPI native PyMuPDF rendering + direct digital vector extraction), images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`), and public Google Drive file/folder URLs via `gdown` with magic-byte format detection.
- **True Hybrid VLM + OCR + Regex Fusion**: Uses Qwen2-VL-2B for visual layout and semantic field extraction, cross-verified and backed up by dual-pass OpenCV + EasyOCR and domain-specific regex parsers across all pages (Front & Back).
- **Interactive Web Interfaces & Reporting**: Provides both a **Gradio Web UI** (for Colab/cloud GPU notebooks) and a **Streamlit Web App** (`app.py`), complete with structured field summary cards, editable data tables, and styled `.xlsx` Excel export via `openpyxl`.
