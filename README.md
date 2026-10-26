# 🪪 Hybrid Aadhaar & Identity Document Field Extractor (Qwen2-VL-2B + OCR)

A multi-modal identity document extraction platform that extracts all 7 core Indian Aadhaar and national identity fields (**12-Digit Document ID**, **Holder Name**, **Date of Birth / Year of Birth**, **Gender**, **Father / Guardian Name**, **Mobile Number**, and **Residential Address**) from single/multi-page PDFs, images, and Google Drive links.

Supports both:
- **Colab / Cloud GPU Notebook ([`ModelFYP (2).ipynb`](./ModelFYP%20(2).ipynb))** with an interactive **Gradio** web interface.
- **Local Web Application ([`app.py`](./app.py))** with an interactive **Streamlit** interface.

---

## ⚡ Key Architectural Solutions & Rectifications

1. **Google Drive Qwen2-VL-2B Checkpoint Validator & Auto-Repair (`verify_qwen_checkpoint`)**:
   - Inspects `model.safetensors.index.json` and `config.json` in `/content/drive/MyDrive/model_ocr/qwen2_vl_2b` (or local Drive path) to verify that both `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors` exist and are non-empty before calling `from_pretrained()`.
   - Automatically repairs incomplete Google Drive model folders using `huggingface_hub.snapshot_download("Qwen/Qwen2-VL-2B-Instruct")`, eliminating `FileNotFoundError` crashes.
2. **Hardware-Aware Dtype Selection**:
   - Dynamically uses `torch.float16` on Tesla T4 GPUs, `torch.bfloat16` on Ampere+ GPUs, and `torch.float32` on CPU fallback.
3. **True 3-Stage Hybrid Aadhaar Field Fusion (`hybrid_fuse_fields`)**:
   - **Stage 1 (Digital PDF Stream)**: Extracts embedded vector text from e-Aadhaar PDFs via PyMuPDF (`fitz`).
   - **Stage 2 (Qwen2-VL-2B Visual Comprehension)**: Extracts structured JSON across complex, noisy, or bilingual Aadhaar layouts.
   - **Stage 3 (Cross-Field & EasyOCR Fallback Validation)**:
     - Automatically extracts `Father / Guardian Name` (`S/O`, `D/O`, `C/O`, `W/O`) even when embedded inside the `Address` block on the back of an Aadhaar card.
     - Validates 10-digit Indian mobile numbers (`6–9`) across VLM and OCR text while strictly preventing collisions with 12-digit Aadhaar IDs.
4. **Multi-Page Front & Back Aggregation (`merge_page_records`)**:
   - Processes all pages of multi-page PDFs (e.g., Page 1 Front + Page 2 Back) and merges non-missing fields into a single complete record.

---

## 📦 Installation & Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Applications

### 1. Local Streamlit Web App (`app.py`)
```bash
streamlit run app.py
```

### 2. Google Colab / Gradio Notebook (`ModelFYP (2).ipynb`)
Open `ModelFYP (2).ipynb` in Google Colab (T4 GPU runtime) and run all cells sequentially.

### 3. Automated Test Suite
```bash
python test_parsers.py
python test_pipeline.py
```
