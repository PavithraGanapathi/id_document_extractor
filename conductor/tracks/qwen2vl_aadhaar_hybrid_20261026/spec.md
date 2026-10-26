# Track Specification: Rectify & Integrate Qwen2-VL-2B Drive Model for Hybrid Aadhaar Extraction

## 1. Overview

This track rectifies all architectural and runtime issues preventing the reliable use of the **Qwen2-VL-2B-Instruct** Vision-Language Model from Google Drive (`/content/drive/MyDrive/model_ocr/qwen2_vl_2b` or local Drive storage) and unifies the **Hybrid Aadhaar Field Extraction Pipeline** across both the Google Colab/Gradio notebook (`ModelFYP (2).ipynb`) and the local Streamlit web application (`app.py`).

## 2. Root Problems Addressed

1. **Incomplete Google Drive Checkpoint Crash (`FileNotFoundError`)**:
   - Previously, `os.path.exists("/content/drive/MyDrive/model_ocr/qwen2_vl_2b")` returned `True` even when `model-00001-of-00002.safetensors` was missing or incomplete in Drive, causing `Qwen2VLForConditionalGeneration.from_pretrained()` to fail immediately.
2. **Hardware Dtype Mismatch**:
   - Hardcoding `torch.bfloat16` on Tesla T4 GPUs (which lack native bfloat16 tensor cores) caused degraded performance or runtime errors.
3. **Disconnected Pseudo-Hybrid Pipeline**:
   - `ModelFYP (2).ipynb` defined regex parsers (`parse_mobile_number`, `parse_father_or_guardian_name`, `parse_holder_name`) and installed `easyocr` / `pymupdf`, but `process_identity_document()` never invoked them as fallbacks or cross-validators.
   - When Qwen2-VL absorbed `S/O`, `D/O`, `C/O`, or `W/O` into the `address` field or missed a mobile number, the Father/Guardian Name and Mobile Number returned `"Not Found"`.
4. **Single-Page PDF Truncation**:
   - `gradio_pipeline()` only processed `pages[0]`, discarding Page 2 on 2-page Front/Back Aadhaar PDFs and losing back-side fields (`Father / Guardian Name`, `Address`, `PIN`).
5. **Hardcoded Sample-Card Tokens**:
   - `clean_name_string()` and `parse_father_or_guardian_name()` contained hardcoded rules for a single test card (`Pavithra`, `Ganapathi`, `anjur`, `erode`) and overly broad 2-letter address cutoffs (`"no"`, `"po"`).

## 3. Functional Requirements

- **FR1: Qwen2-VL-2B Drive Checkpoint Validator & Auto-Repair (`verify_qwen_checkpoint` & `resolve_or_repair_qwen_model_path`)**
  - Inspect `model.safetensors.index.json` (or required weight files `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`) and `config.json` in candidate Drive paths (`/content/drive/MyDrive/model_ocr/qwen2_vl_2b`, `/content/drive/MyDrive/qwen2_vl_2b`, local Drive paths).
  - If a Drive directory exists but is missing `.safetensors` shards, automatically repair it using `huggingface_hub.snapshot_download(repo_id="Qwen/Qwen2-VL-2B-Instruct", local_dir=target_drive_path)` (or fall back cleanly to `"Qwen/Qwen2-VL-2B-Instruct"` if Drive is read-only/unmounted).
- **FR2: Hardware-Aware Qwen2-VL Inference Engine**
  - Automatically select `torch.bfloat16` on Ampere+ GPUs (`torch.cuda.is_bf16_supported()`), `torch.float16` on Tesla T4 GPUs, and `torch.float32` on CPU fallback.
  - Configure `AutoProcessor` with `min_pixels=256*28*28` and `max_pixels=1280*28*28` to preserve small Aadhaar fonts without GPU OOM.
- **FR3: True 3-Stage Hybrid Aadhaar Field Fusion (`hybrid_fuse_fields`)**
  - Extract all 7 core Aadhaar fields: **Document ID** (masked `XXXX XXXX 1234` or unmasked `1234 5678 9012`), **Holder Name**, **Date of Birth**, **Gender**, **Father / Guardian Name**, **Mobile Number**, and **Address**.
  - Cross-verify VLM JSON output with deterministic regex parsers and fallback OCR/Digital PDF text lines:
    - Extract `Father / Guardian Name` from the VLM `address` field or OCR lines if `father_or_guardian` is `"Not Found"`.
    - Validate `Mobile Number` (10-digit starting with `6–9`) across VLM output and OCR corpus while stripping 12-digit Aadhaar IDs.
    - Parse `Gender` (`Male`, `Female`, `Other`) across English, Hindi (`पुरुष` / `महिला`), and Tamil (`ஆண்` / `பெண்`).
    - Extract structured `Address` from VLM or OCR address blocks.
- **FR4: Multi-Page Front/Back Aggregation (`merge_page_records`) & Resilient Google Drive Ingestion**
  - Process all pages of uploaded or Drive-downloaded PDFs and merge non-`"Not Found"` fields across pages so Front + Back Aadhaar documents yield complete records.
  - Port resilient `download_from_gdrive()` (supporting `/file/d/`, `/uc?id=`, `/open?id=`, folder links, and magic-byte extension detection) into `ModelFYP (2).ipynb`.
- **FR5: Generalized Name & Address Parsing**
  - Remove hardcoded person/village names (`Pavithra`, `Ganapathi`, `anjur`) and prevent 2-letter words like `"no"` or `"po"` from truncating valid names unless preceded by structural address markers (`House No`, `Door No`, `H.No`, `P.O.`).

## 4. Non-Functional Requirements

- **Privacy**: Mask first 8 digits of Aadhaar ID (`XXXX XXXX 1234`) by default unless privacy masking is toggled off.
- **Testability**: >80% unit and pipeline test coverage across `test_parsers.py` and `test_pipeline.py` without requiring a live GPU for unit tests (using mocked VLM responses alongside real PDF/OCR pipeline tests).

## 5. Acceptance Criteria

1. `verify_qwen_checkpoint()` accurately detects incomplete Drive folders (missing `model-00001-of-00002.safetensors`) and triggers repair/fallback without throwing `FileNotFoundError`.
2. `hybrid_fuse_fields()` and `merge_page_records()` accurately extract all 7 Aadhaar fields across single-page and 2-page (Front + Back) documents, including extracting `Father / Guardian Name` when embedded inside an `Address` block.
3. Both `app.py` (Streamlit) and `ModelFYP (2).ipynb` (Colab/Gradio) share the rectified hybrid pipeline and pass all automated tests in `test_parsers.py` and `test_pipeline.py`.

## 6. Out of Scope

- Training or fine-tuning Qwen2-VL-2B weights from scratch (this track focuses on loading the existing/Drive Qwen2-VL-2B model and hybrid inference).
- Integration with paid external cloud OCR APIs (OpenAI, Google Cloud Vision, AWS Textract).
