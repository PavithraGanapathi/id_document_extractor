# Implementation Plan: Rectify & Integrate Qwen2-VL-2B Drive Model for Hybrid Aadhaar Extraction

## Phase 1: Qwen2-VL-2B Drive Checkpoint Validator & Generalized Aadhaar Parsers

- [ ] Task: Write failing unit tests for Qwen2-VL Drive checkpoint shard validation, generalized Aadhaar field parsers, and cross-field fallback fusion
    - [ ] Add unit tests in `test_parsers.py` for `verify_qwen_checkpoint()` and `resolve_or_repair_qwen_model_path()` (testing missing `model-00001-of-00002.safetensors`, zero-byte shard files, valid 2-shard directories, and auto-repair fallback)
    - [ ] Add unit tests in `test_parsers.py` for `parse_gender()`, `parse_address()`, Father/Guardian extraction from Address blocks, `hybrid_fuse_fields()`, and `merge_page_records()`
- [ ] Task: Implement Qwen2-VL Drive checkpoint validator, auto-repair loader, and generalized Aadhaar regex parsers in `app.py`
    - [ ] Implement `verify_qwen_checkpoint()` and `resolve_or_repair_qwen_model_path()` in `app.py` with `model.safetensors.index.json` shard verification and `huggingface_hub.snapshot_download` repair
    - [ ] Refactor `clean_name_string()` and `parse_father_or_guardian_name()` to remove hardcoded sample-card tokens and improve address cutoff precision, and add `parse_gender()` and `parse_address()`
    - [ ] Implement `hybrid_fuse_fields()` and `merge_page_records()` to combine Qwen2-VL-2B JSON output with OCR/Digital PDF text lines and cross-field regex validation
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Unified Hybrid Pipeline in Streamlit (`app.py`) & Colab/Gradio (`ModelFYP (2).ipynb`)

- [ ] Task: Write failing integration tests for Hybrid VLM + OCR + Multi-Page PDF extraction pipeline
    - [ ] Add integration tests in `test_pipeline.py` simulating a 2-page Front/Back Aadhaar PDF (Page 1: Holder Name, DOB, Gender, Aadhaar ID; Page 2: S/O Father Name, Address, Mobile), verifying `merge_page_records()`, `hybrid_fuse_fields()`, and 7-field Excel export
- [ ] Task: Upgrade Streamlit app (`app.py`) and `requirements.txt` with Qwen2-VL-2B Drive model support and 7-field Aadhaar extraction
    - [ ] Integrate `load_qwen_vl_model()` and `extract_fields_with_qwen()` into `app.py` with sidebar configuration for Google Drive / local Qwen2-VL-2B model path and hybrid engine selection
    - [ ] Update UI cards, data editor table, and `export_records_to_excel()` in `app.py` to include all 7 Aadhaar fields (`Document ID`, `Holder Name`, `Date of Birth`, `Gender`, `Father / Guardian Name`, `Mobile Number`, `Address`)
    - [ ] Update `requirements.txt` to include `transformers`, `accelerate`, `qwen-vl-utils`, and `huggingface_hub`
- [ ] Task: Rectify and synchronize Colab/Gradio Notebook (`ModelFYP (2).ipynb`)
    - [ ] Update Steps 3 & 4 in `ModelFYP (2).ipynb` to use `verify_qwen_checkpoint()` and `resolve_or_repair_qwen_model_path()` so incomplete Google Drive folders are automatically repaired before calling `from_pretrained()`
    - [ ] Update Steps 5, 6, and 7 in `ModelFYP (2).ipynb` with resilient Google Drive download (`download_from_gdrive` with folder & magic-byte support), EasyOCR + PyMuPDF fallback, `hybrid_fuse_fields()`, and multi-page Front/Back PDF aggregation (`merge_page_records()`)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
