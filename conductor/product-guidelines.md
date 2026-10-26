# Product Guidelines

## 1. Privacy & Security Principles

- **Privacy-by-Default Masking**: All 12-digit Aadhaar numbers must default to masked display (`XXXX XXXX 1234`), revealing the full 12-digit number (`1234 5678 9012`) only when the user explicitly disables the privacy mask toggle.
- **Zero External Third-Party Cloud Vision APIs**: All inference runs strictly on the user's local/Colab GPU and CPU using local/Drive-hosted **Qwen2-VL-2B** weights, **EasyOCR**, and **PyMuPDF**.

## 2. Extraction Accuracy & Reliability Standards

- **Drive Checkpoint Verification Before Load**: Never blindly pass a Google Drive folder path to `from_pretrained` without verifying that all `.safetensors` weight shards exist and are non-empty. Provide clear diagnostic messages and automatic recovery if shards are missing.
- **No Single-Page or Cropping Truncation**: Multi-page PDFs (such as Front + Back Aadhaar scans) must have all pages processed and merged so that back-side fields (`Father / Guardian Name`, `Address`) are never lost.
- **Deterministic Post-Validation**: Every field returned by Qwen2-VL-2B must be validated and normalized by rule-based parsers (`DD/MM/YYYY` for DOB, 10-digit `6–9` validation for Mobile Number with zero Aadhaar ID collision, and prefix/address cleanup for Father/Guardian Name).

## 3. User Experience (UX) & Interface Tone

- **Clear Visual Hierarchy**: Present extracted fields in high-contrast summary cards with explicit status indicators (`Found` / `Valid` vs. `Not Found` / `Missing`).
- **Human-in-the-Loop Editability**: Provide interactive data tables (`st.data_editor` / `gr.DataFrame`) and one-click styled Excel (`.xlsx`) export (`openpyxl`).
- **Transparent Diagnostics**: Expose extraction mode (`Qwen2-VL-2B + Hybrid Rules`, `Digital PDF Vector Stream`, `OCR Dual-Pass`), processing time, and raw OCR/VLM output in a collapsible debug inspector.
