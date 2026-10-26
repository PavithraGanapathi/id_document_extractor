import sys
import os
import cv2
import numpy as np
import fitz
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import app


def test_full_pipeline():
    print("--- 1. Testing Single-Page Digital PDF Extraction ---")
    pdf_path = "test_digital_id.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    text_content = (
        "Unique Identification Authority of India\n"
        "Government of India\n"
        "To\n"
        "Aarav Sharma\n"
        "S/O: Ramesh Kumar Sharma\n"
        "House No 45, Gandhi Road, Sector 4\n"
        "New Delhi, PIN: 110001\n"
        "DOB: 14/08/1995\n"
        "Male\n"
        "Mobile: 9876543210\n"
        "Help: 1947\n"
        "4589 1234 5678\n"
    )
    page.insert_text((50, 72), text_content, fontsize=12)
    doc.save(pdf_path)
    doc.close()

    dig_text, dig_lines = app.extract_pdf_digital_text(pdf_path)
    fused = app.hybrid_fuse_fields(
        vlm_data=None,
        ocr_text=dig_text,
        ocr_lines=dig_lines,
        privacy_mask=True,
    )

    print("Single-Page Digital PDF Results:")
    for k, v in fused.items():
        print(f"  {k}: {v}")

    assert fused["Raw Document ID"] == "4589 1234 5678", f"ID mismatch: {fused['Raw Document ID']}"
    assert fused["Document ID"] == "XXXX XXXX 5678", f"Masked ID mismatch: {fused['Document ID']}"
    assert fused["Date of Birth"] == "14/08/1995", f"DOB mismatch: {fused['Date of Birth']}"
    assert fused["Gender"] == "Male", f"Gender mismatch: {fused['Gender']}"
    assert fused["Mobile Number"] == "9876543210", f"Mobile mismatch: {fused['Mobile Number']}"
    assert fused["Father / Guardian Name"] == "Ramesh Kumar Sharma", f"Father mismatch: {fused['Father / Guardian Name']}"
    assert "110001" in fused["Address"], f"Address mismatch: {fused['Address']}"
    print("✓ Single-page Digital PDF pipeline verified!")

    print("\n--- 2. Testing 2-Page Front/Back Aadhaar PDF Aggregation ---")
    two_page_pdf = "test_two_page_aadhaar.pdf"
    doc2 = fitz.open()
    p_front = doc2.new_page(width=595, height=842)
    p_front.insert_text(
        (50, 72),
        "Government of India\nPAVITHRA G\nDOB: 12/09/2006\nFEMALE\n4351 4601 1054\n",
        fontsize=12,
    )
    p_back = doc2.new_page(width=595, height=842)
    p_back.insert_text(
        (50, 72),
        "Unique Identification Authority of India\nAddress:\nD/O: Ganapathi, 48 Anna Nagar, Erode\nTamil Nadu - 638001\nMobile: 93603 34347\n4351 4601 1054\n",
        fontsize=12,
    )
    doc2.save(two_page_pdf)
    doc2.close()

    pages_2p = app.load_document_pages(two_page_pdf, "test_two_page_aadhaar.pdf")
    assert len(pages_2p) == 2, f"Expected 2 pages, got {len(pages_2p)}"

    doc2_read = fitz.open(two_page_pdf)
    page_records = []
    for page_obj in doc2_read:
        p_txt = page_obj.get_text("text")
        p_lns = [l.strip() for l in p_txt.splitlines() if l.strip()]
        page_records.append(app.hybrid_fuse_fields(vlm_data=None, ocr_text=p_txt, ocr_lines=p_lns, privacy_mask=True))
    doc2_read.close()

    merged_2p = app.merge_page_records(page_records)
    assert merged_2p["Document ID"] == "XXXX XXXX 1054"
    assert merged_2p["Holder Name"] == "Pavithra G"
    assert merged_2p["Date of Birth"] == "12/09/2006"
    assert merged_2p["Gender"] == "Female"
    assert merged_2p["Father / Guardian Name"] == "Ganapathi"
    assert merged_2p["Mobile Number"] == "9360334347"
    assert "638001" in merged_2p["Address"]
    print("✓ 2-Page Front/Back Aadhaar PDF aggregation verified!")

    print("\n--- 3. Testing 7-Field Excel Export ---")
    records = [{
        "File Target": "test_two_page_aadhaar.pdf",
        "Document ID": merged_2p["Document ID"],
        "Holder Name": merged_2p["Holder Name"],
        "Date of Birth": merged_2p["Date of Birth"],
        "Gender": merged_2p["Gender"],
        "Father / Guardian Name": merged_2p["Father / Guardian Name"],
        "Mobile Number": merged_2p["Mobile Number"],
        "Address": merged_2p["Address"],
        "Extraction Mode": "Hybrid Qwen2-VL + OCR",
        "Status": "Success",
    }]
    xlsx_bytes = app.export_records_to_excel(records)
    assert len(xlsx_bytes) > 1000, "Excel output empty"
    print(f"✓ Excel export generated ({len(xlsx_bytes)} bytes)!")

    for f in [pdf_path, two_page_pdf]:
        if os.path.exists(f):
            os.remove(f)

    print("\nAll pipeline integration tests passed successfully!")


if __name__ == "__main__":
    test_full_pipeline()
