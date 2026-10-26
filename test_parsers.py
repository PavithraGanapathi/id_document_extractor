import os
import json
import re
import sys
import tempfile
from typing import List, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import app


def run_tests():
    print("Running parser and Qwen2-VL Drive checkpoint unit tests...")

    # 1. Test 12-digit ID
    id_tests = [
        ("Aadhaar No: 2345 6789 0123", "XXXX XXXX 0123", "2345 6789 0123"),
        ("Identity: 9876-5432-1098", "XXXX XXXX 1098", "9876 5432 1098"),
        ("Text with 345678901234 inside", "XXXX XXXX 1234", "3456 7890 1234"),
        ("Noisy 4567.8901.2345 line", "XXXX XXXX 2345", "4567 8901 2345"),
        ("Real Card: 4351 4601 1054", "XXXX XXXX 1054", "4351 4601 1054"),
        ("Masked Card: XXXX XXXX 1054", "XXXX XXXX 1054", "XXXX XXXX 1054"),
        ("Masked Star: **** **** 1054", "XXXX XXXX 1054", "XXXX XXXX 1054"),
    ]
    for text, exp_masked, exp_raw in id_tests:
        masked, raw = app.parse_document_id(text, [text])
        assert masked == exp_masked and raw == exp_raw, f"Failed ID: {text} -> got {masked}, {raw}"
    print("✓ Document ID tests passed")

    # 2. Test DOB
    dob_tests = [
        ("DOB: 12/09/2005", "12/09/2005"),
        ("DOB : 12-09-2005", "12/09/2005"),
        ("DOB: 12|09|2005", "12/09/2005"),
        ("DOB: 12 I 09 I 2005", "12/09/2005"),
        ("Date of Birth: 05.11.1998", "05/11/1998"),
        ("YOB: 1985", "1985"),
        ("Year of Birth : 1970", "1970"),
        ("Girt Jnatdo@ 12/097006 64815/", "12/09/2006"),
        ("JnatdoA: 1210972006", "12/09/2006"),
        ("DOB: 12 Sep 2006", "12/09/2006"),
        ("Date of Birth: 14-Aug-1995", "14/08/1995"),
    ]
    for text, exp in dob_tests:
        res = app.parse_date_of_birth(text, [text])
        assert res == exp, f"Failed DOB: {text} -> got {res}, expected {exp}"
    print("✓ Date of Birth tests passed")

    # 3. Test Mobile (Collision resistance & OCR typos)
    mob_tests = [
        ("Contact: 9876543210 and ID 2345 6789 0123", "2345 6789 0123", "9876543210"),
        ("Mob: +91 8765432109", "", "8765432109"),
        ("Mub 9360334347 ELLmailh A15", "4351 4601 1054", "9360334347"),
        ("Mobile: 93603 34347", "", "9360334347"),
        ("Only 12 digit 987654321098 here", "9876 5432 1098", "Not Found"),
        ("Standalone 7654321098 in note", "", "7654321098"),
        ("Standalone 76543 21098 in note", "", "7654321098"),
    ]
    for text, did, exp in mob_tests:
        res = app.parse_mobile_number(text, [text], did)
        assert res == exp, f"Failed Mobile: {text} -> got {res}, expected {exp}"
    print("✓ Mobile Number tests passed")

    # 4. Test Father / Guardian Name (generalized without hardcoded village/person names)
    father_tests = [
        (["C/O: Ramesh Kumar, House No 14, Main Road"], "Ramesh Kumar"),
        (["C/0 Suresh Sharma House No 42"], "Suresh Sharma"),
        (["CIO: Jagdish Prasad, Door No 5"], "Jagdish Prasad"),
        (["S/O: Mahesh Chandra, Flat 201"], "Mahesh Chandra"),
        (["S/0: Rajesh Gupta, Colony"], "Rajesh Gupta"),
        (["SO: Anita Roy, Village Rampur"], "Anita Roy"),
        (["D/O: Sunita Verma, Dist Erode"], "Sunita Verma"),
        (["D/0: Kavita Patel, Near Bus Stand"], "Kavita Patel"),
        (["D/O: Karthik Subramanian, 48 Anna Nagar, Madurai - 625020"], "Karthik Subramanian"),
        (["W/O: Vikram Singh, Pin 560001"], "Vikram Singh"),
        (["Father's Name: Prakash Patel"], "Prakash Patel"),
        (["C/O:", "Mukesh Ambani", "Antilia, Altamount Road"], "Mukesh Ambani"),
    ]
    for lines, exp in father_tests:
        res = app.parse_father_or_guardian_name(lines)
        assert res == exp, f"Failed Father: {lines} -> got '{res}', expected '{exp}'"

    garbage_lines = ["oocrana) DoiLoliL Gmirinmla %atosslne 4r Moyrpluoinmo] Biom RAVULL"]
    res_garbage = app.parse_father_or_guardian_name(garbage_lines)
    assert res_garbage == "Not Found", f"Garbage falsely matched: {res_garbage}"
    print("✓ Father/Guardian tests passed")

    # 5. Test Holder Name
    holder_tests = [
        (["PAVITHRA G", "Girt Jnatdo@ 12/097006", "Female"], "Pavithra G"),
        (["RAMESH K", "DOB: 15/08/1990"], "Ramesh K"),
        (["AARAV SHARMA", "Male"], "Aarav Sharma"),
        (["ANANYA KRISHNAN", "DOB: 12/09/2006", "FEMALE"], "Ananya Krishnan"),
    ]
    for lines, exp in holder_tests:
        res = app.parse_holder_name(lines)
        assert res == exp, f"Failed Holder: {lines} -> got '{res}', expected '{exp}'"
    print("✓ Holder Name tests passed")

    # 6. Test Gender & Address Parsers
    assert app.parse_gender("DOB: 14/08/1995\nMALE", ["DOB: 14/08/1995", "MALE"]) == "Male"
    assert app.parse_gender("महिला / FEMALE", ["महिला / FEMALE"]) == "Female"
    assert app.parse_gender("ஆண் / Male", ["ஆண் / Male"]) == "Male"
    addr_lines = [
        "Address:",
        "S/O: Ramesh Kumar Sharma",
        "House No 45, Gandhi Road, Sector 4",
        "New Delhi - 110001"
    ]
    parsed_addr = app.parse_address("\n".join(addr_lines), addr_lines)
    assert "House No 45" in parsed_addr and "110001" in parsed_addr, f"Failed Address: {parsed_addr}"
    print("✓ Gender and Address parser tests passed")

    # 7. Test Qwen2-VL-2B Google Drive Checkpoint Integrity Validator
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Case A: Incomplete Drive directory (like ModelFYP (2)_backup.ipynb where model-00001-of-00002.safetensors is missing)
        incomplete_dir = os.path.join(tmp_dir, "qwen2_vl_2b_incomplete")
        os.makedirs(incomplete_dir, exist_ok=True)
        with open(os.path.join(incomplete_dir, "config.json"), "w") as f:
            json.dump({"model_type": "qwen2_vl"}, f)
        with open(os.path.join(incomplete_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors"
                }
            }, f)

        is_valid, missing_files = app.verify_qwen_checkpoint(incomplete_dir)
        assert not is_valid, "Incomplete checkpoint should fail validation"
        assert "model-00001-of-00002.safetensors" in missing_files, f"Expected missing shard 1, got {missing_files}"

        # Case B: Complete Drive directory with both shards non-empty
        complete_dir = os.path.join(tmp_dir, "qwen2_vl_2b_complete")
        os.makedirs(complete_dir, exist_ok=True)
        with open(os.path.join(complete_dir, "config.json"), "w") as f:
            json.dump({"model_type": "qwen2_vl"}, f)
        with open(os.path.join(complete_dir, "preprocessor_config.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(complete_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors"
                }
            }, f)
        for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
            with open(os.path.join(complete_dir, shard), "wb") as f:
                f.write(b"DUMMY_SAFETENSORS_WEIGHTS_CONTENT")

        is_valid_complete, missing_complete = app.verify_qwen_checkpoint(complete_dir)
        assert is_valid_complete and len(missing_complete) == 0, f"Complete checkpoint failed: {missing_complete}"

        # Case C: resolve_or_repair_qwen_model_path falls back when auto_repair=False and Drive dir is incomplete
        resolved = app.resolve_or_repair_qwen_model_path(
            candidate_paths=[incomplete_dir],
            default_hub_id="Qwen/Qwen2-VL-2B-Instruct",
            auto_repair=False
        )
        assert resolved == "Qwen/Qwen2-VL-2B-Instruct", f"Expected fallback Hub ID, got {resolved}"

        resolved_valid = app.resolve_or_repair_qwen_model_path(
            candidate_paths=[incomplete_dir, complete_dir],
            default_hub_id="Qwen/Qwen2-VL-2B-Instruct",
            auto_repair=False
        )
        assert resolved_valid == complete_dir, f"Expected complete_dir, got {resolved_valid}"
    print("✓ Qwen2-VL-2B Drive checkpoint validator tests passed")

    # 8. Test Hybrid Fusion (VLM JSON + Cross-Field Address/OCR Fallback) & Multi-Page Merging
    vlm_page1 = {
        "document_id": "4351 4601 1054",
        "holder_name": "PAVITHRA G",
        "date_of_birth": "12/097006",
        "gender": "FEMALE",
        "father_or_guardian": "Not Found",
        "mobile_number": "Not Found",
        "address": "Not Found"
    }
    fused_p1 = app.hybrid_fuse_fields(
        vlm_data=vlm_page1,
        ocr_text="PAVITHRA G\nDOB: 12/09/2006\nFemale\nMobile: 93603 34347\n4351 4601 1054",
        ocr_lines=["PAVITHRA G", "DOB: 12/09/2006", "Female", "Mobile: 93603 34347", "4351 4601 1054"],
        privacy_mask=True
    )
    assert fused_p1["Document ID"] == "XXXX XXXX 1054"
    assert fused_p1["Raw Document ID"] == "4351 4601 1054"
    assert fused_p1["Holder Name"] == "Pavithra G"
    assert fused_p1["Date of Birth"] == "12/09/2006"
    assert fused_p1["Gender"] == "Female"
    assert fused_p1["Mobile Number"] == "9360334347"

    # Page 2 (Back of Aadhaar card where Father/Guardian is embedded inside Address)
    vlm_page2 = {
        "document_id": "4351 4601 1054",
        "holder_name": "Not Found",
        "date_of_birth": "Not Found",
        "gender": "Not Found",
        "father_or_guardian": "Not Found",
        "mobile_number": "Not Found",
        "address": "D/O: Ganapathi, 48 Anna Nagar, Erode, Tamil Nadu - 638001"
    }
    fused_p2 = app.hybrid_fuse_fields(
        vlm_data=vlm_page2,
        ocr_text="Address: D/O: Ganapathi, 48 Anna Nagar, Erode, Tamil Nadu - 638001",
        ocr_lines=["Address: D/O: Ganapathi, 48 Anna Nagar, Erode, Tamil Nadu - 638001"],
        privacy_mask=True
    )
    assert fused_p2["Father / Guardian Name"] == "Ganapathi", f"Expected Ganapathi from address fallback, got {fused_p2['Father / Guardian Name']}"

    merged_doc = app.merge_page_records([fused_p1, fused_p2])
    assert merged_doc["Document ID"] == "XXXX XXXX 1054"
    assert merged_doc["Holder Name"] == "Pavithra G"
    assert merged_doc["Date of Birth"] == "12/09/2006"
    assert merged_doc["Gender"] == "Female"
    assert merged_doc["Mobile Number"] == "9360334347"
    assert merged_doc["Father / Guardian Name"] == "Ganapathi"
    assert "638001" in merged_doc["Address"]
    print("✓ Hybrid VLM+OCR fusion and multi-page Front/Back merging tests passed")

    print("\nAll parser and Qwen2-VL checkpoint tests passed successfully!")


if __name__ == "__main__":
    run_tests()
