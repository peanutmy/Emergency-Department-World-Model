"""
Convert all downloaded .docx files under emsim/docs/<category>/
into .pdf files under emsim/pdf/<category>/ using MS Word COM via docx2pdf.

Converts per-category (one Word COM instance per batch) for speed.
"""
import sys
from pathlib import Path

from docx2pdf import convert

ROOT = Path(r"D:\wmed\emsim")
DOCS = ROOT / "docs"
PDFS = ROOT / "pdf"

failed = []
total = 0
converted = 0

categories = sorted([d for d in DOCS.iterdir() if d.is_dir()])
for cat_dir in categories:
    out_dir = PDFS / cat_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for docx in sorted(cat_dir.glob("*.docx")):
        total += 1
        pdf_path = out_dir / (docx.stem + ".pdf")
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"[skip] {cat_dir.name}/{pdf_path.name}")
            converted += 1
            continue
        print(f"[conv] {cat_dir.name}/{docx.name}")
        try:
            convert(str(docx), str(pdf_path))
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                converted += 1
            else:
                failed.append(str(docx))
                print(f"  FAILED (no output): {docx}")
        except Exception as e:
            failed.append(str(docx))
            print(f"  EXCEPTION: {e}")

print(f"\nConverted {converted}/{total}")
if failed:
    print("Failed files:")
    for f in failed:
        print(" ", f)
    (ROOT / "pdf_failed.txt").write_text("\n".join(failed), encoding="utf-8")
    sys.exit(1)
