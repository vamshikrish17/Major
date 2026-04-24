import fitz  # PyMuPDF
import glob
import os

pdf_files = glob.glob("Paper/*.pdf")

with open("scratch_analysis.txt", "w", encoding="utf-8") as f:
    for pdf_path in pdf_files:
        f.write(f"=== {os.path.basename(pdf_path)} ===\n")
        try:
            doc = fitz.open(pdf_path)
            # Extract first two pages
            for i in range(min(2, len(doc))):
                f.write(doc[i].get_text())
                f.write("\n")
            
            # Extract last page
            if len(doc) > 2:
                f.write("\n[LAST PAGE]\n")
                f.write(doc[-1].get_text())
            f.write("\n" + "="*80 + "\n\n")
        except Exception as e:
            f.write(f"Error reading {pdf_path}: {e}\n")
