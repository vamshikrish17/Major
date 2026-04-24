import fitz
import sys

def extract_headers(pdf_path):
    print(f"Analyzing {pdf_path}...\n")
    doc = fitz.open(pdf_path)
    
    # We will look for lines that look like standard IEEE or journal headers
    # E.g. "1. INTRODUCTION", "I. INTRODUCTION", "A. ...", etc.
    # We'll just print out bolder/larger text or structured formats.
    
    for i in range(min(4, len(doc))): # Check first 4 pages for structure
        page = doc[i]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        # Simple heuristic for section titles: Uppercase or starts with numeral/roman Numeral
                        if text and (s["flags"] & 20) or s["size"] > 10.5: # 20 is bold usually, or size > standard body text
                            if len(text) > 3 and not text.isnumeric():
                                print(f"[Page {i+1}] {text}")

if __name__ == "__main__":
    extract_headers("2026084195 (1).pdf")
