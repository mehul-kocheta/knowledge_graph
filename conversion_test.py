from llama_parse import LlamaParse
from pathlib import Path

# =========================
# CONFIG
# =========================

PDF_PATH = r"C:\Users\mehul\Downloads\2407.00681v1.pdf"
OUTPUT_FILE = "parsed_output.md"

# =========================
# PARSER
# =========================

parser = LlamaParse(
    result_type="markdown",   # options: "markdown", "text"
    verbose=True
)

# =========================
# PARSE DOCUMENT
# =========================

print("Parsing PDF...")

documents = parser.load_data(PDF_PATH)

print("Parsing completed.")

# =========================
# SAVE RESULT
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
        f.write("\n\n")

print(f"Saved parsed output → {OUTPUT_FILE}")