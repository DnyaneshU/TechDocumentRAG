import os
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

# Absolute path to your project folder (where parser.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute path to docs folder
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Output folder for parsed and chunked text files
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_pdf(file_path):
    """Parse PDF using unstructured.partition.pdf"""
    try:
        elements = partition_pdf(filename=file_path)
        # Join text elements
        text = "\n".join([el.text for el in elements if hasattr(el, "text")])
        return text
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return ""

def parse_all_docs():
    for filename in tqdm(os.listdir(DOCS_DIR)):
        file_path = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(file_path):
            continue  # skip folders
        if not filename.lower().endswith(".pdf"):
            continue  # skip non-pdfs
        
        print(f"Parsing: {filename}")
        text = parse_pdf(file_path)

        # Save parsed text as .txt in OUTPUT_DIR with same name
        output_file = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

if __name__ == "__main__":
    parse_all_docs()
