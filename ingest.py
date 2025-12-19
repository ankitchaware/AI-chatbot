"""
PDF Ingestion Pipeline for NABARD Annual Reports
Extracts text, chunks it, and creates vector embeddings
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pdfplumber
try:
    import camelot 
    HAS_CAMELOT = True
except ImportError:
    camelot = None
    HAS_CAMELOT = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.vector_store import create_vector_store


def extract_tables_to_text(pdf_path: str) -> str:
    """
    Extract tables from a PDF using Camelot and return them as plain text.
    Falls back silently if Camelot or its dependencies are missing.
    """
    if not HAS_CAMELOT:
        print("Camelot not installed; skipping table extraction.")
        return ""

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if tables.n == 0:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

        extracted_tables = []
        for idx, table in enumerate(tables):
            print(f"   Extracting table {idx + 1}/{tables.n} from {os.path.basename(pdf_path)}")
            df = table.df
            rows = [", ".join(row) for row in df.values.tolist()]
            extracted_tables.append("\n".join(rows))

        joined_tables = "\n\n".join(extracted_tables)
        if joined_tables:
            print(f"   ✓ Extracted tables text length: {len(joined_tables):,} chars")
        return joined_tables

    except Exception as e:
        print(f"   ✗ Table extraction skipped for {os.path.basename(pdf_path)}: {e}")
        return ""

def extract_text_from_pdfs():
    """
    Extract text from all PDF files in data/raw_pdfs/
    
    Returns:
        Combined text from all PDFs
    """
    all_text = ""
    pdf_folder = os.path.join(project_root, "data", "raw_pdfs")
    
    if not os.path.exists(pdf_folder):
        print(f"PDF folder not found: {pdf_folder}")
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder}")
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    for filename in pdf_files:
        print(f"\nProcessing file: {filename}")
        path = os.path.join(pdf_folder, filename)
        print(f"Extracting: {filename}...")
        
        try:
            with pdfplumber.open(path) as pdf:
                print(f"   Total pages: {len(pdf.pages)}")
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n\n"
                    
                    if page_num % 10 == 0:
                        print(f"   Processed {page_num} pages...")

            # Append table text (if any) after full PDF text extraction
            table_text = extract_tables_to_text(path)
            if table_text:
                all_text += table_text + "\n\n"

            print(f"✓ Completed: {filename}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            continue
    
    print(f"\n✓ Total extracted text: {len(all_text):,} characters")
    return all_text

def chunk_text(text):
    print("Chunking text into smaller pieces...")
    """
    Split text into overlapping chunks for better retrieval
    
    Args:
        text: Combined text from all PDFs
    
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    print(f"✓ Created {len(chunks)} chunks")
    
    return chunks

def main():

    """Main ingestion pipeline"""
    print("=" * 60)
    print("NABARD Reports Ingestion Pipeline")
    print("=" * 60)
    
    try:
        combined_text = extract_text_from_pdfs()
        
        if not combined_text.strip():
            raise ValueError("No text extracted from PDFs!")
        
        chunks = chunk_text(combined_text)
        create_vector_store(chunks)
        
        print("\n" + "=" * 60)
        print("✓ INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("You can now run: streamlit run frontend/app.py")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"\n✗ Error during ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting ingestion pipeline...")
    main()







