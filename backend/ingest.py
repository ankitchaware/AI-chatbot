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
from langchain_core.documents import Document
from backend.vector_store import create_vector_store


def extract_tables_to_text(pdf_path: str, page_num: int = None) -> str:
    """
    Extract tables with better formatting and structure preservation.
    """
    if not HAS_CAMELOT:
        print("Camelot not installed; skipping table extraction.")
        return ""

    try:
        page_spec = str(page_num) if page_num else "all"
        tables = camelot.read_pdf(pdf_path, pages=page_spec, flavor="lattice")
        if tables.n == 0:
            tables = camelot.read_pdf(pdf_path, pages=page_spec, flavor="stream")

        extracted_tables = []
        for idx, table in enumerate(tables):
            df = table.df
            table_text = f"TABLE {idx + 1}:\n"
            table_text += df.to_string(index=False)
            extracted_tables.append(table_text)

        return "\n\n".join(extracted_tables)

    except Exception as e:
        print(f"Table extraction error: {e}")
        return ""

def extract_text_from_pdfs():
    """
    Extract text with metadata preservation (page numbers, filenames).
    Returns list of Document objects instead of plain text.
    """
    all_documents = []
    pdf_folder = os.path.join(project_root, "data", "raw_pdfs")
    
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
    
    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")])
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder}")
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    for filename in pdf_files:
        print(f"\nProcessing: {filename}")
        path = os.path.join(pdf_folder, filename)
        
        try:
            with pdfplumber.open(path) as pdf:
                print(f"Total pages: {len(pdf.pages)}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue                  
                    table_text = extract_tables_to_text(path, page_num)
                    
                    full_page_content = text
                    if table_text:
                        full_page_content += f"\n\n{table_text}"                   
                    doc = Document(
                        page_content=full_page_content,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "total_pages": len(pdf.pages)
                        }
                    )
                    all_documents.append(doc)
                    
                    if page_num % 10 == 0:
                        print(f"Processed {page_num} pages...")

            print(f"Completed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\nTotal documents created: {len(all_documents)}")
    return all_documents


def chunk_documents(documents):
    """
    Improved chunking with larger size and better overlap.
    Preserves metadata through the chunking process.
    """
    print("Chunking documents with improved parameters...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  
        chunk_overlap=400,  
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks (avg size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars)")
    
    return chunks


def main():
    """Enhanced ingestion pipeline"""
    print("=" * 60)
    print("NABARD Reports - Enhanced Ingestion Pipeline")
    print("=" * 60)
    
    try:
        documents = extract_text_from_pdfs()
        
        if not documents:
            raise ValueError("No documents extracted from PDFs!")
        chunks = chunk_documents(documents)       
        create_vector_store(chunks)
        
        print("\n" + "=" * 60)
        print("INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("You can now run: streamlit run frontend/app.py")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()