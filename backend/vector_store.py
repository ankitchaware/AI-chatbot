"""
Vector Store Management for NABARD Reports
Creates and loads FAISS vector database with embeddings
"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks):
    print("Creating vector store...")
    """
    Create FAISS vector store from text chunks
    
    Args:
        chunks: List of text chunks from PDF extraction
    
    Returns:
        FAISS vector database
    """
    print(f"Creating vector store with {len(chunks)} chunks...")
    db = FAISS.from_texts(chunks, embeddings)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, "data", "processed", "chunks_embeddings")
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)
    
    print(f"✓ Vector store saved to {save_path}")
    return db

def load_vector_store():
    print("Loading vector store...")
    """
    Load existing FAISS vector store from disk
    Returns:
        FAISS vector database
    Raises:
        FileNotFoundError: If vector store doesn't exist
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    folder = os.path.join(project_root, "data", "processed", "chunks_embeddings")
    
    if not os.path.exists(folder):
        print(f"Vector store not found at {folder}")
        raise FileNotFoundError(
            f"Vector store not found at {folder}!\n"
            "Please run: python backend/ingest.py"
        )
    
    print(f"Loading vector store from {folder}...")
    db = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    print("✓ Vector store loaded successfully")
    
    return db












