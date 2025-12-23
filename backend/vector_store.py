from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def create_vector_store(chunks):
    """
    Create FAISS vector store from Document objects.
    Handles both Document objects and text strings.
    """
    print(f"Creating vector store with {len(chunks)} chunks...")
    
    if chunks and isinstance(chunks[0], Document):
        db = FAISS.from_documents(chunks, embeddings)
    else:
        db = FAISS.from_texts(chunks, embeddings)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, "data", "processed", "chunks_embeddings")
    os.makedirs(save_path, exist_ok=True)
    
    db.save_local(save_path)
    print(f"Vector store saved to {save_path}")
    
    return db


def load_vector_store():
    """
    Load existing FAISS vector store with enhanced retrieval settings.
    """
    print("Loading vector store...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    folder = os.path.join(project_root, "data", "processed", "chunks_embeddings")
    
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Vector store not found at {folder}!\n"
            "Please run: python backend/ingest.py"
        )
    
    db = FAISS.load_local(
        folder, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print(f"Vector store loaded successfully")
    print(f"Index contains {db.index.ntotal} vectors")
    
    return db