# NABARD Rural Credit & MSME Scheme Chatbot

A RAG (Retrieval-Augmented Generation)chatbot that answers questions about NABARD Annual Reports using AI-powered document analysis.

Overview

This project processes NABARD Annual Reports (2021-22 and 2022-23) and enables users to query them using natural language. The chatbot provides accurate, cited answers about rural credit schemes, MSME financing, financial data, and development initiatives.


#Prerequisites
- Python 3.11+
- Groq API Key (FREE - get it at https://console.groq.com/)

Installation

1. **Clone/Download the repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4.Process PDF documents (First time only):
   ```bash
   python backend/ingest.py
   ```

5. Run the application:
   ```bash
   streamlit run frontend/app.py
   ```

The app will open in your browser at `http://localhost:8501`

#Problem Statement

NABARD publishes extensive annual reports containing critical information about rural credit, MSME schemes, and financial operations. These documents are hundreds of pages long, making it difficult to find specific information quickly. This chatbot solves that problem by:

- Processing multiple annual reports automatically
- Enabling natural language queries
- Providing accurate answers with source citations
- Making financial data and scheme information easily accessible

#Technologies Used

- **Frontend:** Streamlit
- **Backend:** LangChain, Python
- **LLM:** Groq API (Llama 3.3 70B) - FREE
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Vector Database:** FAISS
- **PDF Processing:** pdfplumber

#Project Structure

```
Assignment/
├── backend/                    # Backend processing modules
│   ├── ingest.py              # PDF ingestion pipeline
│   ├── rag_pipeline.py        # RAG chain implementation
│   └── vector_store.py        # FAISS vector database
├── frontend/                   # User interface
│   └── app.py                 # Streamlit web app
├── Data/                       # Data storage
│   ├── raw_pdfs/              # Input PDF files
│   └── processed/             # Processed vector store
└── requirements.txt           # Dependencies
```

#Example Queries

- "What is NABARD's total sources of funds in 2022-23?"
- "How much refinance was provided to RRBs?"
- "What are the key MSME schemes mentioned?"
- "What is the total income from financial services?"
- "Summarize NABARD's role in rural development"

