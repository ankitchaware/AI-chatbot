"""
RAG Pipeline using Groq (FREE) and LangChain
Handles retrieval and generation for NABARD chatbot
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from backend.vector_store import load_vector_store
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.1,
    max_tokens=1024,
)


vector_db = load_vector_store()
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}  
)

system_prompt = """You are an expert assistant specializing in NABARD Annual Reports (2021-22 and 2022-23 combined as a single dataset).

Your responsibilities:
1. Answer questions ONLY using the provided context from NABARD reports
2. Cite exact figures with proper units (₹ crore, ₹ lakh, etc.)
3. Be precise and factual - no speculation
4. If information is not available in the context, clearly state: "Information not available in the provided NABARD reports."
5. Format financial data clearly with proper notation
6. Accept natural language questions from the user and answer them clearly and concisely.

Context from NABARD Reports:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    print(f"Retrieved {len(docs)} documents")
    """Format retrieved documents into a single context string"""
    return "\n\n".join([doc.page_content for doc in docs])


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def get_response(query):
    print(f"Processing query: {query}")
    """
    Get response from RAG pipeline
    
    Args:
        query: User question about NABARD reports
    
    Returns:
        tuple: (answer, list of source excerpts)
    """
    try:
        docs = retriever.invoke(query)
        answer = rag_chain.invoke(query)
        sources = [doc.page_content[:350] + "..." for doc in docs]  
        return answer, sources
        
    except Exception as e:
        print(f"Error generating response: {e}")
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, []

if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    test_query = "What is NABARD's total sources of funds in 2022-23?"
    answer, sources = get_response(test_query)
    print(f"\nQuery: {test_query}")
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {len(sources)} documents retrieved")













