import sys
import os
import re

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
    max_tokens=2048,  
)

vector_db = load_vector_store()

retriever = vector_db.as_retriever(
    search_type="mmr",  
    search_kwargs={
        "k": 20,  
        "fetch_k": 40, 
        "lambda_mult": 0.6  
    }
)

ACRONYM_MAP = {
    "RRB": "Regional Rural Bank",
    "MSME": "Micro, Small and Medium Enterprises",
    "SHG": "Self Help Group",
    "NABARD": "National Bank for Agriculture and Rural Development",
    "PMJJBY": "Pradhan Mantri Jeevan Jyoti Bima Yojana",
    "PMSBY": "Pradhan Mantri Suraksha Bima Yojana",
    "APY": "Atal Pension Yojana",
    "FPO": "Farmer Producer Organization",
    "WDRA": "Warehousing Development and Regulatory Authority",
    "RIDF": "Rural Infrastructure Development Fund",
    "LTIF": "Long Term Irrigation Fund",
    "DIDF": "Dairy Processing Infrastructure Development Fund",
    "KCC": "Kisan Credit Card",
}


def preprocess_query(query: str) -> str:
    """
    Enhanced query preprocessing with year detection and fund name expansion.
    """
    enhanced_query = query
    
    for acronym, expansion in ACRONYM_MAP.items():
        if re.search(rf'\b{acronym}\b', query, re.IGNORECASE):
            enhanced_query += f" {expansion}"
    
    year_pattern = r'20\d{2}-\d{2}'
    years_found = re.findall(year_pattern, query)
    if years_found:
        for year in years_found:
            enhanced_query += f" FY{year} {year}"
    
    if re.search(r'\b(sanctioned|disbursed|funds|allocated)\b', query, re.IGNORECASE):
        enhanced_query += " sanctioned disbursed allocated amount"
    
    if re.search(r'\b(KCC|Kisan Credit Card)\b', query, re.IGNORECASE):
        enhanced_query += " saturation drive Phase operative issued"
    
    return enhanced_query


def extract_year_from_metadata(metadata):
    """Extract year from source filename."""
    source = metadata.get('source', '')
    year_match = re.search(r'20(\d{2})[_-](\d{2})', source)
    if year_match:
        return f"20{year_match.group(1)}-{year_match.group(2)}"
    return None


def detect_query_years(query: str):
    """
    Detect which financial years are requested in the query.
    """
    years = []
    
    year_pattern = r'20(\d{2})-(\d{2})'
    matches = re.findall(year_pattern, query)
    
    for match in matches:
        year = f"20{match[0]}-{match[1]}"
        years.append(year)
    
    if re.search(r'\b(between|from|to|and)\b', query, re.IGNORECASE) and len(years) == 2:
        start_year = int(years[0][:4])
        end_year = int(years[-1][:4])
        years = []
        for y in range(start_year, end_year + 1):
            next_y = str(y + 1)[-2:]
            years.append(f"{y}-{next_y}")
    
    return years


system_prompt = """You are an expert assistant specializing in NABARD (National Bank for Agriculture and Rural Development) Annual Reports.

CRITICAL INSTRUCTIONS FOR YEAR-SPECIFIC QUERIES:
1. **When asked about multiple years** (e.g., 2020-21, 2021-22, 2022-23):
   - You MUST provide data for EACH year separately
   - Do NOT provide only cumulative totals
   - Search through ALL context chunks for each specific year
   - If data for a year is missing, explicitly state which year is missing

2. **Distinguish between data types:**
   - "Sanctioned" amounts vs "Disbursed" amounts are DIFFERENT
   - "Loans outstanding" vs "Sanctioned/Disbursed" are DIFFERENT
   - "Cumulative" vs "Year-specific" are DIFFERENT
   - Always specify which type you're reporting

3. **For fund-related queries (LTIF, DIDF):**
   - Look for year-wise sanctioned AND disbursed amounts
   - These are typically in tables or financial statements
   - Format: "In FY20XX-XX, ₹X crore was sanctioned, and ₹Y crore was disbursed"

4. **For KCC queries:**
   - Look for Phase I and Phase II data separately
   - Report both number of KCCs and credit limits sanctioned
   - Check data across ALL requested years

5. **Use ONLY the provided context** - Do not make assumptions
6. **Be precise with numbers** - Always cite exact figures with units (₹ crore, ₹ lakh, lakh KCCs)
7. **Check ALL sources carefully** - Information may be split across multiple chunks

Context from NABARD Reports (with metadata):
{context}

User Question: {question}

Answer format:
- Provide year-by-year breakdown when multiple years are requested
- Use clear section headers for each year if helpful
- Cite specific source and page numbers
- If any year's data is missing, explicitly state it

Answer:"""

prompt = ChatPromptTemplate.from_template(system_prompt)


def format_docs(docs):
    """
    Enhanced formatting with year information highlighted.
    """
    print(f"Retrieved {len(docs)} documents")
    
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        year = extract_year_from_metadata(metadata)
        
        year_info = f" [FY{year}]" if year else ""
        chunk_header = f"--- Source {i}: {source}{year_info} (Page {page}) ---"
        formatted_chunks.append(f"{chunk_header}\n{doc.page_content}\n")
    
    return "\n\n".join(formatted_chunks)


def rerank_docs(docs, query):
    """
    Enhanced reranking that prioritizes:
    1. Keyword relevance
    2. Year coverage (for multi-year queries)
    3. Data type relevance (sanctioned/disbursed vs loans)
    """
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    requested_years = detect_query_years(query)
    
    if 'sanctioned' in query_lower or 'disbursed' in query_lower:
        boost_terms = {'sanctioned', 'disbursed', 'allocated'}
    else:
        boost_terms = set()
    
    scored_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        score = 0
        
        score += sum(3 if term in boost_terms else 1 
                    for term in query_terms if term in content_lower)
        
        doc_year = extract_year_from_metadata(doc.metadata)
        if doc_year and doc_year in requested_years:
            score += 10 
        
        if requested_years and 'cumulative' in content_lower:
            score -= 5
        
        scored_docs.append((score, doc))
    
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    if requested_years:
        final_docs = []
        years_covered = set()
        
        for score, doc in scored_docs:
            doc_year = extract_year_from_metadata(doc.metadata)
            if doc_year in requested_years and doc_year not in years_covered:
                final_docs.append(doc)
                years_covered.add(doc_year)
                if len(years_covered) == len(requested_years):
                    break
        
        for score, doc in scored_docs:
            if doc not in final_docs and len(final_docs) < 15:
                final_docs.append(doc)
        
        return final_docs[:15]
    
    return [doc for _, doc in scored_docs[:15]]


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


def get_response(query: str):
    """
    Enhanced response generation with multi-year query handling.
    """
    print(f"Processing query: {query}")
    
    try:
        requested_years = detect_query_years(query)
        if requested_years:
            print(f"Multi-year query detected: {requested_years}")
        
        enhanced_query = preprocess_query(query)
        print(f"Enhanced query: {enhanced_query}")        
        
        docs = retriever.invoke(enhanced_query)
        print(f"Retrieved {len(docs)} initial documents")
        
        years_in_docs = set()
        for doc in docs:
            doc_year = extract_year_from_metadata(doc.metadata)
            if doc_year:
                years_in_docs.add(doc_year)
        print(f"Years covered in retrieved docs: {years_in_docs}")
        
        reranked_docs = rerank_docs(docs, query)
        print(f"Reranked to {len(reranked_docs)} documents")
        
        formatted_context = format_docs(reranked_docs)
        answer = llm.invoke(
            prompt.format(context=formatted_context, question=query)
        )
        
        sources = []
        for doc in reranked_docs[:5]:  
            metadata = doc.metadata
            doc_year = extract_year_from_metadata(metadata)
            year_info = f" [FY{doc_year}]" if doc_year else ""
            source_info = f"[{metadata.get('source', 'Unknown')}{year_info} - Page {metadata.get('page', 'N/A')}]\n"
            source_info += doc.page_content[:400] + "..."
            sources.append(source_info)
        
        return answer.content if hasattr(answer, 'content') else str(answer), sources
        
    except Exception as e:
        print(f"Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {str(e)}", []


if __name__ == "__main__":
    print("Testing Enhanced RAG Pipeline...")
    
    test_queries = [
        "How much funds were sanctioned and disbursed to LTIF and DIDF in year 2020-21, 2021-22, and 2022-23?",
        "How did the Kisan Credit Card (KCC) saturation drive progress between FY2020-21 and FY2022-23?",
    ]
    
    for test_query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {test_query}")
        print('='*60)
        answer, sources = get_response(test_query)
        print(f"\nAnswer: {answer}")
        print(f"\nSources: {len(sources)} documents")