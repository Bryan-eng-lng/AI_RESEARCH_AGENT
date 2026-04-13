from langchain.tools import tool
import arxiv
import os
from typing import List
from dotenv import load_dotenv 
from langchain_community.tools.tavily_search import TavilySearchResults


# --- HELPER: FORMATTER ---
def format_results(results: List[dict]) -> str:
    """Helper to turn raw tool output into a clean Markdown string for Groq."""
    formatted = []
    for res in results:
        content = res.get("content", res.get("summary", "No content available"))
        url = res.get("url", res.get("link", "No link provided"))
        title = res.get("title", "Untitled Source")
        
        formatted.append(f"### Source: {title}\nURL: {url}\nCONTENT: {content}\n")
    return "\n---\n".join(formatted)

# --- 1. WEB SEARCH TOOL ---
@tool 
def web_search(query: str) -> str:
    """Search the web for real-time information. Best for history, news, and general facts."""
    try:     
        # Increase max_results to 5 for better research depth
        search = TavilySearchResults(max_results=5)
        raw_results = search.invoke(query)
        
        # Pro-Tip: Convert the raw list into a clean string so the Analyst doesn't get confused
        return format_results(raw_results)
    
    except Exception as e:
        return f"Error during web search: {str(e)}"

# --- 2. ARXIV TOOL ---
@tool
def search_arxiv(query: str) -> str:
    """Search research papers on Arxiv. Best for Technical, AI, and Math topics."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "summary": paper.summary[:800], # Slightly longer summary for Groq's 70B model
                "link": paper.entry_id
            })

        return format_results(results) if results else "No papers found for this query."
    except Exception as e:
        return f"Error searching Arxiv: {str(e)}"

# --- 3. KNOWLEDGE BASE (Local RAG) ---
@tool
def search_knowledge_base(query: str) -> str:
    """Search local vector store for previously saved research notes and PDFs."""
    try:
        # Import inside the tool to prevent circular import issues
        from rag import vector_store
        
        # Use similarity search with scores for higher precision
        docs = vector_store.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant local documents found."
            
        formatted_docs = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Local File")
            formatted_docs.append(f"### Local Source: {source_name}\nCONTENT: {doc.page_content}")
            
        return "\n---\n".join(formatted_docs)
    except Exception as e:
        return f"Local Knowledge Base Error: {str(e)}"



