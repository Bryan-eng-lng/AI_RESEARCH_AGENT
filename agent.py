import operator
from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage , SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from tools import search_arxiv, search_knowledge_base, web_search
import os 
from langchain_groq import ChatGroq
from dotenv import load_dotenv 

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")


llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_KEY
)

print("Agent is now initialized with Groq!")


fast_llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-8b-instant",
    api_key = GROQ_KEY
)



class ResearchChapter(BaseModel):
    chapter_title: str = Field(description="Bold, ALL-CAPS chapter title (e.g., 'CHAPTER 1: THE SILENT LOGISTICS')")
    content: str = Field(description="Substantial technical analysis. Depth should match the complexity of the topic.")
    key_takeaway: str = Field(description="One-sentence strategic bottom line.")

class ResearchPlan(BaseModel):
    queries: List[str] = Field(description="4 distinct queries covering depth, context, and contradictions.")
    strategy: str = Field(description="Brief explanation of the search strategy.")

class ResearchSummary(BaseModel):
    title: str = Field(description="The 'Mission' or Dossier Title")
    briefing: str = Field(description="A high-level overview (approx. 200 words) setting the stage.")
    chapters: List[ResearchChapter] = Field(description="3-4 mission-critical chapters providing deep-dive research.")
    visual_diagram_code: str = Field(description="Advanced Mermaid.js code with subgraphs grouping the chapters.")
    sources: List[str] = Field(description="Verified URLs used")
    confidence: float = Field(description="Accuracy score 0.0-1.0")
    final_synthesis: str = Field(description="The 'End Credits' - Future impact and conclusion.")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer 'yes' if grounded in docs, 'no' if hallucinated")

class AgentState(TypedDict):
    query: str
    queries: List[str]
    research_raw: str
    summary: ResearchSummary
    hallucination_grade: str
    iteration_count: int

# --- 2. THE 5 STRATEGIC NODES ---

def strategist_node(state: AgentState):
    print("--- STEP 1: STRATEGIC PLANNING ---")
    structured_llm = llm.with_structured_output(ResearchPlan)
    system_prompt = (
        "You are an Elite Research Strategist. Break this topic into 4 specific dimensions: "
        "1. Core Facts, 2. Technical/Logistical Details, 3. Historical/Contextual Evolution, "
        "and 4. Known Contradictions. Dominate the depth."
    )
    plan = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Create a dominance plan for: {state['query']}")
    ])
    return {"queries": plan.queries, "iteration_count": state.get("iteration_count", 0) + 1}

def deep_crawler_node(state: AgentState):
    print(f"--- STEP 2: DEEP CRAWLING ({len(state['queries'])} queries) ---")
    all_context = []
    for q in state['queries']:
        print(f"  > Executing specialized search: {q}")
        search_results = web_search.invoke(q)
        all_context.append(f"QUERY: {q}\nRESULTS: {search_results}")
    return {"research_raw": "\n\n---\n\n".join(all_context)}

def architect_node(state: AgentState):
    print("--- STEP 3: MISSION DOSSIER SYNTHESIS ---")
    structured_llm = llm.with_structured_output(ResearchSummary)
    
    prompt = (
    f"DATA: {state['research_raw']}\n\n"
    "You are a Lead Systems Engineer performing a Forensic Analysis. "
    "I will fire you if you use generic 'textbook' definitions. "
    "Use the 'Aditya Dhar' Mission Style:\n\n"
    "1. THE 'IDENTIFIER' RULE: Every chapter must mention at least 3 specific technical 'Identifiers' found in the research (e.g., GQA, 128k Vocab, RoPE scaling, 400 TFLOPS, tiktoken, etc.).\n"
    "2. NO GENERALITIES: Do not say 'it is optimized.' Say *how* it is optimized (e.g., 'By using Grouped Query Attention, it reduces KV heads to 8, allowing for a 50GB reduction in VRAM overhead').\n"
    "3. CHAPTER TITLES: Make them aggressive and specific (e.g., 'CHAPTER 1: THE GQA BANDWIDTH BREAKTHROUGH').\n"
    "4. VISUAL: The Mermaid map must be a technical schematic, not a bubble chart."
)
    
    summary = structured_llm.invoke([
        SystemMessage(content="You are a Senior Research Architect. Synthesize data into a professional long-form dossier."),
        HumanMessage(content=prompt)
    ])
    return {"summary": summary}

def hallucination_grader_node(state: AgentState):
    print("--- STEP 4: TRUTH VERIFICATION ---")
    grader_llm = llm.with_structured_output(GradeHallucinations)
    prompt = (
        f"DOCS: {state['research_raw']}\n"
        f"SUMMARY: {state['summary'].model_dump_json()}\n\n"
        "Is this dossier 100% grounded in the provided docs? (yes/no)"
    )
    grade = grader_llm.invoke(prompt)
    return {"hallucination_grade": grade.binary_score}

def director_node(state: AgentState):
    print("--- STEP 5: FINAL REPORTING ---")
    return state

# --- 3. GRAPH CONSTRUCTION ---

def route_after_grader(state: AgentState):
    if state["hallucination_grade"].lower() == "yes" or state["iteration_count"] >= 3:
        return "director"
    print(f"--- QUALITY FAILED (Attempt {state['iteration_count']}/3): RE-PLANNING ---")
    return "strategist"

workflow = StateGraph(AgentState)
workflow.add_node("strategist", strategist_node)
workflow.add_node("crawler", deep_crawler_node)
workflow.add_node("architect", architect_node)
workflow.add_node("grader", hallucination_grader_node)
workflow.add_node("director", director_node)

workflow.set_entry_point("strategist")
workflow.add_edge("strategist", "crawler")
workflow.add_edge("crawler", "architect")
workflow.add_edge("architect", "grader")
workflow.add_conditional_edges("grader", route_after_grader, {"strategist": "strategist", "director": "director"})
workflow.add_edge("director", END)

app = workflow.compile()

# --- 4. EXECUTION & FORMATTING ---
if __name__ == "__main__":
    initial_state = {
        "query": "The architecture and KV-Caching mechanism of Llama 3 70B for high-throughput inference.",
        "iteration_count": 0
    }
    
    result = app.invoke(initial_state)
    summary = result['summary']
    
    print("\n" + "="*60)
    print(f"MISSION: {summary.title.upper()}")
    print("="*60)
    
    print(f"\n[BRIEFING]\n{summary.briefing}")
    
    for chapter in summary.chapters:
        print(f"\n\n{chapter.chapter_title}")
        print("-" * len(chapter.chapter_title))
        print(chapter.content)
        print(f"\n> TAKEAWAY: {chapter.key_takeaway}")
    
    print(f"\n\n{'='*20} STRATEGIC MAP {'='*20}")
    print(summary.visual_diagram_code)
    
    print(f"\n{'='*20} FINAL SYNTHESIS {'='*20}")
    print(summary.final_synthesis)
    print("="*60)












