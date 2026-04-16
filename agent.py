import os
import sqlite3
import requests
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from rag import query_memory, save_to_memory



load_dotenv(dotenv_path=".env")

# ── TOOLS ─────────────────────────────────────────────────────────────────────
web_search = TavilySearch(
    max_results=6,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    include_raw_content=True,
)

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)

# ── PERSISTENCE ───────────────────────────────────────────────────────────────
conn = sqlite3.connect("research_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class ResearchPlan(BaseModel):
    queries: List[str] = Field(description="10-15 high-precision search queries covering different angles.")
    reasoning: str = Field(description="Why these queries were chosen and what gaps they cover.")

class CitedFact(BaseModel):
    """A hard fact with its source ID. Used as sidebar data points."""
    fact: str = Field(description="One hard metric, date, or specification. e.g. 'H100 delivers 3.35 TB/s memory bandwidth'")
    source_id: int = Field(description="The [N] index number from the SOURCE INDEX. Must be a real index number.")

class ResearchChapter(BaseModel):
    title: str = Field(description="Clear, specific chapter title in Title Case. e.g. 'Memory Architecture and Bandwidth Evolution'")
    intro: str = Field(description="2-3 sentence paragraph that sets context and states what this chapter investigates.")
    narrative: str = Field(
        description=(
            "The main investigative body. Write as flowing prose like a journal article or long-form report. "
            "Minimum 200 words. Build an argument - show cause and effect, tensions, decisions, consequences. "
            "Cite sources inline using [N] notation e.g. 'The H100 introduced HBM3 in 2022 [3], doubling bandwidth over the A100 [7].' "
            "Only cite source IDs that exist in the SOURCE INDEX."
        )
    )
    key_facts: List[CitedFact] = Field(description="3-5 hard data points pulled from this chapter as a sidebar. Each must cite a real source_id.")
    takeaway: str = Field(description="One sharp sentence - the single most important insight from this chapter.")

class FinalDossier(BaseModel):
    title: str = Field(description="Specific, descriptive report title.")
    key_findings: List[str] = Field(description="5 most important findings from the entire report. Each should be a complete sentence with a hard fact.")
    executive_summary: str = Field(
        description=(
            "Half-page executive summary (150-200 words). Written for a senior decision-maker. "
            "Cover: what was investigated, the 3 most important findings, and the strategic implication. "
            "Must contain specific metrics and dates, not vague statements."
        )
    )
    chapters: List[ResearchChapter]
    synthesis: str = Field(
        description=(
            "Final synthesis paragraph (100-150 words). Reveal a non-obvious connection across chapters. "
            "What does the data collectively suggest that no single chapter states explicitly?"
        )
    )

class QualityVerdict(BaseModel):
    score: int = Field(description="Quality score 1-10. 8+ means publish-ready.")
    gaps: List[str] = Field(default_factory=list, description="Specific missing data points. Empty list if none.")
    follow_up_queries: List[str] = Field(default_factory=list, description="New queries to fill gaps. Empty list if score >= 8.")
    verdict: str = Field(description="APPROVED or NEEDS_MORE_RESEARCH")

    @field_validator("gaps", "follow_up_queries", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []

class AgentState(Dict):
    topic: str
    plan: ResearchPlan
    raw_data: str
    raw_report: str
    source_index: Dict[int, str]
    source_titles: Dict[int, str]
    memory_context: str
    quality: QualityVerdict
    iteration: int
    research_rounds: int

# ── SYSTEM PROMPTS ─────────────────────────────────────────────────────────────
STRATEGIST_PROMPT = """You are an elite intelligence analyst. Build a comprehensive research plan.
Generate 10-15 search queries covering:
- Technical specifications and hard metrics
- Historical timeline with specific dates and events
- Key figures, engineers, decision-makers
- Comparative analysis and bottlenecks
- Failures, weaknesses, and overlooked angles
Every query must target a specific data point. No generic queries."""

ARCHITECT_PROMPT = """You are a world-class investigative analyst writing a long-form research report.

CITATION RULES:
1. A SOURCE INDEX will be provided: [1] https://... - "Title"
2. Cite retrieved facts inline using [N] notation in narratives.
3. Every key_fact MUST have a source_id from the index.
4. NEVER invent source IDs. NEVER cite a number not in the index.
5. NEVER fabricate quotes or statistics and attribute them to a source.

WRITING RULES - CRITICAL:
1. You are an expert. USE YOUR OWN DEEP KNOWLEDGE to write rich, detailed narratives.
2. Retrieved data gives you grounded facts to cite. Your knowledge fills the depth.
3. Each chapter narrative MUST be at least 300 words. No exceptions.
4. Write like a senior engineer explaining to another engineer - specific, technical, opinionated.
5. Show tradeoffs, engineering decisions, real-world implications.
6. MINIMUM 5 chapters. Each on its own distinct angle.
7. Executive summary MUST be at least 150 words covering key findings and strategic implications.
8. key_findings must be 5 sentences each with a hard number or specific fact."""

CRITIC_PROMPT = """You are a ruthless research quality auditor for a top-tier journal.
Score the report 1-10:
- 1-4: Generic, missing fundamental facts, reads like a summary
- 5-7: Decent but narratives are thin, missing key metrics or angles
- 8-10: Expert-level, specific, narrative-driven, cross-referenced, surprising insights
Be harsh. Only approve (score >= 8) if a senior researcher would find it valuable."""

# ── NODES ──────────────────────────────────────────────────────────────────────
def strategist_node(state: AgentState):
    print("\n[STRATEGIST] Building search vectors...")
    memory_context = query_memory(state["topic"])
    print(f"  Memory: {'found past research' if memory_context else 'starting fresh'}")

    structured_llm = llm.with_structured_output(ResearchPlan)
    memory_hint = f"\n\nPAST RESEARCH (avoid re-searching these):\n{memory_context}" if memory_context else ""
    time.sleep(5)
    plan = structured_llm.invoke([
        SystemMessage(content=STRATEGIST_PROMPT),
        HumanMessage(content=f"Build research plan for: {state['topic']}{memory_hint}")
    ])
    print(f"  {len(plan.queries)} queries planned")
    return {
        "plan": plan,
        "iteration": 1,
        "research_rounds": 0,
        "source_index": {},
        "source_titles": {},
        "memory_context": memory_context or "",
    }

def hitl_node(state: AgentState):
    print(f"\n[COMMANDER REVIEW] {len(state['plan'].queries)} vectors ready")
    for i, q in enumerate(state['plan'].queries, 1):
        print(f"  {i:02d}. {q}")
    return state

def deep_fetch(url: str, max_chars: int = 5000) -> str:
    """Fetch full page content from a URL. Falls back gracefully."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return ""
        text = resp.text
        # Strip HTML tags simply
        import re
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""

def crawler_node(state: AgentState):
    queries = state['plan'].queries
    rounds = state.get("research_rounds", 0) + 1
    print(f"\n[CRAWLER] Round {rounds} - {len(queries)} queries...")

    source_index: Dict[int, str] = {}
    source_titles: Dict[int, str] = {}
    for k, v in (state.get("source_index") or {}).items():
        source_index[int(k)] = v
    for k, v in (state.get("source_titles") or {}).items():
        source_titles[int(k)] = v

    seen_urls = set(source_index.values())
    # Store (sid, url, snippet) for deep fetch candidates
    new_sources = []
    raw_chunks = []

    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q[:65]}...")
        try:
            response = web_search.invoke(q)
            items = response.get("results", []) if isinstance(response, dict) else response
            for r in items:
                url = (r.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                sid = len(source_index) + 1
                source_index[sid] = url
                title = (r.get("title") or url)[:80]
                source_titles[sid] = title
                content = (r.get("raw_content") or r.get("content") or "")[:600]
                new_sources.append((sid, url, title, content))
        except Exception as e:
            print(f"  Query failed: {e}")

    # Deep fetch top 12 most relevant sources for full content
    print(f"  Deep fetching top sources for full content...")
    deep_fetch_count = 0
    for sid, url, title, snippet in new_sources:
        # Skip PDFs, social media, video sites
        skip = any(x in url for x in ["youtube.com", "twitter.com", "linkedin.com", ".pdf", "reddit.com"])
        if not skip and deep_fetch_count < 5:
            full_content = deep_fetch(url, max_chars=2000)
            if len(full_content) > len(snippet):
                content = full_content
                deep_fetch_count += 1
            else:
                content = snippet
        else:
            content = snippet
        raw_chunks.append(f"[{sid}] SOURCE: {url}\nTITLE: {title}\n{content}")

    print(f"  {len(source_index)} sources indexed, {deep_fetch_count} deep fetched")
    existing_raw = state.get("raw_data", "")
    combined = existing_raw + f"\n\n=== ROUND {rounds} ===\n\n" + "\n---\n".join(raw_chunks)
    return {
        "raw_data": combined,
        "source_index": source_index,
        "source_titles": source_titles,
        "research_rounds": rounds,
    }

def architect_node(state: AgentState):
    print("\n[ARCHITECT] Writing research report...")

    top_index = dict(sorted(state["source_index"].items())[:30])
    top_titles = {k: state["source_titles"].get(k, "") for k in top_index}
    index_str = "\n".join(
        f"[{sid}] {url} - \"{top_titles.get(sid, '')}\""
        for sid, url in top_index.items()
    )

    raw = state["raw_data"]
    if len(raw) > 12000:
        raw = raw[:12000] + "\n[truncated]"

    memory_section = (
        f"\n\nPAST RESEARCH CONTEXT:\n{state['memory_context'][:2000]}"
        if state.get("memory_context") else ""
    )

    # STEP 1: Free-form writing - no schema = no length constraints
    write_prompt = f"""You are a world-class investigative journalist writing a 3000-word deep-dive research article for a technical publication like MIT Technology Review or IEEE Spectrum.

TOPIC: {state['topic']}

SOURCES (cite as [N] inline when using specific facts):
{index_str}

RAW DATA FROM SOURCES:
{raw}{memory_section}

Write a complete, deeply researched article. Requirements:
- Minimum 3000 words total
- Write in flowing paragraphs, not bullet points
- Each section must be at least 400 words
- Use your expert knowledge to explain WHY and HOW, not just WHAT
- Cite specific facts from sources using [N] notation
- Include specific numbers, dates, technical details
- Show tensions, tradeoffs, engineering decisions and their consequences
- Write like you are explaining to a senior engineer who wants depth

Structure your article with these clearly labeled sections:
## TITLE
## KEY_FINDINGS (5 numbered findings, each with a hard fact)
## EXECUTIVE_SUMMARY (200+ words)
## SECTION: [name]
[400+ words of deep analysis]
## SECTION: [name]
[400+ words]
[continue for 5-6 sections total]
## SYNTHESIS
[150+ words of non-obvious cross-section insight]

Begin writing now. Do not truncate. Write the complete article."""

    print("  Step 1: Free-form writing...")
    raw_report = llm.invoke([
        SystemMessage(content="You are an expert research analyst. Write detailed, long-form content. Never truncate. Always complete every section fully."),
        HumanMessage(content=write_prompt)
    ]).content
    print(f"  Raw report: {len(raw_report)} chars written")

    with open("raw_report_debug.txt", "w", encoding="utf-8") as f:
        f.write(raw_report)

    return {"raw_report": raw_report}

def factcheck_node(state: AgentState):
    print("\n[FACTCHECK] Verifying citations...")
    import re
    valid_ids = set(state["source_index"].keys())
    raw = state.get("raw_report", "")
    cited = set(int(x) for x in re.findall(r'\[(\d+)\]', raw))
    valid_cited = cited & valid_ids
    invalid_cited = cited - valid_ids
    print(f"  {len(valid_cited)} valid citations, {len(invalid_cited)} unverified")
    return {}

def critic_node(state: AgentState):
    iteration = state.get("iteration", 1)
    print(f"\n[CRITIC] Quality audit - iteration {iteration}...")
    time.sleep(15)
    structured_llm = llm.with_structured_output(QualityVerdict)
    raw = state.get("raw_report", "")[:3000]
    verdict = structured_llm.invoke([
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=f"TOPIC: {state['topic']}\n\nREPORT EXCERPT:\n{raw}")
    ])
    print(f"  Score: {verdict.score}/10 | {verdict.verdict}")
    if verdict.gaps:
        print(f"  Gaps: {', '.join(verdict.gaps[:2])}")
    return {"quality": verdict, "iteration": iteration + 1}

def refine_node(state: AgentState):
    verdict = state["quality"]
    print(f"\n[REFINE] {len(verdict.follow_up_queries)} follow-up queries added")
    return {
        "plan": ResearchPlan(
            queries=verdict.follow_up_queries,
            reasoning=f"Filling gaps: {', '.join(verdict.gaps)}"
        )
    }

def should_refine(state: AgentState) -> str:
    verdict = state.get("quality")
    iteration = state.get("iteration", 1)
    if verdict and verdict.verdict == "NEEDS_MORE_RESEARCH" and iteration <= 4:
        return "refine"
    return "export"

# ── PDF HELPERS ────────────────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00b7": "*",
        "\u2022": "-", "\u00ae": "(R)", "\u2122": "(TM)", "\u00a0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def draw_line(pdf, w, r=180, g=180, b=180):
    pdf.set_draw_color(r, g, b)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + w, pdf.get_y())
    pdf.ln(3)

# ── PDF EXPORT ─────────────────────────────────────────────────────────────────
def export_to_pdf(raw_report: str, source_index: Dict, source_titles: Dict, topic: str):
    import re
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(18, 18, 18)
    w = 210 - 36

    # Parse title from raw report
    title_match = re.search(r'##\s*TITLE\s*\n(.+)', raw_report)
    title = title_match.group(1).strip() if title_match else topic

    # Parse sections
    sections = re.split(r'\n##\s+', raw_report)

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(w, 11, sanitize(title.upper()), align="C")
    pdf.ln(4)
    draw_line(pdf, w, 20, 20, 20)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    date_str = datetime.now().strftime("%B %d, %Y")
    pdf.cell(w, 6, f"Research Report  |  Generated {date_str}  |  {len(source_index)} verified sources",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    # Key findings on cover
    findings_match = re.search(r'KEY_FINDINGS[:\s]*\n(.*?)(?=\n##|\nEXECUTIVE|\Z)', raw_report, re.DOTALL)
    if findings_match:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(w, 8, "KEY FINDINGS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        draw_line(pdf, w, 180, 180, 180)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        for line in findings_match.group(1).strip().split("\n"):
            if line.strip():
                pdf.multi_cell(w, 6, sanitize(line.strip()))
                pdf.ln(1)

    pdf.ln(8)

    # TOC
    draw_line(pdf, w)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(w, 8, "TABLE OF CONTENTS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    draw_line(pdf, w)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    section_titles = re.findall(r'\n##\s+SECTION:\s*(.+)', raw_report)
    toc = ["Executive Summary"] + section_titles + ["Synthesis", "Verified Sources"]
    for i, item in enumerate(toc, 1):
        pdf.cell(w, 6, sanitize(f"  {i}.  {item}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 287, 210, 10, "F")

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
    exec_match = re.search(r'EXECUTIVE_SUMMARY[:\s]*\n(.*?)(?=\n##\s+SECTION|\Z)', raw_report, re.DOTALL)
    if exec_match:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(w, 10, "EXECUTIVE SUMMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        draw_line(pdf, w, 20, 20, 20)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(w, 7, sanitize(exec_match.group(1).strip()))

    # ── SECTIONS ──────────────────────────────────────────────────────────────
    section_blocks = re.findall(r'\n##\s+SECTION:\s*(.+?)\n(.*?)(?=\n##\s+SECTION:|\n##\s+SYNTHESIS|\Z)',
                                 raw_report, re.DOTALL)
    for idx, (sec_title, sec_body) in enumerate(section_blocks, 1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(w, 7, f"Section {idx}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(w, 10, sanitize(sec_title.strip()))
        draw_line(pdf, w, 20, 20, 20)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(w, 6, sanitize(sec_body.strip()))

    # ── SYNTHESIS ─────────────────────────────────────────────────────────────
    synth_match = re.search(r'##\s+SYNTHESIS\s*\n(.*?)(?=\n##|\Z)', raw_report, re.DOTALL)
    if synth_match:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(w, 10, "FINAL SYNTHESIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        draw_line(pdf, w, 20, 20, 20)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(w, 7, sanitize(synth_match.group(1).strip()))

    # ── VERIFIED SOURCES ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(w, 10, "VERIFIED SOURCES", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    draw_line(pdf, w, 20, 20, 20)
    pdf.ln(4)
    for sid in sorted(source_index.keys()):
        url = source_index[sid]
        title_s = source_titles.get(sid, url)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(w, 5, sanitize(f"[{sid}]  {title_s}"))
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(0, 60, 160)
        for chunk in [url[i:i+90] for i in range(0, len(url), 90)]:
            pdf.cell(w, 4, sanitize(f"      {chunk}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 287, 210, 10, "F")

    filename = f"REPORT_{title[:35].replace(' ', '_').upper()}.pdf"
    filename = "".join(c for c in filename if c not in r'\/:*?"<>|')
    pdf.output(filename)
    print(f"\n[EXPORTED] {filename} | {len(section_blocks)} sections | {len(source_index)} sources")
    return filename

# ── GRAPH ──────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("strategist",       strategist_node)
builder.add_node("commander_review", hitl_node)
builder.add_node("crawler",          crawler_node)
builder.add_node("architect",        architect_node)
builder.add_node("factcheck",        factcheck_node)
builder.add_node("critic",           critic_node)
builder.add_node("refine",           refine_node)

builder.set_entry_point("strategist")
builder.add_edge("strategist",       "commander_review")
builder.add_edge("commander_review", "crawler")
builder.add_edge("crawler",          "architect")
builder.add_edge("architect",        "factcheck")
builder.add_edge("factcheck",        "critic")
builder.add_conditional_edges("critic", should_refine, {"refine": "refine", "export": END})
builder.add_edge("refine",           "crawler")

app = builder.compile(checkpointer=memory, interrupt_before=["commander_review"])

# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "NVIDIA H100 vs A100 architecture"
    thread_id = topic[:30].replace(" ", "_")
    thread_config = {"configurable": {"thread_id": thread_id}}

    existing_state = app.get_state(thread_config)
    if not existing_state.values:
        print(f"\n[MISSION] {topic}")
        print("=" * 60)
        for event in app.stream({"topic": topic}, thread_config):
            pass

    current = app.get_state(thread_config)
    plan = current.values.get("plan")
    print("\n" + "=" * 60)
    print("MISSION PAUSED - COMMANDER REVIEW")
    print("=" * 60)
    for i, q in enumerate(plan.queries, 1):
        print(f"  {i:02d}. {q}")
    print(f"\nReasoning: {plan.reasoning}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == "y":
        print("\n[RESUMING]")
        final_result = app.invoke(None, thread_config)
        export_to_pdf(
            final_result.get("raw_report", ""),
            final_result["source_index"],
            final_result["source_titles"],
            topic,
        )

        print("\n[MEMORY] Saving to long-term memory...")
        raw = final_result.get("raw_report", "")
        save_to_memory(topic, raw[:3000], list(final_result["source_index"].values()))
        print("[DONE] Research saved.")
    else:
        print("Mission aborted.")


