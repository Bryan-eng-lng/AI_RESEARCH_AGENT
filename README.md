# AI Research Agent

A self-correcting, memory-enabled research agent that autonomously crawls the web, verifies citations, and generates structured research reports as PDFs.

Built with LangGraph, LangChain, and Groq/OpenAI.

---

## Architecture

```
User Input (Topic)
      │
      ▼
┌─────────────┐
│  STRATEGIST │  Plans 10-15 targeted search queries
│    NODE     │  Checks long-term memory for past research
└──────┬──────┘
       │
       ▼  [HUMAN-IN-THE-LOOP PAUSE]
┌─────────────┐
│  COMMANDER  │  User reviews and approves the search plan
│   REVIEW    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   CRAWLER   │  Executes queries via Tavily
│    NODE     │  Deep-fetches full page content (not just snippets)
│             │  Builds numbered source index [1], [2], [3]...
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ARCHITECT  │  Writes long-form research report
│    NODE     │  Two-step: free-form writing → structure parsing
│             │  Cites sources inline using [N] notation
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  FACTCHECK  │  Strips any citation not in the real crawled index
│    NODE     │  Hallucination firewall
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    CRITIC   │  Scores report quality 1-10
│    NODE     │  If score < 8 → generates follow-up queries
└──────┬──────┘
       │
    ┌──┴──┐
    │     │
  PASS  FAIL → back to CRAWLER (max 4 rounds)
    │
    ▼
┌─────────────┐
│  PDF EXPORT │  Cover page, TOC, sections, synthesis, verified sources
└─────────────┘
       │
       ▼
┌─────────────┐
│  RAG MEMORY │  Saves research to ChromaDB for future runs
└─────────────┘
```

---

## Key Design Decisions

**Why LangGraph?**
Simple LLM chains go A→B→C with no way back. Research is iterative - sometimes you need to loop back and dig deeper. LangGraph enables conditional loops, state persistence, and human-in-the-loop checkpoints that aren't possible with basic chaining.

**Hallucination Firewall**
During crawling, every source gets a numbered ID. The LLM can only cite IDs from that list. The factcheck node then verifies every citation against the real crawled index and removes any claim citing a non-existent source. This is a structural solution to hallucination, not a prompt trick.

**Two-Step Writing**
Using structured output (JSON schema) for long-form writing causes LLMs to write minimally to satisfy the schema. The architect node first writes freely as a journalist (no schema constraints), then a second call parses that into structure. This produces significantly richer content.

**Episodic Memory**
Every completed research run is chunked and stored in ChromaDB. Future runs on related topics retrieve past research, allowing the strategist to avoid re-searching known information and the architect to cross-reference findings across sessions.

**Deep Fetch**
Tavily returns snippets by default. The crawler additionally fetches full page content from the top sources using direct HTTP requests, providing 3-4x more raw material for the architect to work with.

---

## Features

- Self-correcting quality loop with critic scoring
- Human-in-the-loop approval checkpoint
- Grounded citations - every claim tied to a real crawled URL
- Hallucination detection and removal
- Long-term memory across research sessions
- Deep web content fetching
- Professional PDF output with cover page, TOC, sections, and verified sources

---

## Setup

```bash
pip install langchain-groq langchain-tavily langchain-chroma langgraph fpdf2 requests python-dotenv
```

Create a `.env` file:
```
TAVILY_API_KEY = your_tavily_key
GROQ_API_KEY = your_groq_key
```

Get free API keys:
- Tavily: [tavily.com](https://tavily.com)
- Groq: [console.groq.com](https://console.groq.com)

---

## Usage

```bash
python agent.py "The rise and fall of BlackBerry 2000-2016"
```

Or any topic:
```bash
python agent.py "NVIDIA H100 vs A100 architecture"
python agent.py "Tesla FSD v12 neural network architecture"
```

The agent will:
1. Plan search queries and pause for your approval
2. Crawl the web and deep-fetch content
3. Write and self-correct the report
4. Export a PDF to the current directory

---

## Project Structure

```
ai_research_agent/
├── agent.py      # Main agent - graph, nodes, PDF export
├── rag.py        # ChromaDB memory layer
├── tools.py      # Arxiv and web search tools
└── .env          # API keys
```

---

## Tech Stack

- **LangGraph** - Agent orchestration and state management
- **LangChain** - LLM abstractions and tool integrations
- **Groq / OpenAI** - LLM inference
- **Tavily** - Web search API
- **ChromaDB** - Vector store for long-term memory
- **fpdf2** - PDF generation
