# 🔥 Flame in Freefall
### AI-Powered Fire Safety Insights from Microgravity Combustion Data
**NASA International Space Apps Challenge 2026 — Local Event: Haldwani, India**  
**Team:** Eagle20  
**Challenge:** *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*

---

## 🌌 Overview

For decades, NASA has studied how flames and fire behave in microgravity through experiments aboard the International Space Station (ISS), the Space Shuttle, drop towers, and uncrewed Cygnus cargo vehicles. As humanity prepares for long-duration missions back to the Moon (Artemis) and onto Mars, understanding microgravity combustion physics is vital to crew survival.

However, the volume and complexity of NASA physical sciences results makes them difficult to find, compare, and synthesize.

**Flame in Freefall** is an interactive web dashboard that:
1. **Summarizes** decades of NASA microgravity combustion experiments.
2. **Ranks** findings and materials by relevance and risk.
3. **Interprets** physics findings into actionable fire safety guidelines for lunar and Martian habitats.
4. **Reasons** using a hybrid **Graph-RAG** architecture grounded in real NASA data.

---

## 🏗️ System Architecture

```
                                  +---------------------------------+
                                  |   NASA PSI & NTRS Repositories  |
                                  |  (BASS, ACME, SAFFIRE, SoFIE)   |
                                  +----------------+----------------+
                                                   |
                                                   v
                                  +----------------+----------------+
                                  |     Data Ingestion Pipeline     |
                                  | (Overlapping Semantic Chunking) |
                                  +--------+---------------+--------+
                                           |               |
                    +----------------------+               +---------------------+
                    v                                                            v
+---------------------------------------+                +---------------------------------------+
|        Vector Store (Semantic)        |                |        Knowledge Graph (NetworkX)     |
|   - ChromaDB / NumPy Cosine Search    |                |   - Experiment -> Material Edges      |
|   - Gemini text-embedding-004         |                |   - Condition -> Finding Edges        |
+-------------------+-------------------+                +-------------------+-------------------+
                    |                                                        |
                    +----------------------+        +------------------------+
                                           v        v
                                  +----------------+----------------+
                                  |          RAG Pipeline           |
                                  |    - Context Fusion             |
                                  |    - Multi-Turn Memory (JSON)   |
                                  | - OpenRouter/Groq/Gemini + SSE  |
                                  +----------------+----------------+
                                                   |
                                                   v
                                  +---------------------------------+
                                  |   FastAPI Web Backend (server)  |
                                  |   - REST API (/api/*)           |
                                  |   - Serves frontend/index.html  |
                                  +----------------+----------------+
                                                   |
                                                   v
                                  +---------------------------------+
                                  |   HTML / JS Web Frontend        |
                                  | - Mission Overview Dashboard    |
                                  | - Grounded FlameAI Chat         |
                                  | - Experiment Explorer & Filters |
                                  | - Side-by-Side Experiment Diff  |
                                  | - Habitat Fire Safety Scorer    |
                                  | - Knowledge Graph Visualizer    |
                                  +---------------------------------+
```

---

## ✨ Key Features

### 1. 📊 Mission Overview Dashboard
- High-level mission stat cards: Ingested Experiments, Materials Tested, Key Findings, and Safety Guidelines.
- Real-time **Chart.js** charts showing experiment distributions across NASA programs and material types.
- Live feed of key experimental insights with direct attribution.
- **AI Mission Brief** (`/api/summary`): headline synthesis, top-ranked experiment, highest-risk material in a 34% O₂ exploration atmosphere, priority directive, and recommended actions — all derived from the ingested datasets.

### 2. 🤖 FlameAI Chat (Hybrid Graph-RAG Pipeline)
- Conversational assistant powered by **OpenRouter, Google Gemini, or Groq** (provider order: OpenRouter → Google → Groq), falling back to fully offline knowledge-grounding when no provider is reachable.
- **Token streaming over SSE** (`POST /api/chat/stream`) with a non-streaming fallback, rendered progressively in the chat UI.
- Safe **markdown rendering** for bold text, scientific parameters, bullet points, numbered lists, and clickable citations.
- **Graph-Augmented Retrieval (Graph-RAG):** Combines dense vector similarity with knowledge graph entity traversal to eliminate hallucinations.
- Cites real NASA datasets and research papers for every answer, with **one-click deep links** to PSI experiments, NTRS records, NASA-STD-6001, and the official challenge brief.
- **Strict citation protocol:** the model may only use source URLs supplied in the retrieved context and must fall back to *"Exact URL not provided in database"* rather than inventing a link.
- **Exhaustive challenge-brief mode:** questions about the NASA Space Apps challenge brief retrieve the complete `challenge_brief.json` corpus and require full, sequentially numbered coverage of every job, requirement, physics benchmark, and judging criterion.
- **Scope protocol:** out-of-domain questions are answered generally and tagged with the mandatory amber badge *"Warning: This answer is not from the database of NASA."* No NASA sources are attached to those answers. Domain detection uses word-boundary matching, so words like "flexible" or phrases like "bass guitar" are not misclassified.
- **Inline charts on request:** ask for a graph/plot of an experiment (e.g. "plot SAFFIRE-I" or "graph BASS") and the chat renders a matplotlib chart generated directly from the experiment record, including misspelled names.
- **Thematic charts:** requests without a specific experiment (e.g. "graph spherical flames", "plot quenching distance", "chart cool flames", "graph flame spread", "chart material flammability") render grounded multi-panel figures from local findings and material data.
- Multi-turn conversation memory with persistence, bounded size, thread-safe atomic writes, and a Clear Chat control.

### 3. 🔬 Experiment Explorer
- Search and browse 12+ major NASA combustion experiments (BASS, BASS-II, ACME-BRE, SAFFIRE I-III, FLEX-2, SPICE, SoFIE).
- Filter by category, test materials, atmospheric pressure, and oxygen concentrations.
- Detailed card view with flammability boundaries, flame sphericity, spread rates, and O₂ / pressure / flow parameter tags.

### 4. ⚖️ Side-by-Side Experiment Comparison
- Compare two NASA experiments across key combustion parameters:
  - Flame Spread Rate
  - Extinction Time
  - Soot Yield
  - Heat Release Rate
  - Overall Safety Score
- Interactive **5-axis radar chart** comparing combustion metrics visually.
- Scorecard table with colored per-metric deltas (`Δ Exp1 − Exp2`) highlighting which experiment scored higher.

### 5. 🛡️ Fire Safety Risk Scorer for Space Habitats
- Custom assessment tool tailored for **ISS (LEO)**, **Lunar Habitats (0.16g)**, and **Martian Habitats (0.38g)**.
- Select target materials (e.g. Nomex®, PMMA, Silicone, Kapton, Cotton) and ambient oxygen levels (up to exploration 34% O₂).
- Generates safety scores (0-100), key risk factors, and engineering recommendations.
- Ranked safety tables for habitat structural materials.

### 6. 🕸️ Knowledge Graph Visualizer
- Interactive NetworkX graph visualizer showing the web of connections between:
  - 🟠 Experiments
  - 🔵 Materials
  - 🟢 Findings
  - 🔴 Safety Implications
  - 🟣 Safety Guidelines
- Built from the real dataset schemas (`materials_tested`, `parameters`, `key_findings`, `safety_implications`, plus findings, materials, and guideline records): 316 nodes and 527 edges.
- Material-name normalization maps experiment labels such as `PMMA (acrylic)` to the canonical material record.
- Clickable nodes with detailed properties sidebar.

### 7. 🌗 Dark / Light Theme
- Accessible slider switch in the top navigation, plus an appearance control in the mobile drawer.
- Theme preference persists in `localStorage` and defaults to light mode.
- Semantic CSS colour tokens restyle the navigation, glass surfaces, inputs, chat bubbles, tables, footer, ambient glow, Chart.js charts, and the knowledge-graph canvas.
- Chart.js visualisations are regenerated with theme-aware colours when the switch changes, and the browser `theme-color` meta tag updates in sync.

### 8. 📈 Explainable Rankings & Unified Search
Every ranked item exposes its score, component breakdown, rank, and human-readable reasons.
- **Experiments** — safety findings, research findings, material breadth, category relevance, recency, and platform access.
- **Materials** — flammability, toxicity, extinction O₂ threshold, smoke output, and environment adjustment (ISS / Lunar / Martian / 34% O₂).
- **Findings** — safety relevance, quantitative evidence, and explicit spacecraft implications.
- **Guidelines** — priority, risk level, experiment coverage, and environment applicability.
- **Unified search** across all four datasets with typed results and snippets (`/api/search`).
- **Similar findings** retrieval by category, source experiment, and vocabulary overlap.

---

## 📂 NASA Combustion Datasets Covered

Our knowledge base incorporates published experimental findings from NASA's Physical Sciences Informatics (PSI) system and NASA Technical Reports Server (NTRS):

| Experiment | Facility / Platform | Primary Research Focus |
|------------|---------------------|------------------------|
| **BASS / BASS-II** | ISS Microgravity Science Glovebox (MSG) | Burning & extinction limits of solid polymers (PMMA, cotton) under forced airflow |
| **ACME (BRE, CLD, s-Flame)** | ISS Combustion Integrated Rack (CIR) | Gaseous diffusion flames, flame lift-off, and electric-field suppression |
| **SAFFIRE I – VI** | Northrop Grumman Cygnus Spacecraft | Real-scale spacecraft fire propagation and atmospheric depletion dynamics |
| **FLEX / FLEX-2** | ISS Multi-user Droplet Combustion App. | Droplet combustion, radiative extinction, and discovery of low-temperature cool flames |
| **SPICE** | ISS CIR | Smoke point, soot volume fractions, and agglomeration in microgravity coflows |
| **SoFIE** | ISS CIR | Solid fuel ignition and extinction under exploration atmosphere conditions |
| **Challenge Brief** | NASA Space Apps 2026 | Official challenge scope, three mandatory AI jobs (Summarise, Rank, Interpret), technical requirements, combustion benchmarks, and judging criteria |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- Windows, macOS, or Linux (no GUI toolkit required — the app runs in the browser)

### 2. Installation
Clone or navigate to the repository directory:
```bash
cd spaceappchallenge
```

Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure the API Key
Open `.env` in the project root. Either an OpenRouter key (`sk-or-...`) or a Google AI Studio key (`AIza...`) works:
```env
GEMINI_API_KEY=sk-or-v1...your_key_here
```
- **OpenRouter keys** use `google/gemini-2.5-flash`, falling back to free models if the primary is rate-limited or out of credits.
- **Google AI Studio keys** use the Gemini SDK/REST directly.
- **Groq (optional):** set `GROQ_API_KEY=gsk_...` to enable `qwen/qwen3.8-27b` as a secondary provider when OpenRouter or Gemini fails.
- **Provider order:** OpenRouter → Google Gemini → Groq → offline grounded fallback.
- **Token budget:** `CHAT_MAX_TOKENS=1024` for normal answers; reasoning-style OpenRouter fallbacks receive `FALLBACK_MAX_TOKENS=2500` so they can finish thinking and still emit an answer.

Embeddings run locally with ChromaDB's bundled ONNX MiniLM model (downloaded once on first run, no API cost). Override with `EMBEDDING_PROVIDER=gemini|openrouter|local|tfidf` if needed.

Without a usable key, the app still works using the local NASA knowledge base grounding.

### REST API
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Status, AI provider, data counts, graph size, vector-index stats |
| GET | `/api/stats` | Dashboard counts and distributions |
| GET | `/api/filters` | Available filter vocabulary |
| GET | `/api/summary` | Mission brief: headline, top experiment, highest-risk material, priority directive |
| GET | `/api/rank/experiments?category=&limit=` | Explainable experiment ranking |
| GET | `/api/rank/materials?environment=&limit=` | Fire-risk material ranking |
| GET | `/api/rank/findings?category=&relevance=&limit=` | Safety-relevance finding ranking |
| GET | `/api/rank/guidelines?environment=&limit=` | Priority guideline ranking |
| GET | `/api/search?q=&limit=` | Unified search across every dataset |
| GET | `/api/experiments?query=&category=&platform=&limit=` | Search/filter experiments |
| GET | `/api/experiments/{id}` | Experiment detail + graph connections + related findings/guidelines |
| GET | `/api/materials?query=&category=`, `/api/materials/{name}` | Material list/detail + safety implications |
| GET | `/api/findings?query=&category=&relevance=` | Findings list |
| GET | `/api/findings/{id}` | Finding detail + similar findings |
| GET | `/api/guidelines?query=&environment=&risk_level=` | Guideline list |
| GET | `/api/guidelines/{id}` | Guideline detail |
| GET | `/api/compare?e1=&e2=` | Radar metrics for two experiments (fuzzy name resolution) |
| POST | `/api/score` | Data-driven habitat fire-risk score and ranked safety table |
| GET | `/api/graph` | Knowledge graph nodes/edges |
| POST | `/api/chat` | Non-streaming RAG answer |
| POST | `/api/chat/stream` | SSE stream: `sources` → `token` → `done` |
| DELETE | `/api/chat/memory` | Clear conversation memory |

**Operational safeguards**
- Chat messages are capped at `MAX_CHAT_MESSAGE_CHARS` (default 4000) with Pydantic validation.
- RAG context is truncated to `MAX_CONTEXT_CHARS` (default 12000) before generation; exhaustive challenge-brief questions use `BRIEF_MAX_CONTEXT_CHARS` (default 40000) so the full document can be covered.
- Conversation memory is capped at `MEMORY_MAX_TURNS` (default 200) and written atomically.
- CORS defaults to `*` without credentials; set `CORS_ORIGINS=https://your-host` to lock it down.
- On startup the vector index is rebuilt automatically when the corpus digest or embedding backend changes (local MiniLM 384-dim vs Gemini 768-dim), preventing silent dimension mismatches.

### 4. Run the Web Application
Start the FastAPI backend, which also serves the HTML frontend:
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```
On Windows you can also double-click `run_frontend.bat`.

Open **http://localhost:8000** in your browser.

---

## 👥 Team Eagle20
- **Hackathon:** NASA Space Apps Challenge 2026
- **Location:** Haldwani, Uttarakhand, India
- **Motto:** *Born to soar. Built to dominate.*
