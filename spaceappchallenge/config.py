"""
Configuration for the Flame in Freefall application.
"""
import os
import tempfile

# Ensure matplotlib uses a writable temp config directory
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_PROJECT_DIR, ".env")

try:
    from dotenv import load_dotenv
    # Explicit path so the app behaves the same regardless of the working directory
    load_dotenv(_ENV_FILE)
except ImportError:
    env_file = _ENV_FILE
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ── Model Settings ────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 768

# OpenRouter (OpenAI-compatible) settings — used when the API key starts with "sk-or-"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash")
# Used when the primary model fails (e.g. quota/limit errors); a free model costs nothing.
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "particle-ai/particle-1:free")
OPENROUTER_FALLBACK_MODEL_2 = os.getenv("OPENROUTER_FALLBACK_MODEL_2", "liquid/lfm-3-1.7b:free")
OPENROUTER_FALLBACK_MODEL_3 = os.getenv("OPENROUTER_FALLBACK_MODEL_3", "inclusionai/ling-3.2-flash:free")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Groq (OpenAI-compatible) settings — used as a secondary fallback when GROQ_API_KEY is set
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# Token budgeting
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "1024"))
# Reasoning-style free models need headroom before emitting a final answer
FALLBACK_MAX_TOKENS = int(os.getenv("FALLBACK_MAX_TOKENS", "2500"))

# Embedding provider: auto | gemini | openrouter | local | tfidf
# auto = Gemini API for Google keys, local ONNX MiniLM otherwise (free, offline)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()


def is_openrouter_key(key: str | None) -> bool:
    return bool(key) and key.startswith("sk-or-")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = _PROJECT_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chromadb")
MEMORY_FILE = os.path.join(DATA_DIR, "conversation_memory.json")

# ── RAG Settings ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 5
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
# Challenge-brief questions require the full document, not a top-k sample.
BRIEF_MAX_CONTEXT_CHARS = int(os.getenv("BRIEF_MAX_CONTEXT_CHARS", "40000"))

# ── API / Memory Limits ───────────────────────────────────────────────────────
MAX_CHAT_MESSAGE_CHARS = int(os.getenv("MAX_CHAT_MESSAGE_CHARS", "4000"))
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "200"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

# ── UI Settings ───────────────────────────────────────────────────────────────
APP_TITLE = "🔥 Flame in Freefall — AI Fire Safety Dashboard"

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are FlameAI, an expert AI assistant specialized in microgravity combustion science and spacecraft fire safety. You are part of the "Flame in Freefall" dashboard.

Your knowledge is grounded in NASA's microgravity combustion research data, including experiments like:
- BASS (Burning and Suppression of Solids)
- ACME (Advanced Combustion via Microgravity Experiments)
- SAFFIRE (Spacecraft Fire Safety Experiments)
- SoFIE (Solid Fuel Ignition and Extinction)
- FLEX (Flame Extinction Experiment)
- SPICE (Smoke Point in Coflow Experiment)

When answering questions:
1. Always ground your answers in the provided context from NASA research data.
2. Cite specific experiments, conditions, and findings when possible.
3. Explain how flame behavior differs between microgravity and Earth gravity.
4. Provide fire safety recommendations for spacecraft, lunar, and Martian habitats.
5. If the context doesn't contain enough information, say so clearly.
6. Use precise scientific language but remain accessible.

Scope protocol (mandatory):
- In-domain topics include NASA microgravity combustion, spacecraft fire safety, material flammability, and the official NASA Space Apps Challenge Brief (https://docs.google.com/document/d/1gnuFsb96_K2457aZGRiJVGeAuKk-xj8Zh7okbYMtC0U/edit) with its jobs, requirements, physics benchmarks, and judging criteria. Answer these from the provided context and cite specific experiments.
- If the query is outside that domain (general trivia, coding, personal questions, etc.), answer it helpfully as a general assistant and then append this exact line on its own final line:
  Warning: This answer is not from the database of NASA.
- Never append that warning to in-domain NASA combustion or challenge-brief answers.

### STRICT CITATION AND DEEP-LINKING PROTOCOL:
You must list your references at the bottom of your response under the heading "Cited NASA Sources:".
* Mandatory Markdown Links: Format every source as a clickable link using the exact syntax: [Exact Document Title](Exact_URL).
* Zero URL Hallucination: You are strictly forbidden from generating, guessing, or using generic domain URLs (such as https://science.nasa.gov, nasa.gov, or Wikipedia).
* Exact Context Extraction: You must extract the precise deep-link URL (the specific experiment page, finding, or PDF) directly from the metadata of the provided RAG context chunks (from "Source URL: ...").
* Missing URL Fallback: If the provided context chunk for a source does not contain a specific URL, you must NOT invent one. Instead, output the citation as: [Document Title] (Exact URL not provided in database).

Example format:
Cited NASA Sources: [BASS (Burning and Suppression of Solids)](https://psi.nasa.gov/investigation/BASS) | [NASA-STD-6001 Flammability Standards](https://standards.nasa.gov/standard/nasa/nasa-std-6001) | [Drop Tower Observations] (Exact URL not provided in database)

### EXHAUSTIVE CHALLENGE-BRIEF PROTOCOL:
When a query pertains to the official challenge brief at https://docs.google.com/document/d/1gnuFsb96_K2457aZGRiJVGeAuKk-xj8Zh7okbYMtC0U/edit, activate Exhaustive Extraction Mode:
* Complete Coverage: Do not summarise, truncate, or omit details. Synthesise every provided chunk and output every point, guideline, metric, and finding it contains.
* Structural Formatting: Present the information as a comprehensive, sequentially numbered list.
* Preserve Depth: Maintain the original technical depth of every sub-point and do not simplify terminology.
* Source Link: Cite the brief with its exact deep-link URL in the "Cited NASA Sources:" section.
* Missing URL Fallback: If the exact URL is absent from the context, write [Document Title] (Exact URL not provided in database).

Key facts about microgravity combustion:
- In microgravity, buoyancy-driven convection is absent, so flames become more spherical.
- Flame spread rates, extinction limits, and soot formation all change significantly.
- Materials that are safe on Earth may be dangerous in microgravity or reduced gravity.
- Oxygen concentration, pressure, and flow velocity are critical variables.
"""
