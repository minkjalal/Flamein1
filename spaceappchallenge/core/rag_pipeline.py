"""
RAG (Retrieval-Augmented Generation) pipeline for the Flame in Freefall application.
Orchestrates vector search, knowledge graph queries, conversation memory,
and Gemini generation to produce grounded, source-cited answers.
"""

import logging
import re
from typing import Any, Generator, Optional

from config import BRIEF_MAX_CONTEXT_CHARS, MAX_CONTEXT_CHARS, TOP_K_RESULTS
from core.gemini_client import GeminiClient
from core.knowledge_graph import KnowledgeGraph
from core.memory import ConversationMemory
from core.vectorstore import VectorStore

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_WARNING = "Warning: This answer is not from the database of NASA."

# Prefix/word stems — matched at a word boundary on the left only.
_DOMAIN_STEMS = (
    "flammab", "buoyan", "extinguish", "combust", "ignit", "pyroly",
    "convect", "ventilat", "oxidiz", "spher", "microgravity",
)

# Whole words — ambiguous short tokens (e.g. "flex") must not match "flexible".
_DOMAIN_WORDS = (
    "nasa", "space", "spacecraft", "iss", "micro-gravity", "zero-g",
    "zero gravity", "freefall", "free fall", "astronaut", "habitat",
    "lunar", "moon", "martian", "mars", "leo", "exploration atmosphere",
    "flame", "fire", "burn", "soot", "smoke", "airflow", "ventilation",
    "oxygen", "o2", "loi", "limiting oxygen", "saffire", "sofie", "flex",
    "acme", "pmma", "nomex", "teflon", "ptfe", "kapton", "silicone",
    "epoxy", "nylon", "cotton", "polyethylene", "velcro", "polycarbonate",
    "combustor", "spherical flame", "cool flame", "gravity",
    "thermal runaway", "bass-ii", "bass-2", "bass-i",
    "challenge", "challenge brief", "spaceapps", "space apps", "hackathon",
    "judging", "judging criteria", "submission", "deliverable", "rubric",
    "mandatory ai jobs", "challenge document", "the brief", "this brief",
    "official brief", "brief document",
)

# Short tokens that are also common English words — only count with context.
_AMBIGUOUS_WORDS = ("bass", "spice")

# Uppercase experiment acronyms keep their meaning without extra context.
_ACRONYM_PATTERN = re.compile(r"\b(BASS(?:-I{1,2}|-[12])?|SPICE)\b")

_STEM_PATTERNS = [
    re.compile(rf"\b{re.escape(stem)}") for stem in _DOMAIN_STEMS
]
_WORD_PATTERNS = [
    re.compile(rf"\b{re.escape(word)}\b") for word in _DOMAIN_WORDS
]
_AMBIGUOUS_PATTERNS = [
    re.compile(rf"\b{re.escape(word)}\b") for word in _AMBIGUOUS_WORDS
]
_DOMAIN_CONTEXT_PATTERN = re.compile(
    r"\b(experiment|study|research|test|data|result|finding|nasa|space|"
    r"microgravity|combustion|fire|flame|spacecraft|habitat|saffire|iss)\b"
)

_BRIEF_QUERY_PATTERN = re.compile(
    r"\b(challenge\s+brief|challenge\s+document|challenge_brief|1gnufsb96|"
    r"docs\.google\.com|space\s*apps|judging\s+criteria|mandatory\s+ai\s+jobs|"
    r"submission\s+requirements|official\s+brief|brief\s+document|"
    r"the\s+brief|this\s+brief)\b",
    re.IGNORECASE,
)
_BRIEF_BARE_PATTERN = re.compile(r"\bbrief\b", re.IGNORECASE)
_BRIEF_CONTEXT_PATTERN = re.compile(
    r"\b(challenge|hackathon|judging|submission|space\s*apps|rubric)\b",
    re.IGNORECASE,
)
_CHALLENGE_BRIEF_URL = (
    "https://docs.google.com/document/d/"
    "1gnuFsb96_K2457aZGRiJVGeAuKk-xj8Zh7okbYMtC0U/edit"
)


def is_brief_query(question: str) -> bool:
    if _BRIEF_QUERY_PATTERN.search(question):
        return True
    return bool(
        _BRIEF_BARE_PATTERN.search(question)
        and _BRIEF_CONTEXT_PATTERN.search(question)
    )


def is_domain_query(question: str) -> bool:
    """Heuristic domain check used as a deterministic safety net for the
    prompt-level scope router in SYSTEM_PROMPT."""
    if _ACRONYM_PATTERN.search(question):
        return True
    q = question.lower()
    if any(pattern.search(q) for pattern in _STEM_PATTERNS):
        return True
    if any(pattern.search(q) for pattern in _WORD_PATTERNS):
        return True
    if any(pattern.search(q) for pattern in _AMBIGUOUS_PATTERNS):
        return bool(_DOMAIN_CONTEXT_PATTERN.search(q))
    return False


def ensure_scope_warning(answer: str, question: str) -> str:
    """Append the mandatory non-NASA warning to out-of-domain answers."""
    if is_domain_query(question):
        return answer
    if OUT_OF_SCOPE_WARNING.lower() in answer.lower():
        return answer
    separator = "" if answer.endswith("\n") else "\n\n"
    return f"{answer}{separator}{OUT_OF_SCOPE_WARNING}"


class _ScopeAwareStream:
    """Stream wrapper that appends the out-of-scope warning at end-of-stream."""

    def __init__(self, generator: Generator[str, None, None], question: str):
        self._gen = generator
        self._out_of_scope = not is_domain_query(question)
        self._collected: list[str] = []
        self._finished = False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        try:
            chunk = next(self._gen)
            self._collected.append(chunk)
            return chunk
        except StopIteration:
            self._finished = True
            text = "".join(self._collected)
            if self._out_of_scope and OUT_OF_SCOPE_WARNING.lower() not in text.lower():
                separator = "" if text.endswith("\n") else "\n\n"
                return f"{separator}{OUT_OF_SCOPE_WARNING}"
            raise


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(
        self,
        vectorstore: VectorStore,
        knowledge_graph: KnowledgeGraph,
        gemini_client: GeminiClient,
        memory: ConversationMemory,
    ):
        self.vectorstore = vectorstore
        self.knowledge_graph = knowledge_graph
        self.gemini_client = gemini_client
        self.memory = memory

    # ── Main Query ────────────────────────────────────────────────────────────

    def query(
        self,
        user_question: str,
        top_k: int = TOP_K_RESULTS,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Run the query through the AI routing layer, deciding whether RAG retrieval is required."""
        # ── Layer 1: AI Query Router (Conversational & Intent Classification) ────
        routed_response = self._evaluate_query_intent(user_question)
        if routed_response is not None:
            self.memory.add_turn(user_question, routed_response["answer"], [])
            return routed_response

        # ── Out-of-scope protocol: answer generally, cite no NASA sources ───────
        if not is_domain_query(user_question):
            if stream:
                gen = self.gemini_client.generate_response(
                    user_question, context=None, stream=True
                )
                collector = _StreamCollector(
                    _ScopeAwareStream(gen, user_question), self.memory, user_question, []
                )
                return {
                    "answer": collector,
                    "response": collector,
                    "sources": [],
                    "kg_context": "",
                }
            answer = ensure_scope_warning(
                str(self.gemini_client.generate_response(
                    user_question, context=None, stream=False
                )),
                user_question,
            )
            self.memory.add_turn(user_question, answer, [])
            return {"answer": answer, "response": answer, "sources": [], "kg_context": ""}

        # ── Layer 2: Domain RAG Pipeline (Combustion Science Retrieval) ──────────
        # Step 1 – Vector search (challenge-brief questions need the full document)
        brief_query = is_brief_query(user_question)
        effective_top_k = max(top_k, 12) if brief_query else top_k
        search_results = self.vectorstore.search(user_question, top_k=effective_top_k)
        if brief_query:
            brief_chunks = self.vectorstore.get_documents_by_source(
                "challenge_brief.json"
            )
            if brief_chunks:
                seen_texts = {chunk["text"] for chunk in brief_chunks}
                other_results = [
                    result for result in search_results
                    if result["text"] not in seen_texts
                ]
                search_results = brief_chunks + other_results

        # Step 2 – Knowledge graph
        kg_subgraph = self._query_knowledge_graph(user_question)
        kg_context = self.knowledge_graph.to_context_string(kg_subgraph) if kg_subgraph else ""

        # Step 3 – Combine context
        vs_context = self._format_vector_results(search_results)
        combined_context = self._combine_context(vs_context, kg_context)

        # Step 4 – Conversation memory
        memory_context = self.memory.get_recent_context(n=5)
        if memory_context:
            combined_context = f"{combined_context}\n\n{memory_context}"

        context_limit = BRIEF_MAX_CONTEXT_CHARS if brief_query else MAX_CONTEXT_CHARS
        combined_context = self._truncate_context(combined_context, context_limit)

        # Step 5 – Prepare user question
        prompt = user_question

        # Step 6 & 7 – Generate and return
        sources = self._extract_sources(search_results)

        if stream:
            response_gen = self.gemini_client.generate_response(
                prompt, context=combined_context, stream=True
            )
            scoped_gen = _ScopeAwareStream(response_gen, user_question)
            collector = _StreamCollector(scoped_gen, self.memory, user_question, sources)
            return {
                "answer": collector,
                "response": collector,
                "sources": sources,
                "kg_context": kg_context,
            }
        else:
            response_text = self.gemini_client.generate_response(
                prompt, context=combined_context, stream=False
            )
            response_text = ensure_scope_warning(str(response_text), user_question)
            self.memory.add_turn(user_question, response_text, sources)
            return {
                "answer": response_text,
                "response": response_text,
                "sources": sources,
                "kg_context": kg_context,
            }

    def _evaluate_query_intent(self, question: str) -> Optional[dict[str, Any]]:
        """Intelligent routing layer: determines if the query requires NASA RAG retrieval."""
        import re
        q = question.strip().lower()
        q_clean = re.sub(r"[^\w\s]", "", q).strip()

        # 1. Greetings (helo, hello, hi, hey, etc.)
        greetings = {
            "hi", "helo", "hello", "hey", "hiya", "howdy", "sup", "yo", "greetings",
            "good morning", "good afternoon", "good evening", "hey there", "hello there"
        }
        if q_clean in greetings or re.match(r"^(hi|helo|hello|hey|howdy)(\s+(there|flameai|bot|assistant))?$", q_clean):
            answer = (
                "Hello! I am FlameAI, your expert assistant for microgravity combustion science and spacecraft fire safety.\n\n"
                "I can assist you with:\n"
                "• Microgravity flame behavior, sphericity, and cool flames (FLEX, FLEX-2)\n"
                "• Spacecraft fire experiments (BASS, SAFFIRE, ACME, SoFIE, SPICE)\n"
                "• Material flammability in space (Nomex, PMMA, silicone, Teflon, Kapton)\n"
                "• Fire safety and suppression guidelines for ISS, Lunar (0.16g), and Martian (0.38g) habitats\n\n"
                "What would you like to explore today?"
            )
            return {"answer": answer, "response": answer, "sources": [], "kg_context": ""}

        # 2. Capabilities & Identity
        capabilities = {
            "who are you", "what are you", "what can you do", "help", "commands",
            "what is this app", "what is this", "features", "about", "what is flame in freefall"
        }
        if q_clean in capabilities or re.match(r"^(who|what)\s+are\s+you$", q_clean) or q_clean == "help":
            answer = (
                "I am FlameAI, part of the Flame in Freefall dashboard for the NASA Space Apps Challenge 2026.\n\n"
                "Capabilities:\n"
                "• Grounded Research Retrieval: Instant insights from NASA PSI & NTRS experiments (BASS, ACME, SAFFIRE, FLEX, SoFIE, SPICE).\n"
                "• Spacecraft Material Flammability: Limiting Oxygen Index (LOI), flame spread velocity, and toxicity ratings.\n"
                "• Lunar & Martian Habitat Safety: Specific guidelines for partial gravity and exploration atmospheres (34% O2).\n"
                "• Knowledge Graph Analysis: Relates experiments, materials, and safety directives.\n\n"
                "Try asking: 'Why do flames become spherical in microgravity?' or 'What did SAFFIRE teach us about spacecraft fires?'"
            )
            return {"answer": answer, "response": answer, "sources": [], "kg_context": ""}

        # 3. Gratitude & Closings
        closings = {
            "thanks", "thank you", "thx", "bye", "goodbye", "see you", "ok", "okay",
            "cool", "great", "awesome", "perfect", "got it"
        }
        if q_clean in closings or re.match(r"^(thanks|thank you)(\s+so\s+much|\s+a\s+lot)?$", q_clean):
            answer = (
                "You are welcome! Let me know if you need any more analysis on microgravity combustion or fire safety parameters."
            )
            return {"answer": answer, "response": answer, "sources": [], "kg_context": ""}

        # 4. Small talk / status checks
        if re.match(
            r"^(how are you( doing| today)?|how s it going|how is it going|"
            r"whats up|what is up|are you (there|ok|okay)|nice to meet you|"
            r"who made you|who built you)\??$",
            q_clean,
        ):
            answer = (
                "I'm running well and ready to analyse NASA's microgravity combustion data. "
                "Ask me about flame behavior, material flammability, or spacecraft fire safety."
            )
            return {"answer": answer, "response": answer, "sources": [], "kg_context": ""}

        # Query requires deep NASA knowledge base retrieval -> proceed to RAG
        return None

    # ── Convenience Methods ───────────────────────────────────────────────────

    def get_sources(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict[str, str]]:
        """Return relevant source documents without generating an AI response."""
        results = self.vectorstore.search(query, top_k=top_k)
        return self._extract_sources(results)

    def rank_experiments(self, criteria: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Rank experiments by relevance to the given criteria.

        Uses vector search scores as the ranking signal.
        """
        results = self.vectorstore.search(criteria, top_k=top_k)

        # Group by experiment and take the best score per experiment
        experiment_scores: dict[str, dict[str, Any]] = {}
        for r in results:
            exp_name = r["metadata"].get("experiment_name", "unknown")
            score = r.get("score", 0)
            if exp_name not in experiment_scores or score > experiment_scores[exp_name]["score"]:
                experiment_scores[exp_name] = {
                    "experiment_name": exp_name,
                    "score": score,
                    "best_chunk": r["text"][:200],
                    "source": r["metadata"].get("source", ""),
                    "category": r["metadata"].get("category", ""),
                }

        ranked = sorted(experiment_scores.values(), key=lambda x: x["score"], reverse=True)
        return ranked

    # ── Private helpers ───────────────────────────────────────────────────────

    def _query_knowledge_graph(self, question: str):
        """Try to find a relevant knowledge-graph subgraph for the question.

        Extracts candidate entity names from the question by looking for
        known graph nodes that appear (case-insensitive) in the question text.
        """
        question_lower = question.lower()
        matching_nodes: list[str] = []

        for node in self.knowledge_graph.graph.nodes:
            node_lower = node.lower()
            if len(node_lower) < 3:
                continue
            # Word-boundary match so "ISS" does not match "kiss" and short
            # material names do not match inside unrelated words.
            if re.search(rf"\b{re.escape(node_lower)}\b", question_lower):
                matching_nodes.append(node)

        if not matching_nodes:
            return None

        # Prefer the longest (most specific) matches, e.g. an experiment name
        # over a material mentioned inside it.
        matching_nodes.sort(key=len, reverse=True)

        # Merge subgraphs for all matched nodes
        import networkx as nx

        combined = nx.DiGraph()
        for node in matching_nodes[:5]:  # limit to top 5 matches
            sg = self.knowledge_graph.query_related(node, depth=2)
            if sg:
                combined = nx.compose(combined, sg)

        return combined if combined.number_of_nodes() > 0 else None

    @staticmethod
    def get_source_url(name: str = "", source: str = "") -> str:
        """Resolve a precise deep-link for an experiment, standard, or brief."""
        combined = f"{name} {source}".lower()
        if any(term in combined for term in (
            "1gnufsb96", "challenge_brief", "challenge brief", "google doc",
        )):
            return _CHALLENGE_BRIEF_URL
        if "bass" in combined:
            return "https://psi.nasa.gov/investigation/BASS"
        if "acme" in combined:
            return "https://psi.nasa.gov/investigation/ACME"
        if "saffire" in combined:
            return "https://ntrs.nasa.gov/citations/20160005234"
        if "flex" in combined:
            return "https://psi.nasa.gov/investigation/FLEX"
        if "sofie" in combined:
            return "https://psi.nasa.gov/investigation/SoFIE"
        if "spice" in combined:
            return "https://psi.nasa.gov/investigation/SPICE"
        if any(term in combined for term in (
            "material", "polymer", "acrylic", "nomex", "teflon", "kapton",
            "flammability", "nasa-std-6001", "std-6001",
            "guideline", "safety", "protocol", "habitat",
        )):
            return "https://standards.nasa.gov/standard/nasa/nasa-std-6001"
        return "Exact URL not provided in database"

    @staticmethod
    def _format_vector_results(results: list[dict[str, Any]]) -> str:
        """Format results with explicit source titles and deep links for citation."""
        if not results:
            return ""

        parts: list[str] = ["## Retrieved NASA Research Chunks"]
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            title = (
                meta.get("record_title")
                or meta.get("experiment_name")
                or meta.get("title")
                or "Unknown Source"
            )
            source_file = (
                meta.get("source", "").replace(".json", "").replace("_", " ").title()
            )
            if source_file and source_file.lower() not in str(title).lower():
                full_title = f"{title} ({source_file})"
            else:
                full_title = str(title)

            url = str(meta.get("url") or "").strip()
            if not url.startswith("http"):
                url = RAGPipeline.get_source_url(full_title, meta.get("source", ""))

            parts.append(f"### Source {i} — {full_title}")
            parts.append(f"Source Title: {full_title}")
            parts.append(f"Source URL: {url}")
            parts.append("Content: " + result["text"])
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _combine_context(vs_context: str, kg_context: str) -> str:
        """Merge vector-store and knowledge-graph context strings."""
        sections = [s for s in [vs_context, kg_context] if s]
        return "\n\n".join(sections) if sections else "No relevant context found in the database."

    @staticmethod
    def _truncate_context(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
        """Cap the injected context so prompts stay within token budgets."""
        if limit <= 0 or len(text) <= limit:
            return text
        cut = text[:limit]
        boundary = cut.rfind("\n")
        if boundary > limit // 2:
            cut = cut[:boundary]
        return cut.rstrip() + "\n\n[Context truncated to fit the model budget]"

    @staticmethod
    def _extract_sources(results: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Pull unique source titles and deep links out of search results."""
        sources: list[dict[str, str]] = []
        seen: set[str] = set()
        for result in results:
            meta = result.get("metadata", {})
            name = (
                meta.get("record_title")
                or meta.get("experiment_name")
                or "NASA Combustion Record"
            )
            source_file = (
                meta.get("source", "").replace(".json", "").replace("_", " ").title()
            )
            label = f"{name} ({source_file})" if source_file else str(name)
            url = str(meta.get("url") or "").strip()
            if not url.startswith("http"):
                url = RAGPipeline.get_source_url(str(name), meta.get("source", ""))
            key = f"{label}|{url}"
            if key not in seen and name != "unknown":
                seen.add(key)
                sources.append({"title": label, "url": url})
        return sources


class _StreamCollector:
    """Iterator wrapper that captures streamed text and saves it to memory
    once the stream is exhausted."""

    def __init__(
        self,
        generator: Generator[str, None, None],
        memory: ConversationMemory,
        user_question: str,
        sources: list[Any],
    ):
        self._gen = generator
        self._memory = memory
        self._user_question = user_question
        self._sources = sources
        self._collected: list[str] = []

    def __iter__(self):
        return self

    def __next__(self) -> str:
        try:
            chunk = next(self._gen)
            self._collected.append(chunk)
            return chunk
        except StopIteration:
            full_response = "".join(self._collected)
            if full_response:
                self._memory.add_turn(self._user_question, full_response, self._sources)
            raise

    def __str__(self) -> str:
        if not self._collected:
            for _ in self:
                pass
        return "".join(self._collected)
