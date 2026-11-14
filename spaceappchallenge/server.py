"""
FastAPI backend for Flame in Freefall.

Serves the RAG chat pipeline, the knowledge graph, explainable rankings,
mission-brief analytics, and the dashboard frontend.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (
    APP_TITLE,
    CORS_ORIGINS,
    MAX_CHAT_MESSAGE_CHARS,
    TOP_K_RESULTS,
)
from core.charts import maybe_generate_chart
from core.gemini_client import GeminiClient
from core.ingestion import load_and_chunk_data
from core.insights import (
    build_filters,
    build_summary,
    find_similar_findings,
    rank_experiments,
    rank_findings,
    rank_guidelines,
    rank_materials,
    safety_score_for_material,
    search_all,
)
from core.knowledge_graph import KnowledgeGraph
from core.memory import ConversationMemory
from core.rag_pipeline import RAGPipeline
from core.vectorstore import VectorStore
from utils.data_loader import (
    get_experiment_by_id,
    get_material_by_name,
    load_all_data,
)
from utils.helpers import color_for_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlameServer")

app = FastAPI(title=APP_TITLE, version="2.2.0")

_allow_all_origins = "*" in CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all_origins else CORS_ORIGINS,
    # Wildcard origins cannot be combined with credentialed requests.
    allow_credentials=not _allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Singleton Core Engine State ────────────────────────────────────────────────
class EngineState:
    def __init__(self):
        self.all_data = load_all_data()
        self.kg = KnowledgeGraph()
        self.kg.build_from_data(
            self.all_data.get("experiments", []),
            self.all_data.get("materials", []),
            self.all_data.get("findings", []),
            self.all_data.get("guidelines", []),
        )

        self.vectorstore = VectorStore()
        self._sync_vector_index()

        self.gemini = GeminiClient()
        self.memory = ConversationMemory()
        self.rag = RAGPipeline(
            vectorstore=self.vectorstore,
            knowledge_graph=self.kg,
            gemini_client=self.gemini,
            memory=self.memory,
        )

    def _sync_vector_index(self) -> None:
        """Rebuild the index when the corpus or embedding model changed."""
        chunks = load_and_chunk_data()
        if not chunks:
            return
        try:
            if self.vectorstore.index_is_current(chunks):
                return
            logger.info("Vector index is stale — rebuilding from %d chunks.", len(chunks))
            self.vectorstore.clear()
            added = self.vectorstore.add_documents(chunks)
            logger.info("Vector index rebuilt with %d chunks.", added)
        except Exception as exc:
            logger.error("Vector index sync failed: %s", exc)

    def experiments(self) -> list[dict[str, Any]]:
        return self.all_data.get("experiments", [])

    def materials(self) -> list[dict[str, Any]]:
        return self.all_data.get("materials", [])

    def findings(self) -> list[dict[str, Any]]:
        return self.all_data.get("findings", [])

    def guidelines(self) -> list[dict[str, Any]]:
        return self.all_data.get("guidelines", [])

    def challenge_brief(self) -> list[dict[str, Any]]:
        return self.all_data.get("challenge_brief", [])


engine = EngineState()


# ── Request / Response Models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)


class ScoreRequest(BaseModel):
    material: str = Field(..., min_length=1, max_length=120)
    environment: str = Field("ISS", max_length=120)
    oxygen_level: str = Field("20.9%", max_length=60)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _require_experiment(identifier: str) -> dict[str, Any]:
    experiment = get_experiment_by_id(identifier)
    if experiment is None:
        available = ", ".join(
            exp.get("acronym", exp.get("name", "")) for exp in engine.experiments()
        )
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{identifier}' not found. Available: {available}",
        )
    return experiment


def _experiment_related_findings(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    acronym = str(experiment.get("acronym", "")).lower()
    name = str(experiment.get("name", "")).lower()
    related = []
    for finding in engine.findings():
        source = str(finding.get("source_experiment", "")).lower()
        if source and (source == acronym or source in name or acronym in source):
            related.append(finding)
    return related


def _experiment_related_guidelines(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {
        str(experiment.get("acronym", "")).lower(),
        str(experiment.get("id", "")).lower(),
        str(experiment.get("name", "")).lower(),
    }
    related = []
    for guideline in engine.guidelines():
        for reference in guideline.get("related_experiments", []) or []:
            ref = str(reference).lower()
            if ref and any(ref == candidate or ref in candidate for candidate in candidates):
                related.append(guideline)
                break
    return related


def _parse_oxygen_level(value: str) -> Optional[float]:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def _metrics_for(experiment: dict[str, Any]) -> dict[str, float]:
    """Curated comparison metrics for the radar chart, keyed by experiment family."""
    families = {
        "SAFFIRE": {"flame_spread": 7.2, "extinction_time": 6.8, "soot_yield": 5.5, "heat_release": 6.5, "safety_score": 7.2},
        "BASS": {"flame_spread": 5.8, "extinction_time": 7.5, "soot_yield": 4.5, "heat_release": 7.8, "safety_score": 6.5},
        "SOFIE": {"flame_spread": 3.2, "extinction_time": 8.5, "soot_yield": 2.5, "heat_release": 4.2, "safety_score": 8.8},
        "FLEX": {"flame_spread": 2.1, "extinction_time": 4.5, "soot_yield": 6.2, "heat_release": 8.9, "safety_score": 4.5},
        "ACME": {"flame_spread": 4.5, "extinction_time": 7.0, "soot_yield": 3.5, "heat_release": 5.6, "safety_score": 7.8},
        "SPICE": {"flame_spread": 3.8, "extinction_time": 6.0, "soot_yield": 8.5, "heat_release": 5.0, "safety_score": 6.0},
    }
    haystack = f"{experiment.get('acronym', '')} {experiment.get('name', '')}".upper()
    for family, metrics in families.items():
        if family in haystack:
            return metrics
    return {"flame_spread": 5.0, "extinction_time": 6.0, "soot_yield": 5.0, "heat_release": 6.0, "safety_score": 7.0}


# ── Health & Dashboard Metadata ────────────────────────────────────────────────
@app.get("/api/health")
def health():
    index_stats = engine.vectorstore.get_collection_stats()
    return {
        "status": "healthy",
        "version": "2.2.0",
        "title": APP_TITLE,
        "ai_engine": engine.gemini.provider_label,
        "ai_configured": engine.gemini.is_configured,
        "experiments_count": len(engine.experiments()),
        "data": {
            "experiments": len(engine.experiments()),
            "materials": len(engine.materials()),
            "findings": len(engine.findings()),
            "guidelines": len(engine.guidelines()),
            "challenge_brief": len(engine.challenge_brief()),
        },
        "knowledge_graph": {
            "nodes": engine.kg.graph.number_of_nodes(),
            "edges": engine.kg.graph.number_of_edges(),
        },
        "vector_index": index_stats,
        "memory_turns": len(engine.memory),
    }


@app.get("/api/stats")
def get_stats():
    experiments = engine.experiments()
    materials = engine.materials()

    categories: dict[str, int] = {}
    for experiment in experiments:
        category = experiment.get("category", "General")
        categories[category] = categories.get(category, 0) + 1

    material_types: dict[str, int] = {}
    for material in materials:
        category = material.get("category", "Other")
        material_types[category] = material_types.get(category, 0) + 1

    return {
        "total_experiments": len(experiments),
        "materials_tested": len(materials),
        "key_findings": len(engine.findings()),
        "safety_guidelines": len(engine.guidelines()),
        "challenge_brief_records": len(engine.challenge_brief()),
        "categories": categories,
        "material_types": material_types,
        "ai_configured": engine.gemini.is_configured,
        "ai_engine": engine.gemini.provider_label,
        "knowledge_graph": {
            "nodes": engine.kg.graph.number_of_nodes(),
            "edges": engine.kg.graph.number_of_edges(),
        },
        "vector_documents": engine.vectorstore.get_collection_stats().get("document_count", 0),
    }


@app.get("/api/filters")
def get_filters():
    return build_filters(engine.all_data)


@app.get("/api/summary")
def get_summary():
    return build_summary(
        engine.experiments(),
        engine.materials(),
        engine.findings(),
        engine.guidelines(),
    )


# ── Rankings ──────────────────────────────────────────────────────────────────
@app.get("/api/rank/experiments")
def rank_experiments_endpoint(
    category: Optional[str] = Query(None),
    limit: int = Query(12, ge=1, le=100),
):
    return {"experiments": rank_experiments(engine.experiments(), category, limit)}


@app.get("/api/rank/materials")
def rank_materials_endpoint(
    environment: str = Query("ISS"),
    limit: int = Query(16, ge=1, le=100),
):
    return {"materials": rank_materials(engine.materials(), environment, limit)}


@app.get("/api/rank/findings")
def rank_findings_endpoint(
    category: Optional[str] = Query(None),
    relevance: Optional[str] = Query(None),
    limit: int = Query(25, ge=1, le=100),
):
    return {"findings": rank_findings(engine.findings(), category, relevance, limit)}


@app.get("/api/rank/guidelines")
def rank_guidelines_endpoint(
    environment: Optional[str] = Query(None),
    limit: int = Query(15, ge=1, le=100),
):
    return {"guidelines": rank_guidelines(engine.guidelines(), environment, limit)}


@app.get("/api/search")
def search_endpoint(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=100),
):
    results = search_all(
        q,
        engine.experiments(),
        engine.materials(),
        engine.findings(),
        engine.guidelines(),
        limit,
    )
    return {"query": q, "count": len(results), "results": results}


# ── Dataset Listings ─────────────────────────────────────────────────────────
@app.get("/api/experiments")
def list_experiments(
    query: Optional[str] = Query(None, max_length=200),
    category: Optional[str] = Query(None),
    platform: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=100),
):
    experiments = engine.experiments()
    if query:
        q = query.lower()

        def matches(experiment: dict[str, Any]) -> bool:
            haystack = " ".join(
                str(experiment.get(key, ""))
                for key in ("name", "acronym", "id", "description", "category")
            ).lower()
            if q in haystack:
                return True
            for key in ("objectives", "materials_tested", "key_findings", "safety_implications"):
                if any(q in str(item).lower() for item in experiment.get(key, []) or []):
                    return True
            return False

        experiments = [experiment for experiment in experiments if matches(experiment)]

    if category and category not in ("All", "All Categories"):
        experiments = [experiment for experiment in experiments if experiment.get("category") == category]
    if platform and platform not in ("All", "All Platforms"):
        experiments = [experiment for experiment in experiments if experiment.get("platform") == platform]

    return {"experiments": experiments[:limit], "count": len(experiments)}


@app.get("/api/experiments/{identifier}")
def get_experiment(identifier: str):
    experiment = _require_experiment(identifier)
    connections = engine.kg.get_experiment_connections(
        experiment.get("name", experiment.get("acronym", identifier))
    )
    related_materials = [
        material for material in engine.materials()
        if material.get("name") in connections.get("materials", [])
    ]
    return {
        "experiment": experiment,
        "connections": connections,
        "related_materials": related_materials,
        "related_findings": _experiment_related_findings(experiment),
        "related_guidelines": _experiment_related_guidelines(experiment),
        "metrics": _metrics_for(experiment),
    }


@app.get("/api/materials")
def list_materials(
    query: Optional[str] = Query(None, max_length=200),
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=100),
):
    materials = engine.materials()
    if query:
        q = query.lower()
        materials = [
            material for material in materials
            if q in " ".join(
                str(material.get(key, ""))
                for key in ("name", "category", "microgravity_behavior", "safety_recommendation")
            ).lower()
        ]
    if category and category not in ("All", "All Categories"):
        materials = [material for material in materials if material.get("category") == category]
    return {"materials": materials[:limit], "count": len(materials)}


@app.get("/api/materials/{name}")
def get_material(name: str, environment: str = Query("ISS")):
    material = get_material_by_name(name)
    if material is None:
        raise HTTPException(status_code=404, detail=f"Material '{name}' not found.")

    ranked = rank_materials(engine.materials(), environment, 100)
    rank_entry = next((item for item in ranked if item["name"] == material.get("name")), None)

    related_experiments = []
    for reference in material.get("experiments_tested_in", []) or []:
        experiment = get_experiment_by_id(str(reference))
        if experiment:
            related_experiments.append(experiment)

    return {
        "material": material,
        "ranking": rank_entry,
        "safety_implications": engine.kg.get_safety_implications(str(material.get("name"))),
        "related_experiments": related_experiments,
    }


@app.get("/api/findings")
def list_findings(
    query: Optional[str] = Query(None, max_length=200),
    category: Optional[str] = Query(None),
    relevance: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=100),
):
    findings = engine.findings()
    if query:
        q = query.lower()
        findings = [
            finding for finding in findings
            if q in " ".join(
                str(finding.get(key, ""))
                for key in ("title", "description", "category", "source_experiment", "quantitative_result")
            ).lower()
        ]
    if category and category not in ("All", "All Categories"):
        findings = [finding for finding in findings if finding.get("category") == category]
    if relevance and relevance not in ("All", "All Levels"):
        findings = [
            finding for finding in findings
            if str(finding.get("safety_relevance", "")).lower() == relevance.lower()
        ]
    return {"findings": findings[:limit], "count": len(findings)}


@app.get("/api/findings/{finding_id}")
def get_finding(finding_id: str):
    finding = next(
        (item for item in engine.findings() if str(item.get("id", "")).lower() == finding_id.lower()),
        None,
    )
    if finding is None:
        raise HTTPException(status_code=404, detail=f"Finding '{finding_id}' not found.")
    return {
        "finding": finding,
        "similar_findings": find_similar_findings(
            finding.get("id", ""), engine.findings(), limit=5
        ),
    }


@app.get("/api/guidelines")
def list_guidelines(
    query: Optional[str] = Query(None, max_length=200),
    environment: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=100),
):
    guidelines = engine.guidelines()
    if query:
        q = query.lower()
        guidelines = [
            guideline for guideline in guidelines
            if q in " ".join(
                str(guideline.get(key, ""))
                for key in ("title", "category", "environment", "recommendation", "rationale")
            ).lower()
        ]
    if environment:
        guidelines = [
            guideline for guideline in guidelines
            if environment.lower() in str(guideline.get("environment", "")).lower()
        ]
    if risk_level:
        guidelines = [
            guideline for guideline in guidelines
            if str(guideline.get("risk_level", "")).lower() == risk_level.lower()
        ]
    return {"guidelines": guidelines[:limit], "count": len(guidelines)}


@app.get("/api/guidelines/{guideline_id}")
def get_guideline(guideline_id: str):
    guideline = next(
        (
            item for item in engine.guidelines()
            if str(item.get("id", "")).lower() == guideline_id.lower()
        ),
        None,
    )
    if guideline is None:
        raise HTTPException(status_code=404, detail=f"Guideline '{guideline_id}' not found.")
    return {"guideline": guideline}


# ── Comparison & Scoring ──────────────────────────────────────────────────────
@app.get("/api/compare")
def compare_experiments(
    e1: str = Query("SAFFIRE-II"),
    e2: str = Query("BASS-II"),
):
    exp1 = _require_experiment(e1)
    exp2 = _require_experiment(e2)

    return {
        "exp1": {
            "name": exp1.get("name", e1),
            "acronym": exp1.get("acronym"),
            "metrics": _metrics_for(exp1),
            "details": exp1,
            "connections": engine.kg.get_experiment_connections(
                exp1.get("name", exp1.get("acronym", e1))
            ),
        },
        "exp2": {
            "name": exp2.get("name", e2),
            "acronym": exp2.get("acronym"),
            "metrics": _metrics_for(exp2),
            "details": exp2,
            "connections": engine.kg.get_experiment_connections(
                exp2.get("name", exp2.get("acronym", e2))
            ),
        },
    }


@app.post("/api/score")
def calculate_score(req: ScoreRequest):
    material = get_material_by_name(req.material)
    resolved = material is not None
    if material is None:
        material = {"name": req.material}

    oxygen = _parse_oxygen_level(req.oxygen_level)
    result = safety_score_for_material(material, req.environment, oxygen)

    base_factors = result["factors"]
    if not resolved:
        base_factors.insert(
            0, f"No exact database record for '{req.material}'; using neutral defaults."
        )
    base_factors.insert(
        0,
        f"{req.environment} alters buoyancy and oxidizer transport relative to 1g",
    )

    recommendation = material.get("safety_recommendation")
    if resolved and recommendation:
        recs = recommendation
    else:
        recs = (
            f"Re-test {req.material} under NASA-STD-6001 Test 1 conditions for the "
            f"{req.environment}. Maintain ventilation control and early smoke detection."
        )

    ranked_table = []
    for candidate in engine.materials():
        iss = safety_score_for_material(candidate, "ISS", oxygen)
        lunar = safety_score_for_material(candidate, "Lunar", oxygen)
        martian = safety_score_for_material(candidate, "Martian", oxygen)
        ranked_table.append({
            "material": candidate.get("name"),
            "iss": round(iss["score"], 1),
            "lunar": round(lunar["score"], 1),
            "martian": round(martian["score"], 1),
            "risk": iss["risk_level"],
        })
    ranked_table.sort(key=lambda row: row["iss"], reverse=True)

    return {
        "material": req.material,
        "material_resolved": resolved,
        "environment": req.environment,
        "oxygen_level": req.oxygen_level,
        "oxygen_percent": oxygen,
        "score": result["score"],
        "risk_level": result["risk_level"],
        "risk_color": color_for_risk(result["risk_level"]),
        "factors": base_factors,
        "recommendations": recs,
        "ranked_table": ranked_table,
        "breakdown": result["components"],
    }


# ── Graph ────────────────────────────────────────────────────────────────────
@app.get("/api/graph")
def get_graph():
    return engine.kg.get_graph_data()


# ── Chat ─────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    chart = maybe_generate_chart(message)
    if chart is not None:
        engine.memory.add_turn(message, chart["answer"], chart["sources"])
        return {
            "answer": chart["answer"],
            "sources": chart["sources"],
            "chart": chart["chart"],
            "kg_context": "",
            "provider": engine.gemini.provider_label,
            "history": engine.memory.get_full_history()[-10:],
        }

    result = engine.rag.query(message, top_k=TOP_K_RESULTS, stream=False)
    raw_answer = result.get("answer") or result.get("response") or "No response generated."

    return {
        "answer": str(raw_answer),
        "sources": result.get("sources", []),
        "kg_context": result.get("kg_context", ""),
        "provider": engine.gemini.provider_label,
        "history": engine.memory.get_full_history()[-10:],
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE token stream for the chat UI: sources -> tokens -> done."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    def event(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def generate():
        try:
            chart = maybe_generate_chart(message)
            if chart is not None:
                engine.memory.add_turn(message, chart["answer"], chart["sources"])
                yield event({
                    "type": "done",
                    "answer": chart["answer"],
                    "sources": chart["sources"],
                    "chart": chart["chart"],
                    "provider": engine.gemini.provider_label,
                })
                return

            result = engine.rag.query(message, top_k=TOP_K_RESULTS, stream=True)
            sources = result.get("sources", [])
            yield event({"type": "sources", "sources": sources})

            answer_iter = result.get("answer") or result.get("response")
            chunks: list[str] = []
            for chunk in answer_iter:
                chunks.append(chunk)
                yield event({"type": "token", "token": chunk})

            yield event({
                "type": "done",
                "answer": "".join(chunks),
                "sources": sources,
                "kg_context": result.get("kg_context", ""),
                "provider": engine.gemini.provider_label,
                "history": engine.memory.get_full_history()[-10:],
            })
        except Exception as exc:
            logger.exception("Streaming chat failed")
            yield event({"type": "error", "error": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/chat/memory")
def clear_memory():
    engine.memory.clear()
    return {"message": "Memory cleared"}


# ── Mount Static Frontend Directory ───────────────────────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/manifest.json")
    def serve_manifest():
        return FileResponse(os.path.join(FRONTEND_DIR, "manifest.json"))

    @app.get("/sw.js")
    def serve_sw():
        return FileResponse(os.path.join(FRONTEND_DIR, "sw.js"))
