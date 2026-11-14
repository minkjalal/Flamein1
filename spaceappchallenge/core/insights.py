"""
Explainable ranking and mission-brief analytics for the Flame in Freefall app.

All scoring is deterministic and derived from the ingested NASA datasets so
every ranked item can be traced back to source fields.
"""

import math
import re
from collections import Counter
from typing import Any, Optional

_CURRENT_YEAR = 2026

_RELEVANCE_WEIGHT = {"Critical": 1.0, "High": 0.75, "Medium": 0.5}
_RISK_WEIGHT = {"Critical": 1.0, "High": 0.8, "Medium": 0.6, "Low": 0.35}
_PRIORITY_SCORE = {1: 100.0, 2: 80.0, 3: 60.0}
_SMOKE_WEIGHT = {
    "very high": 1.0,
    "high": 0.8,
    "moderate to high": 0.75,
    "moderate": 0.6,
    "low to moderate": 0.4,
    "low": 0.25,
    "very low": 0.1,
}
_CATEGORY_STRENGTH = {
    "Large-Scale Fire": 1.0,
    "Solid Combustion": 0.85,
    "Droplet Combustion": 0.8,
    "Gaseous Combustion": 0.7,
}
_ENVIRONMENT_RISK_SHIFT = {
    "iss": 0.0,
    "lunar": 4.0,
    "martian": 5.0,
    "deep-space": 2.0,
    "exploration": 8.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _snippet(text: Any, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]+", str(text).lower())
        if len(token) >= 3
    }


def _experiment_end_year(years: Any) -> Optional[int]:
    if not years:
        return None
    text = str(years).lower()
    if "present" in text:
        return _CURRENT_YEAR
    matches = re.findall(r"(?:19|20)\d{2}", text)
    return int(matches[-1]) if matches else None


def _risk_band(score: float) -> str:
    if score >= 70:
        return "CRITICAL"
    if score >= 55:
        return "HIGH"
    if score >= 40:
        return "MODERATE"
    return "LOW"


def _environment_key(environment: Optional[str]) -> str:
    text = (environment or "iss").lower()
    for key in ("exploration", "lunar", "martian", "deep-space", "iss"):
        if key in text:
            return key
    return "iss"


def _match_environment(environment: Optional[str], text: str) -> bool:
    if not environment:
        return False
    env = environment.lower()
    body = text.lower()
    aliases = {
        "lunar": ("lunar", "moon", "0.16g"),
        "martian": ("martian", "mars", "0.38g"),
        "deep-space": ("deep-space", "deep space", "transit", "transit vehicle"),
        "exploration": ("exploration", "34%", "56 kpa", "low-pressure"),
        "iss": ("iss", "international space station", "leo"),
    }
    key = _environment_key(environment)
    if not any(alias in env for alias in aliases[key]):
        return False
    return any(alias in body for alias in aliases[key])


# ── Experiments ───────────────────────────────────────────────────────────────


def rank_experiments(
    experiments: list[dict[str, Any]],
    category: Optional[str] = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Rank experiments by safety-research value, with an explainable breakdown."""
    if category and category not in ("All", "All Categories"):
        experiments = [e for e in experiments if e.get("category") == category]

    ranked: list[dict[str, Any]] = []
    for exp in experiments:
        findings = _as_list(exp.get("key_findings") or exp.get("findings"))
        safety = _as_list(exp.get("safety_implications") or exp.get("safety"))
        materials = _as_list(exp.get("materials_tested") or exp.get("materials"))
        category_weight = _CATEGORY_STRENGTH.get(exp.get("category", ""), 0.6)
        end_year = _experiment_end_year(exp.get("years"))
        recency = 1.0 if end_year is None else _clamp(1.0 - (_CURRENT_YEAR - end_year) * 0.04, 0.3, 1.0)
        platform = 1.0 if exp.get("platform") == "ISS" else 0.85

        components = {
            "safety_findings": _clamp(len(safety) / 4.0) * 0.30,
            "research_findings": _clamp(len(findings) / 6.0) * 0.25,
            "material_breadth": _clamp(len(materials) / 6.0) * 0.15,
            "category_relevance": category_weight * 0.15,
            "recency": recency * 0.10,
            "platform_access": platform * 0.05,
        }
        score = round(sum(components.values()) * 100, 1)

        reasons = [
            f"{len(findings)} documented finding(s)",
            f"{len(safety)} spacecraft safety implication(s)",
            f"{len(materials)} material(s) tested",
        ]
        if exp.get("category"):
            reasons.append(f"category: {exp['category']}")
        if exp.get("years"):
            reasons.append(f"flight window: {exp['years']}")

        ranked.append({
            "id": exp.get("id") or exp.get("acronym") or exp.get("name"),
            "name": exp.get("name") or exp.get("acronym"),
            "acronym": exp.get("acronym"),
            "category": exp.get("category", "General"),
            "platform": exp.get("platform"),
            "years": exp.get("years"),
            "score": score,
            "breakdown": {key: round(val * 100, 1) for key, val in components.items()},
            "findings_count": len(findings),
            "safety_count": len(safety),
            "materials_count": len(materials),
            "reasons": reasons,
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for position, item in enumerate(ranked, 1):
        item["rank"] = position
    return ranked[: max(1, limit)]


# ── Materials ─────────────────────────────────────────────────────────────────


def rank_materials(
    materials: list[dict[str, Any]],
    environment: Optional[str] = "ISS",
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Rank materials by fire risk (higher score = more hazardous)."""
    env_key = _environment_key(environment)
    env_shift = _ENVIRONMENT_RISK_SHIFT.get(env_key, 0.0)

    ranked: list[dict[str, Any]] = []
    for mat in materials:
        flammability = _num(mat.get("flammability_rating"))
        toxicity = _num(mat.get("toxicity_rating"))
        o2_threshold = _num(mat.get("extinction_o2_threshold"))
        smoke = str(mat.get("smoke_production") or "").lower()
        smoke_weight = next(
            (weight for label, weight in _SMOKE_WEIGHT.items() if label in smoke),
            0.5,
        )
        has_microgravity_evidence = bool(_as_list(mat.get("experiments_tested_in")))

        flam_risk = _clamp((flammability or 5.0) / 10.0)
        tox_risk = _clamp((toxicity or 5.0) / 10.0)
        o2_risk = 0.35 if o2_threshold is None else _clamp((30.0 - o2_threshold) / 16.0)

        components = {
            "flammability": flam_risk * 0.45,
            "toxicity": tox_risk * 0.20,
            "oxygen_threshold": o2_risk * 0.20,
            "smoke_output": smoke_weight * 0.15,
        }
        base = sum(components.values()) * 100
        score = round(_clamp(base + env_shift, 0, 100), 1)

        reasons = []
        if flammability is not None:
            reasons.append(f"flammability rating {flammability:g}/10")
        if o2_threshold is not None:
            reasons.append(f"extinction near {o2_threshold:g}% O2")
        if toxicity is not None:
            reasons.append(f"toxicity rating {toxicity:g}/10")
        if smoke:
            reasons.append(f"smoke output: {smoke}")
        if has_microgravity_evidence:
            reasons.append("flight-tested in microgravity")

        ranked.append({
            "name": mat.get("name"),
            "category": mat.get("category"),
            "score": score,
            "risk_level": _risk_band(score),
            "environment": environment or "ISS",
            "flammability_rating": flammability,
            "toxicity_rating": toxicity,
            "extinction_o2_threshold": o2_threshold,
            "smoke_production": mat.get("smoke_production"),
            "breakdown": {key: round(val * 100, 1) for key, val in components.items()},
            "environment_adjustment": env_shift,
            "reasons": reasons,
            "recommendation": mat.get("safety_recommendation"),
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for position, item in enumerate(ranked, 1):
        item["rank"] = position
    return ranked[: max(1, limit)]


# ── Findings ──────────────────────────────────────────────────────────────────


def rank_findings(
    findings: list[dict[str, Any]],
    category: Optional[str] = None,
    relevance: Optional[str] = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Rank research findings by safety relevance and evidence strength."""
    if category and category not in ("All", "All Categories"):
        findings = [f for f in findings if f.get("category") == category]
    if relevance and relevance not in ("All", "All Levels"):
        findings = [
            f for f in findings
            if str(f.get("safety_relevance", "")).lower() == relevance.lower()
        ]

    ranked: list[dict[str, Any]] = []
    for finding in findings:
        rel_weight = _RELEVANCE_WEIGHT.get(str(finding.get("safety_relevance", "")), 0.5)
        score = rel_weight * 70.0
        if finding.get("quantitative_result"):
            score += 15.0
        if finding.get("implications_for_spacecraft"):
            score += 15.0

        reasons = [f"safety relevance: {finding.get('safety_relevance', 'unspecified')}"]
        if finding.get("quantitative_result"):
            reasons.append("quantitative result available")
        if finding.get("implications_for_spacecraft"):
            reasons.append("explicit spacecraft implication")

        ranked.append({
            "id": finding.get("id"),
            "title": finding.get("title"),
            "category": finding.get("category"),
            "source_experiment": finding.get("source_experiment"),
            "safety_relevance": finding.get("safety_relevance"),
            "quantitative_result": finding.get("quantitative_result"),
            "description": _snippet(finding.get("description")),
            "implications_for_spacecraft": finding.get("implications_for_spacecraft"),
            "score": round(score, 1),
            "reasons": reasons,
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for position, item in enumerate(ranked, 1):
        item["rank"] = position
    return ranked[: max(1, limit)]


def find_similar_findings(
    finding_id: str,
    findings: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return findings that share category, source experiment, or vocabulary."""
    target = next((f for f in findings if f.get("id") == finding_id), None)
    if target is None:
        return []

    target_tokens = _tokenize(
        f"{target.get('title', '')} {target.get('description', '')} "
        f"{target.get('category', '')} {target.get('quantitative_result', '')}"
    )
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in findings:
        if candidate is target or candidate.get("id") == target.get("id"):
            continue
        tokens = _tokenize(
            f"{candidate.get('title', '')} {candidate.get('description', '')} "
            f"{candidate.get('category', '')} {candidate.get('quantitative_result', '')}"
        )
        if not target_tokens or not tokens:
            continue
        overlap = len(target_tokens & tokens)
        score = overlap / math.sqrt(len(target_tokens) * len(tokens))
        if candidate.get("category") == target.get("category"):
            score += 0.2
        if candidate.get("source_experiment") == target.get("source_experiment"):
            score += 0.15
        scored.append((score, candidate))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "id": candidate.get("id"),
            "title": candidate.get("title"),
            "category": candidate.get("category"),
            "source_experiment": candidate.get("source_experiment"),
            "similarity": round(score, 3),
        }
        for score, candidate in scored[: max(1, limit)]
    ]


# ── Guidelines ────────────────────────────────────────────────────────────────


def rank_guidelines(
    guidelines: list[dict[str, Any]],
    environment: Optional[str] = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Rank safety guidelines by priority, risk, and environmental applicability."""
    ranked: list[dict[str, Any]] = []
    for guideline in guidelines:
        priority = guideline.get("priority")
        priority_score = _PRIORITY_SCORE.get(priority, 55.0)
        risk_score = _RISK_WEIGHT.get(str(guideline.get("risk_level", "")), 0.6) * 100.0
        related = _as_list(guideline.get("related_experiments"))
        coverage = _clamp(len(related) / 3.0) * 100.0
        env_match = 100.0 if _match_environment(
            environment,
            f"{guideline.get('environment', '')} {guideline.get('rationale', '')}",
        ) else 40.0

        score = (
            priority_score * 0.45
            + risk_score * 0.35
            + coverage * 0.10
            + env_match * 0.10
        )

        reasons = []
        if priority:
            reasons.append(f"priority {priority}")
        if guideline.get("risk_level"):
            reasons.append(f"risk level: {guideline['risk_level']}")
        if related:
            reasons.append(f"linked to {len(related)} experiment(s)")
        if env_match == 100.0:
            reasons.append(f"directly applicable to {environment}")

        ranked.append({
            "id": guideline.get("id"),
            "title": guideline.get("title"),
            "category": guideline.get("category"),
            "environment": guideline.get("environment"),
            "risk_level": guideline.get("risk_level"),
            "priority": priority,
            "recommendation": guideline.get("recommendation"),
            "rationale": _snippet(guideline.get("rationale"), 300),
            "related_experiments": related,
            "score": round(_clamp(score, 0, 100), 1),
            "reasons": reasons,
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for position, item in enumerate(ranked, 1):
        item["rank"] = position
    return ranked[: max(1, limit)]


# ── Unified search ────────────────────────────────────────────────────────────


def _smoke_weight(value: Any) -> float:
    text = str(value or "").lower()
    return next(
        (weight for label, weight in _SMOKE_WEIGHT.items() if label in text),
        0.5,
    )


def _safety_band(score: float) -> str:
    if score >= 75:
        return "LOW"
    if score >= 55:
        return "MODERATE"
    if score >= 35:
        return "HIGH"
    return "CRITICAL"


def safety_score_for_material(
    material: dict[str, Any],
    environment: Optional[str] = "ISS",
    oxygen_level: Optional[float] = None,
) -> dict[str, Any]:
    """Derive a 0–100 safety score (higher = safer) from material data."""
    flammability = _num(material.get("flammability_rating"))
    toxicity = _num(material.get("toxicity_rating"))
    o2_threshold = _num(material.get("extinction_o2_threshold"))
    smoke = _smoke_weight(material.get("smoke_production"))

    flam_safe = 50.0 if flammability is None else (10.0 - flammability) / 9.0 * 100.0
    tox_safe = 50.0 if toxicity is None else (10.0 - toxicity) / 9.0 * 100.0
    o2_safe = 50.0 if o2_threshold is None else _clamp((o2_threshold - 14.0) / 16.0) * 100.0
    smoke_safe = (1.0 - smoke) * 100.0

    components = {
        "flammability_safety": flam_safe * 0.40,
        "toxicity_safety": tox_safe * 0.20,
        "oxygen_threshold_safety": o2_safe * 0.20,
        "smoke_safety": smoke_safe * 0.20,
    }
    base = sum(components.values())

    env_key = _environment_key(environment)
    env_penalty = {"lunar": 3.0, "martian": 5.0, "deep-space": 2.0}.get(env_key, 0.0)

    o2_penalty = 0.0
    if oxygen_level is not None:
        if oxygen_level >= 34:
            o2_penalty = 14.0
        elif oxygen_level >= 30:
            o2_penalty = 9.0
        elif oxygen_level >= 24:
            o2_penalty = 4.0
        elif oxygen_level >= 22:
            o2_penalty = 1.0
        elif oxygen_level < 18:
            o2_penalty = -2.0

    score = round(_clamp(base - env_penalty - o2_penalty, 5.0, 98.0), 1)

    factors = []
    if flammability is not None:
        factors.append(f"Flammability rating {flammability:g}/10 in the NASA material database")
    if o2_threshold is not None:
        factors.append(f"Extinction oxygen threshold is {o2_threshold:g}% O2")
    if toxicity is not None:
        factors.append(f"Combustion toxicity rating {toxicity:g}/10")
    if material.get("smoke_production"):
        factors.append(f"Smoke production: {material['smoke_production']}")
    factors.append(
        f"Environmental penalty for {environment or 'ISS'} is {env_penalty:g} points; "
        f"oxygen adjustment is {o2_penalty:+g} points"
    )

    return {
        "score": score,
        "risk_level": _safety_band(score),
        "components": {key: round(value, 1) for key, value in components.items()},
        "environment_penalty": env_penalty,
        "oxygen_penalty": o2_penalty,
        "factors": factors,
    }


def search_all(
    query: str,
    experiments: list[dict[str, Any]],
    materials: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    guidelines: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Lexical search across every dataset, returning typed, scored results."""
    q = " ".join(str(query).lower().split())
    if not q:
        return []
    query_tokens = _tokenize(q)

    def text_score(text: str, exact_bonus: float = 0.0) -> float:
        body = str(text).lower()
        if not body:
            return 0.0
        score = 0.0
        if q in body:
            score += 1.0 + exact_bonus
        if query_tokens:
            body_tokens = _tokenize(body)
            overlap = len(query_tokens & body_tokens)
            score += overlap / len(query_tokens)
            for token in query_tokens:
                if len(token) >= 4 and token in body:
                    score += 0.15
        return score

    results: list[dict[str, Any]] = []

    for exp in experiments:
        fields = " ".join(str(exp.get(key, "")) for key in (
            "name", "acronym", "category", "platform", "description", "years",
        ))
        fields += " " + " ".join(str(m) for m in _as_list(exp.get("materials_tested")))
        fields += " " + " ".join(str(f) for f in _as_list(exp.get("key_findings")))
        score = text_score(fields, exact_bonus=1.0 if q in str(exp.get("acronym", "")).lower() else 0.0)
        if score > 0:
            results.append({
                "type": "experiment",
                "id": exp.get("id"),
                "title": exp.get("name") or exp.get("acronym"),
                "subtitle": f"{exp.get('category', 'General')} · {exp.get('platform', '')}".strip(" ·"),
                "snippet": _snippet(exp.get("description")),
                "score": round(score, 3),
            })

    for mat in materials:
        fields = " ".join(str(mat.get(key, "")) for key in (
            "name", "category", "microgravity_behavior", "earth_vs_micro",
            "safety_recommendation", "common_uses",
        ))
        score = text_score(fields)
        if score > 0:
            results.append({
                "type": "material",
                "id": mat.get("name"),
                "title": mat.get("name"),
                "subtitle": f"Material · {mat.get('category', 'Uncategorised')}",
                "snippet": _snippet(mat.get("microgravity_behavior")),
                "score": round(score, 3),
            })

    for finding in findings:
        fields = " ".join(str(finding.get(key, "")) for key in (
            "title", "category", "source_experiment", "description",
            "quantitative_result", "implications_for_spacecraft",
        ))
        score = text_score(fields)
        if score > 0:
            results.append({
                "type": "finding",
                "id": finding.get("id"),
                "title": finding.get("title"),
                "subtitle": f"Finding · {finding.get('source_experiment', '')} · "
                            f"{finding.get('safety_relevance', '')}".strip(" ·"),
                "snippet": _snippet(finding.get("description")),
                "score": round(score, 3),
            })

    for guideline in guidelines:
        fields = " ".join(str(guideline.get(key, "")) for key in (
            "id", "title", "category", "environment", "recommendation", "rationale",
        ))
        fields += " " + " ".join(str(e) for e in _as_list(guideline.get("related_experiments")))
        score = text_score(fields)
        if score > 0:
            results.append({
                "type": "guideline",
                "id": guideline.get("id"),
                "title": guideline.get("title"),
                "subtitle": f"Guideline · {guideline.get('category', '')} · "
                            f"{guideline.get('risk_level', '')}".strip(" ·"),
                "snippet": _snippet(guideline.get("recommendation")),
                "score": round(score, 3),
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(1, limit)]


# ── Mission brief ─────────────────────────────────────────────────────────────


def build_summary(
    experiments: list[dict[str, Any]],
    materials: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    guidelines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose the dashboard mission brief with ranked highlights."""
    top_experiments = rank_experiments(experiments, limit=3)
    risky_materials = rank_materials(materials, environment="Exploration (34% O2)", limit=3)
    priority_guidelines = rank_guidelines(guidelines, limit=3)
    critical_findings = rank_findings(findings, relevance="Critical", limit=3)

    categories = Counter(exp.get("category", "General") for exp in experiments)
    tested_materials = [
        mat for mat in materials if _as_list(mat.get("experiments_tested_in"))
    ]
    experiments_with_safety = [
        exp for exp in experiments if _as_list(exp.get("safety_implications"))
    ]

    coverage = {
        "flight_tested_materials": len(tested_materials),
        "total_materials": len(materials),
        "experiments_with_safety_data": len(experiments_with_safety),
        "total_experiments": len(experiments),
        "critical_guidelines": sum(
            1 for g in guidelines if str(g.get("risk_level", "")).lower() == "critical"
        ),
    }

    top_exp = top_experiments[0] if top_experiments else None
    top_mat = risky_materials[0] if risky_materials else None
    top_guideline = priority_guidelines[0] if priority_guidelines else None

    headline_parts = [
        f"{len(experiments)} microgravity combustion experiments",
        f"{len(materials)} spacecraft materials",
        f"{len(findings)} research findings",
        f"{len(guidelines)} fire-safety directives",
    ]
    headline = (
        "NASA's flame-in-freefall corpus contains "
        + ", ".join(headline_parts[:-1]) + f", and {headline_parts[-1]}."
    )

    actions = []
    for guideline in priority_guidelines:
        if guideline.get("recommendation"):
            actions.append({
                "guideline_id": guideline.get("id"),
                "title": guideline.get("title"),
                "action": _snippet(guideline.get("recommendation"), 260),
                "priority": guideline.get("priority"),
            })

    return {
        "headline": headline,
        "counts": {
            "experiments": len(experiments),
            "materials": len(materials),
            "findings": len(findings),
            "guidelines": len(guidelines),
        },
        "categories": dict(categories.most_common()),
        "coverage": coverage,
        "top_experiment": top_exp,
        "highest_risk_material": top_mat,
        "priority_guideline": top_guideline,
        "critical_findings": critical_findings,
        "priority_actions": actions,
        "generated_for": "Exploration atmosphere (34% O2, 56 kPa)",
    }


def build_filters(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """Return the filter vocabulary available to the dashboard."""
    experiments = data.get("experiments", [])
    materials = data.get("materials", [])
    findings = data.get("findings", [])
    guidelines = data.get("guidelines", [])

    def uniq(values: list[Any]) -> list[str]:
        return sorted({str(v) for v in values if v})

    return {
        "experiment_categories": uniq([e.get("category") for e in experiments]),
        "platforms": uniq([e.get("platform") for e in experiments]),
        "material_categories": uniq([m.get("category") for m in materials]),
        "finding_categories": uniq([f.get("category") for f in findings]),
        "finding_relevance": uniq([f.get("safety_relevance") for f in findings]),
        "guideline_environments": uniq([g.get("environment") for g in guidelines]),
        "guideline_risk_levels": uniq([g.get("risk_level") for g in guidelines]),
    }


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
