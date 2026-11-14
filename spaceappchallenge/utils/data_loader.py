"""
Data loader utilities for loading and querying knowledge base files.
"""

import json
import logging
import os
import re
from typing import Any, Optional

from config import KNOWLEDGE_BASE_DIR

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _load_json_file(filename: str) -> Any:
    """Helper to load a JSON file from the knowledge base directory."""
    path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
    if not os.path.isfile(path):
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error reading %s: %s", path, e)
        return None


def load_experiments() -> list[dict[str, Any]]:
    """Load all experiment records from experiments.json."""
    data = _load_json_file("experiments.json")
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("experiments", [])
    return []


def load_materials() -> list[dict[str, Any]]:
    """Load all material records from materials.json."""
    data = _load_json_file("materials.json")
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("materials", [])
    return []


def load_findings() -> list[dict[str, Any]]:
    """Load all research findings from findings.json."""
    data = _load_json_file("findings.json")
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("findings", [])
    return []


def load_safety_guidelines() -> list[dict[str, Any]]:
    """Load all safety guidelines from safety_guidelines.json."""
    data = _load_json_file("safety_guidelines.json")
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("guidelines", [])
    return []


def load_challenge_brief() -> list[dict[str, Any]]:
    """Load the official NASA Space Apps challenge brief records."""
    data = _load_json_file("challenge_brief.json")
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("challenge_brief", data.get("brief", []))
    return []


def load_all_data() -> dict[str, list[dict[str, Any]]]:
    """Load all knowledge base data into a consolidated dictionary."""
    return {
        "experiments": load_experiments(),
        "materials": load_materials(),
        "findings": load_findings(),
        "guidelines": load_safety_guidelines(),
        "challenge_brief": load_challenge_brief(),
    }


def get_experiment_by_id(exp_id: str) -> Optional[dict[str, Any]]:
    """Find an experiment by id, acronym, or name (case-insensitive).

    Falls back to prefix/substring matching so values like ``BASS`` or
    ``SAFFIRE-I`` resolve to their full record. Ambiguous prefixes such as
    ``SAFFIRE`` prefer the shortest (most general) acronym.
    """
    target = _normalise(exp_id)
    if not target:
        return None

    experiments = load_experiments()

    for exp in experiments:
        if any(
            _normalise(exp.get(key, "")) == target
            for key in ("id", "acronym", "name")
        ):
            return exp

    prefix_matches = [
        exp for exp in experiments
        if any(
            _normalise(exp.get(key, "")).startswith(target)
            for key in ("acronym", "id", "name")
        )
    ]
    if prefix_matches:
        return min(prefix_matches, key=lambda e: len(e.get("acronym", e.get("name", ""))))

    substring_matches = [
        exp for exp in experiments
        if any(
            target in _normalise(exp.get(key, ""))
            for key in ("acronym", "id", "name", "description")
        )
    ]
    if substring_matches:
        return min(substring_matches, key=lambda e: len(e.get("acronym", e.get("name", ""))))

    return None


def get_material_by_name(name: str) -> Optional[dict[str, Any]]:
    """Find a material by name (case-insensitive, tolerant of descriptions)."""
    target = _normalise(name)
    if not target:
        return None

    materials = load_materials()

    for mat in materials:
        mat_name = _normalise(mat.get("name", ""))
        if mat_name == target:
            return mat

    target_base = _normalise(re.sub(r"\(.*?\)", "", target))
    for mat in materials:
        mat_name = _normalise(mat.get("name", ""))
        mat_base = _normalise(re.sub(r"\(.*?\)", "", mat_name))
        if mat_name in target or target in mat_name or (mat_base and mat_base == target_base):
            return mat

    target_tokens = set(re.findall(r"[a-z0-9]+", target))
    best_match = None
    best_overlap = 0
    for mat in materials:
        mat_tokens = set(re.findall(r"[a-z0-9]+", _normalise(mat.get("name", ""))))
        overlap = len(target_tokens & mat_tokens)
        if overlap > best_overlap:
            best_match, best_overlap = mat, overlap
    return best_match if best_overlap > 0 else None
