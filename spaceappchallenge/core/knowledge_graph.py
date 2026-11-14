"""
NetworkX-based knowledge graph for microgravity combustion experiments.
Provides structured relationship querying to augment the RAG pipeline.
"""

import logging
import re
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)

_MAX_LABEL = 140

# Tokens that carry no identity when matching material names.
_MATERIAL_STOPWORDS = {
    "cloth", "fabric", "blend", "film", "sheet", "rubber", "composite",
    "insulation", "foam", "wire", "cross", "linked", "plexiglass", "acrylic",
}


class KnowledgeGraph:
    """In-memory knowledge graph of experiment entities and their relationships."""

    # Canonical node types
    NODE_TYPES = {
        "experiment", "material", "condition", "finding",
        "safety_implication", "guideline",
    }

    # Canonical edge (relationship) types
    EDGE_TYPES = {
        "tests_material",
        "uses_condition",
        "produces_finding",
        "implies_safety",
        "has_guideline",
    }

    def __init__(self):
        self.graph = nx.DiGraph()
        logger.info("KnowledgeGraph initialised (empty).")

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_data(
        self,
        experiments_data: list[dict[str, Any]],
        materials_data: Optional[list[dict[str, Any]]] = None,
        findings_data: Optional[list[dict[str, Any]]] = None,
        guidelines_data: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Populate the graph from the experiment, material, finding, and
        guideline datasets.

        Experiment records use ``materials_tested``, ``parameters``,
        ``key_findings``, and ``safety_implications`` — the historical
        ``materials`` / ``conditions`` / ``findings`` keys are still accepted
        as fallbacks.
        """
        self.graph.clear()

        materials_data = materials_data or []
        findings_data = findings_data or []
        guidelines_data = guidelines_data or []

        aliases, canonical_tokens = _build_material_aliases(materials_data)
        material_nodes: set[str] = set()

        for record in experiments_data:
            exp_name = record.get("name") or record.get("acronym") or record.get("id")
            if not exp_name:
                continue
            exp_name = str(exp_name)
            self._add_node(exp_name, "experiment", record)

            # Materials tested
            exp_materials: list[str] = []
            for mat in _as_list(record.get("materials_tested") or record.get("materials")):
                raw_name = _get_name(mat)
                mat_name = _resolve_material_name(raw_name, aliases, canonical_tokens)
                if mat_name:
                    self._add_node(mat_name, "material", mat if isinstance(mat, dict) else {})
                    self.graph.add_edge(exp_name, mat_name, relationship="tests_material")
                    exp_materials.append(mat_name)
                    material_nodes.add(mat_name)

            # Test conditions (parameter ranges / values)
            params = record.get("parameters")
            if isinstance(params, dict):
                for param_name, param_value in params.items():
                    cond = _format_condition(param_name, param_value)
                    if cond:
                        self._add_node(cond, "condition", {"parameter": param_name})
                        self.graph.add_edge(exp_name, cond, relationship="uses_condition")
            for cond in _as_list(record.get("conditions")):
                cond_name = _get_name(cond) or _flatten_scalar(cond)
                if cond_name:
                    self._add_node(cond_name, "condition", cond if isinstance(cond, dict) else {})
                    self.graph.add_edge(exp_name, cond_name, relationship="uses_condition")

            # Key findings
            for finding in _as_list(record.get("key_findings") or record.get("findings")):
                finding_text = _get_description(finding) or _flatten_scalar(finding)
                if finding_text:
                    short = _short(finding_text)
                    self._add_node(short, "finding", finding if isinstance(finding, dict) else {})
                    self.graph.add_edge(exp_name, short, relationship="produces_finding")

            # Safety implications
            for safety in _as_list(
                record.get("safety_implications") or record.get("safety")
            ):
                safety_text = _get_description(safety) or _flatten_scalar(safety)
                if not safety_text:
                    continue
                short = _short(safety_text)
                self._add_node(
                    short, "safety_implication",
                    safety if isinstance(safety, dict) else {},
                )
                self.graph.add_edge(exp_name, short, relationship="implies_safety")
                for mat_name in exp_materials:
                    self.graph.add_edge(mat_name, short, relationship="implies_safety")

        # Material records: safety guidance and experiment back-links
        for mat in materials_data:
            raw_name = _get_name(mat)
            if not raw_name:
                continue
            mat_name = _resolve_material_name(raw_name, aliases, canonical_tokens) or raw_name
            self._add_node(mat_name, "material", mat)
            material_nodes.add(mat_name)

            recommendation = mat.get("safety_recommendation")
            if recommendation:
                short = _short(str(recommendation))
                self._add_node(short, "safety_implication", {"material": mat_name})
                self.graph.add_edge(mat_name, short, relationship="implies_safety")

            for exp_ref in _as_list(mat.get("experiments_tested_in")):
                exp_node = self._fuzzy_find(str(exp_ref))
                if exp_node and self.graph.nodes[exp_node].get("node_type") == "experiment":
                    self.graph.add_edge(exp_node, mat_name, relationship="tests_material")

        # Standalone findings dataset
        for finding in findings_data:
            title = finding.get("title") or _get_description(finding)
            if not title:
                continue
            title = _short(str(title))
            self._add_node(title, "finding", finding)

            exp_ref = finding.get("source_experiment")
            exp_node = self._fuzzy_find(str(exp_ref)) if exp_ref else None
            if exp_node and self.graph.nodes[exp_node].get("node_type") == "experiment":
                self.graph.add_edge(exp_node, title, relationship="produces_finding")

            conditions = finding.get("conditions")
            if isinstance(conditions, dict):
                for key, value in conditions.items():
                    cond = _format_condition(key, value)
                    if cond:
                        self._add_node(cond, "condition", {"parameter": key})
                        if exp_node:
                            self.graph.add_edge(exp_node, cond, relationship="uses_condition")

            implication = finding.get("implications_for_spacecraft")
            if implication:
                short = _short(str(implication))
                self._add_node(short, "safety_implication", {"source_finding": title})
                origin = exp_node or title
                self.graph.add_edge(origin, short, relationship="implies_safety")

        # Safety guidelines dataset
        for guideline in guidelines_data:
            title = guideline.get("title") or guideline.get("id")
            if not title:
                continue
            title = _short(str(title))
            self._add_node(title, "guideline", guideline)
            for exp_ref in _as_list(guideline.get("related_experiments")):
                exp_node = self._fuzzy_find(str(exp_ref))
                if exp_node and self.graph.nodes[exp_node].get("node_type") == "experiment":
                    self.graph.add_edge(exp_node, title, relationship="has_guideline")

        logger.info(
            "KnowledgeGraph built — %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_related(self, node_name: str, depth: int = 2) -> Optional[nx.DiGraph]:
        """Return a subgraph of nodes within *depth* hops of *node_name*.

        Uses undirected traversal (ignores edge direction) to find all
        neighbours, then returns the directed subgraph induced by those nodes.
        """
        if node_name not in self.graph:
            # Try case-insensitive fuzzy match
            node_name = self._fuzzy_find(node_name)
            if node_name is None:
                return None

        undirected = self.graph.to_undirected()
        related_nodes = set(
            nx.single_source_shortest_path_length(undirected, node_name, cutoff=depth).keys()
        )
        return self.graph.subgraph(related_nodes).copy()

    def get_experiment_connections(self, experiment_name: str) -> dict[str, list[str]]:
        """Return all directly-connected entities for an experiment, grouped by type."""
        connections: dict[str, list[str]] = {
            "materials": [],
            "conditions": [],
            "findings": [],
            "safety_implications": [],
            "guidelines": [],
        }

        resolved = experiment_name
        if experiment_name not in self.graph:
            resolved = self._fuzzy_find(experiment_name)
            if resolved is None:
                return connections

        for _, target, data in self.graph.edges(resolved, data=True):
            rel = data.get("relationship", "")
            node_type = self.graph.nodes[target].get("node_type", "")
            if rel == "tests_material" or node_type == "material":
                connections["materials"].append(target)
            elif rel == "uses_condition" or node_type == "condition":
                connections["conditions"].append(target)
            elif rel == "produces_finding" or node_type == "finding":
                connections["findings"].append(target)
            elif rel == "implies_safety" or node_type == "safety_implication":
                connections["safety_implications"].append(target)
            elif rel == "has_guideline" or node_type == "guideline":
                connections["guidelines"].append(target)

        return connections

    def get_safety_implications(self, material: str) -> list[str]:
        """Return safety-implication nodes linked to a given material."""
        resolved = material
        if material not in self.graph:
            resolved = self._fuzzy_find(material)
            if resolved is None:
                return []

        implications: list[str] = []
        for _, target, data in self.graph.edges(resolved, data=True):
            if data.get("relationship") == "implies_safety":
                implications.append(target)

        # Also check inbound edges (in case an experiment links safety → material)
        for source, _, data in self.graph.in_edges(resolved, data=True):
            if self.graph.nodes[source].get("node_type") == "safety_implication":
                implications.append(source)

        return list(dict.fromkeys(implications))

    # ── Visualisation / Export ────────────────────────────────────────────────

    def get_graph_data(self) -> dict[str, list[dict[str, Any]]]:
        """Return nodes and edges in a JSON-serialisable format.

        The canonical node key must win over any record attributes (records also
        carry an "id" field), so edges stay resolvable on the frontend.
        """
        nodes = [
            {**attrs, "id": n}
            for n, attrs in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, **{k: val for k, val in d.items()}}
            for u, v, d in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def to_context_string(self, subgraph: Optional[nx.DiGraph] = None) -> str:
        """Convert a subgraph (or the full graph) into a natural-language string
        suitable for injecting into a RAG prompt."""
        g = subgraph if subgraph is not None else self.graph

        if g.number_of_nodes() == 0:
            return ""

        lines: list[str] = []

        # Group nodes by type
        by_type: dict[str, list[str]] = {}
        for node, attrs in g.nodes(data=True):
            ntype = attrs.get("node_type", "other")
            by_type.setdefault(ntype, []).append(node)

        for ntype in [
            "experiment", "material", "condition", "finding",
            "safety_implication", "guideline",
        ]:
            nodes = by_type.get(ntype, [])
            if not nodes:
                continue
            label = ntype.replace("_", " ").title() + "s"
            lines.append(f"## {label}")
            for n in nodes:
                lines.append(f"- {n}")

        lines.append("")
        lines.append("## Relationships")
        for u, v, d in g.edges(data=True):
            rel = d.get("relationship", "related_to").replace("_", " ")
            lines.append(f"- {u} → ({rel}) → {v}")

        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _add_node(self, name: str, node_type: str, attrs: Any = None):
        """Add or update a node with a type label and optional extra attributes."""
        extra = {}
        if isinstance(attrs, dict):
            extra = {k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool))}
        self.graph.add_node(str(name), node_type=node_type, **extra)

    def _fuzzy_find(self, name: str) -> Optional[str]:
        """Case-insensitive lookup, preferring exact then shortest prefix matches."""
        lower = str(name).strip().lower()
        if not lower:
            return None

        nodes = list(self.graph.nodes)
        for node in nodes:
            if node.lower() == lower:
                return node

        prefix_matches = [node for node in nodes if node.lower().startswith(lower)]
        if prefix_matches:
            return min(prefix_matches, key=len)

        substring_matches = [node for node in nodes if lower in node.lower()]
        if substring_matches:
            return min(substring_matches, key=len)

        return None


# ── Module-level helpers ──────────────────────────────────────────────────────


def _as_list(value: Any) -> list:
    """Normalise a value to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _get_name(item: Any) -> Optional[str]:
    """Extract a name string from a dict-or-string item."""
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        return (item.get("name") or item.get("material") or item.get("label") or "").strip() or None
    return None


def _get_description(item: Any) -> Optional[str]:
    """Extract a description string from a dict-or-string item."""
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        return (
            item.get("description") or item.get("text") or item.get("finding") or item.get("implication") or ""
        ).strip() or None
    return None


def _flatten_scalar(value: Any) -> Optional[str]:
    """Best-effort string conversion for scalar-ish values."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _short(text: str, limit: int = _MAX_LABEL) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _normalize(name: str) -> str:
    return " ".join(str(name).lower().split())


def _material_tokens(name: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize(name))
        if token not in _MATERIAL_STOPWORDS and len(token) >= 3
    }
    return tokens


def _build_material_aliases(
    materials: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Map normalized aliases to canonical material node names."""
    aliases: dict[str, str] = {}
    canonical_tokens: dict[str, set[str]] = {}

    for mat in materials:
        name = _get_name(mat)
        if not name:
            continue
        canonical = name.strip()
        aliases[_normalize(canonical)] = canonical

        base = re.sub(r"\(.*?\)", "", canonical)
        base_norm = _normalize(base)
        if base_norm:
            aliases[base_norm] = canonical

        canonical_tokens[canonical] = _material_tokens(canonical)

    return aliases, canonical_tokens


def _resolve_material_name(
    raw_name: Optional[str],
    aliases: dict[str, str],
    canonical_tokens: dict[str, set[str]],
) -> Optional[str]:
    if not raw_name:
        return None
    norm = _normalize(raw_name)
    if norm in aliases:
        return aliases[norm]

    base = re.sub(r"\(.*?\)", "", raw_name)
    base_norm = _normalize(base)
    if base_norm in aliases:
        return aliases[base_norm]

    tokens = _material_tokens(raw_name)
    if tokens:
        best_name: Optional[str] = None
        best_score = 0.0
        for canonical, canon_tokens in canonical_tokens.items():
            shared = tokens & canon_tokens
            if not shared:
                continue
            score = len(shared) / max(1, min(len(tokens), len(canon_tokens)))
            if score > best_score:
                best_name, best_score = canonical, score
        if best_name and best_score >= 0.5:
            return best_name

    return raw_name.strip() or None


def _format_condition(name: str, value: Any) -> Optional[str]:
    """Render a parameter/condition key-value pair into a readable node label."""
    label = str(name).replace("_", " ").strip().title()
    if not label:
        return None

    detail = ""
    if isinstance(value, dict):
        if "min" in value and "max" in value:
            detail = f"{value['min']}–{value['max']} {value.get('unit', '')}".strip()
        elif "value" in value:
            detail = f"{value['value']} {value.get('unit', '')}".strip()
        else:
            detail = ", ".join(f"{k}: {v}" for k, v in value.items())
    elif isinstance(value, (str, int, float)):
        detail = str(value)

    return f"{label} ({detail})" if detail else label
