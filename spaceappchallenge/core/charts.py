"""
On-demand chart generation for the Flame in Freefall chat.

Resolves an experiment mentioned in a user message (including misspellings
like "saphire 1" -> SAFFIRE-I) or a curated topic such as spherical flames,
cool flames, flame spread, or material flammability, then renders a grounded
matplotlib figure as a base64 data URL for inline display.
"""

import base64
import difflib
import io
import logging
import re
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure

from utils.data_loader import load_experiments, load_findings, load_materials

logger = logging.getLogger(__name__)

_CHART_INTENT_RE = re.compile(
    r"\b(graph|graphs|plot|plots|chart|charts|visuali[sz]e|visuali[sz]ation|diagram)\b",
    re.IGNORECASE,
)
_ARABIC_TO_ROMAN = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v"}
_ROMAN_TO_ARABIC = {v: k for k, v in _ARABIC_TO_ROMAN.items()}
_MICRO_COLOR = "#FF5E3A"
_EARTH_COLOR = "#3B82F6"
_TEXT_COLOR = "#111827"
_MUTED_COLOR = "#6B7280"
_WARN_COLOR = "#F59E0B"

_TOPIC_PATTERNS = [
    ("spherical", re.compile(
        r"\b(spheric\w*|sphere|quench\w*|flame (?:shape|geometry)|spherical flames?)\b",
        re.IGNORECASE,
    )),
    ("cool_flame", re.compile(
        r"\b(cool[ -]?flames?|low[- ]temperature combustion)\b", re.IGNORECASE
    )),
    ("flame_spread", re.compile(
        r"\b(flame spread|spread rates?)\b", re.IGNORECASE
    )),
    ("materials", re.compile(
        r"\b(material flammability|flammability ratings?|flammable materials|material rankings?)\b",
        re.IGNORECASE,
    )),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _aliases(exp: dict[str, Any]) -> set[str]:
    acronym = exp.get("acronym") or ""
    aliases = {acronym, exp.get("name") or "", exp.get("id") or ""}

    parts = _norm(acronym).split()
    if len(parts) >= 2 and parts[-1] in _ROMAN_TO_ARABIC:
        base = " ".join(parts[:-1])
        arabic = _ROMAN_TO_ARABIC[parts[-1]]
        aliases.update({f"{base} {arabic}", f"{base} {parts[-1]}", f"{base}-{parts[-1]}", f"{base}{arabic}"})
    return {a for a in aliases if a}


def resolve_experiment(message: str) -> Optional[dict[str, Any]]:
    """Fuzzy-match an experiment from a free-text message."""
    experiments = load_experiments()
    query = _norm(message)
    if not query:
        return None

    padded = f" {query} "
    compact_query = query.replace(" ", "")
    tokens = query.split()
    best: Optional[dict[str, Any]] = None
    best_score = 0.0

    for exp in experiments:
        score = 0.0
        for alias in _aliases(exp):
            alias_norm = _norm(alias)
            if not alias_norm:
                continue
            if alias_norm.replace(" ", "") in compact_query:
                score = max(score, 0.95 if compact_query != alias_norm.replace(" ", "") else 1.0)
            if f" {alias_norm} " in padded:
                score = max(score, 1.0)

        parts = _norm(exp.get("acronym") or "").split()
        if parts:
            base = parts[0]
            numeral = parts[1] if len(parts) > 1 else ""
            for idx, token in enumerate(tokens):
                if len(token) < 4:
                    continue
                ratio = difflib.SequenceMatcher(None, token, base).ratio()
                if ratio < 0.70:
                    continue
                candidate = ratio * 0.9
                if numeral:
                    next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
                    roman = _ARABIC_TO_ROMAN.get(next_token, next_token)
                    candidate += 0.1 if roman == numeral else -0.35
                score = max(score, candidate)

        if score > best_score:
            best_score, best = score, exp

    if best is None or best_score < 0.72:
        return None
    return best


def wants_chart(message: str) -> bool:
    return bool(_CHART_INTENT_RE.search(message))


def maybe_generate_chart(message: str) -> Optional[dict[str, Any]]:
    """Return a chart payload for chart-intent messages, else None."""
    if not wants_chart(message):
        return None

    exp = resolve_experiment(message)
    if exp is not None:
        try:
            data_url = _render_chart(exp)
        except Exception as e:
            logger.warning("Chart generation failed for %s: %s", exp.get("acronym"), e)
        else:
            name = exp.get("name") or exp.get("acronym") or "Experiment"
            return {
                "answer": _grounded_summary(exp),
                "sources": [f"{name} (Experiments)"],
                "chart": data_url,
            }

    topic = match_topic(message)
    if topic is not None:
        try:
            return _build_topic_payload(topic)
        except Exception as e:
            logger.warning("Topic chart generation failed for %s: %s", topic, e)

    return None


def match_topic(message: str) -> Optional[str]:
    """Match a chart request to a curated knowledge-base topic."""
    for name, pattern in _TOPIC_PATTERNS:
        if pattern.search(message):
            return name
    return None


# ── Thematic charts ───────────────────────────────────────────────────────────

def _find_finding(fragment: str) -> Optional[dict[str, Any]]:
    for finding in load_findings():
        if fragment.lower() in (finding.get("title") or "").lower():
            return finding
    return None


def _fig_to_data_url(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_topic_payload(topic: str) -> dict[str, Any]:
    builders = {
        "spherical": _spherical_payload,
        "cool_flame": _cool_flame_payload,
        "flame_spread": _flame_spread_payload,
        "materials": _materials_payload,
    }
    builder = builders.get(topic)
    if builder is None:
        raise ValueError(f"unknown chart topic: {topic}")
    return builder()


def _source_label(finding: Optional[dict[str, Any]], fallback: str) -> str:
    title = (finding or {}).get("title") or fallback
    return f"{title} (Findings)"


def _spherical_payload() -> dict[str, Any]:
    quench = _find_finding("Low-Speed Quenching Distance")
    standoff_finding = _find_finding("Burning Rate Emulation")
    sphericity_finding = _find_finding("Quiescent Sphericity")

    quench_text = (quench or {}).get("quantitative_result", "")
    match = re.search(
        r"from\s+([\d.]+)\s*mm\s*\(1g\)\s+to\s+([\d.]+)\s*mm\s*\(0g\)",
        quench_text,
        re.IGNORECASE,
    )
    earth_q, micro_q = (float(match.group(1)), float(match.group(2))) if match else (1.8, 5.2)

    standoff_pct = 300.0
    if standoff_finding:
        m = re.search(
            r"increased by\s+([\d.]+)\s*%",
            standoff_finding.get("quantitative_result", ""),
            re.IGNORECASE,
        )
        if m:
            standoff_pct = float(m.group(1))

    sphericity = 0.98
    if sphericity_finding:
        m = re.search(
            r"sphericity index\s*>\s*([\d.]+)",
            sphericity_finding.get("quantitative_result", ""),
            re.IGNORECASE,
        )
        if m:
            sphericity = float(m.group(1))

    fig = Figure(figsize=(9.2, 4.8), dpi=140, facecolor="#ffffff")
    ax1, ax2 = fig.subplots(1, 2)

    _style_axes(ax1)
    labels = ["Earth (1g)", "Microgravity (0g)"]
    values = [earth_q, micro_q]
    colors = [_EARTH_COLOR, _MICRO_COLOR]
    y_positions = [1, 0]
    ax1.barh(y_positions, values, color=colors, height=0.5)
    for y, value in zip(y_positions, values):
        ax1.annotate(
            f"{value:g} mm", xy=(value, y), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=10,
            fontweight="bold", color=_TEXT_COLOR,
        )
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels, fontsize=10, color=_TEXT_COLOR)
    ax1.set_xlim(0, max(values) * 1.35)
    ax1.set_xlabel("Quenching distance (mm)", fontsize=10, color=_MUTED_COLOR)
    ax1.set_title(
        f"Quenching gap expands {micro_q / earth_q:.1f}×",
        fontsize=11, fontweight="bold", color=_TEXT_COLOR,
    )

    _style_axes(ax2)
    relative = [100.0, 100.0 + standoff_pct]
    ax2.bar([0, 1], relative, color=colors, width=0.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Earth (1g)", "Microgravity (0g)"], fontsize=10, color=_TEXT_COLOR)
    ax2.set_ylim(0, max(relative) * 1.25)
    ax2.set_ylabel("Flame standoff distance (% of 1g)", fontsize=10, color=_MUTED_COLOR)
    ax2.annotate("baseline", xy=(0, relative[0]), xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=10, fontweight="bold", color=_TEXT_COLOR)
    ax2.annotate(f"+{standoff_pct:g}%", xy=(1, relative[1]), xytext=(0, 8),
                 textcoords="offset points", ha="center", fontsize=10,
                 fontweight="bold", color=_WARN_COLOR)
    ax2.set_title("Flame stands farther from surfaces", fontsize=11,
                  fontweight="bold", color=_TEXT_COLOR)
    ax2.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Spherical flames in microgravity — NASA findings", x=0.04, y=0.97,
                 ha="left", fontsize=14, fontweight="bold", color=_TEXT_COLOR)
    fig.text(
        0.04, 0.89,
        f"Quenching (ACME-s-Flame) · Standoff (ACME-BRE) · Droplet sphericity index > {sphericity:g} (FLEX)",
        fontsize=9, color=_MUTED_COLOR, ha="left",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))

    answer = (
        "Spherical flames in microgravity (grounded in NASA findings):\n\n"
        f"- **Quenching distance** expands from {earth_q:g} mm at 1g to {micro_q:g} mm at 0g "
        f"(~{micro_q / earth_q:.1f}×), so flame arrestors and mesh screens sized for Earth "
        "may allow microgravity flames to penetrate.\n"
        f"- **Flame standoff distance** from burner surfaces increases by about {standoff_pct:g}% "
        "versus 1g boundary layers (ACME-BRE), changing heat transfer back to the fuel.\n"
        f"- **Droplet flame sphericity** exceeds {sphericity:g} in quiescent microgravity, with peak "
        "gas temperatures 150–250 K lower than buoyant flames (FLEX).\n\n"
        "Chart generated from local NASA knowledge-base findings."
    )
    return {
        "answer": answer,
        "sources": [
            _source_label(quench, "Low-Speed Quenching Distance Expansion"),
            _source_label(standoff_finding, "ACME-BRE Boundary Layer Attachment"),
            _source_label(sphericity_finding, "Quiescent Sphericity in Microgravity"),
        ],
        "chart": _fig_to_data_url(fig),
    }


def _cool_flame_payload() -> dict[str, Any]:
    finding = _find_finding("Cool Flame Burning")
    text = (finding or {}).get("quantitative_result", "")
    temp_match = re.search(r"between\s+([\d.]+)\s*K\s*-\s*([\d.]+)\s*K", text, re.IGNORECASE)
    temp_lo, temp_hi = (float(temp_match.group(1)), float(temp_match.group(2))) if temp_match else (600.0, 800.0)
    duration_match = re.search(r"up to\s+([\d.]+)\s*seconds", text, re.IGNORECASE)
    duration = float(duration_match.group(1)) if duration_match else 40.0

    fig = Figure(figsize=(9.2, 4.8), dpi=140, facecolor="#ffffff")
    ax1, ax2 = fig.subplots(1, 2)

    _style_axes(ax1)
    ax1.barh([0], [temp_hi - temp_lo], left=temp_lo, color=_MICRO_COLOR, height=0.45)
    ax1.annotate(f"{temp_lo:g}–{temp_hi:g} K", xy=((temp_lo + temp_hi) / 2, 0), xytext=(0, 12),
                 textcoords="offset points", ha="center", fontsize=11,
                 fontweight="bold", color=_TEXT_COLOR)
    ax1.set_yticks([0])
    ax1.set_yticklabels(["Cool flame"], fontsize=10, color=_TEXT_COLOR)
    ax1.set_xlim(0, 1000)
    ax1.set_xlabel("Temperature (K)", fontsize=10, color=_MUTED_COLOR)
    ax1.set_title("Low-temperature oxidation window", fontsize=11,
                  fontweight="bold", color=_TEXT_COLOR)

    _style_axes(ax2)
    ax2.bar([0], [duration], color=_WARN_COLOR, width=0.5)
    ax2.annotate(f"up to {duration:g} s", xy=(0, duration), xytext=(0, 8),
                 textcoords="offset points", ha="center", fontsize=11,
                 fontweight="bold", color=_TEXT_COLOR)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Post-extinction"], fontsize=10, color=_TEXT_COLOR)
    ax2.set_ylim(0, duration * 1.35)
    ax2.set_ylabel("Invisible reaction duration (s)", fontsize=10, color=_MUTED_COLOR)
    ax2.set_title("Cool flames persist after visible extinction", fontsize=11,
                  fontweight="bold", color=_TEXT_COLOR)
    ax2.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Cool flames in microgravity — NASA findings", x=0.04, y=0.97,
                 ha="left", fontsize=14, fontweight="bold", color=_TEXT_COLOR)
    fig.text(0.04, 0.89, "FLEX-2 droplet combustion · low-temperature hydrocarbon oxidation",
             fontsize=9, color=_MUTED_COLOR, ha="left")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))

    answer = (
        "Cool flames in microgravity (grounded in NASA findings):\n\n"
        f"- After the visible hot flame extinguishes, droplet burning can continue in a "
        f"non-luminous cool-flame regime at roughly {temp_lo:g}–{temp_hi:g} K.\n"
        f"- The invisible reaction can persist for up to {duration:g} seconds post-extinction "
        "(FLEX-2), so a fire may still be chemically active after it appears out.\n"
        "- This low-temperature mode is driven by hydrocarbon oxidation pathways below "
        "classical flammability limits.\n\n"
        "Chart generated from local NASA knowledge-base findings."
    )
    return {
        "answer": answer,
        "sources": [
            _source_label(finding, "Cool Flame Burning Below Flammability Limits"),
            "FLEX-2 (Experiments)",
        ],
        "chart": _fig_to_data_url(fig),
    }


def _flame_spread_payload() -> dict[str, Any]:
    finding = _find_finding("Concurrent Solid Fuel Flame Spread")
    text = (finding or {}).get("quantitative_result", "")
    match = re.search(
        r"between\s+([\d.]+)\s*mm/s\s+and\s+([\d.]+)\s*mm/s\s+vs\s+([\d.]+)\s*mm/s",
        text,
        re.IGNORECASE,
    )
    micro_lo, micro_hi, earth = (
        (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        if match else (0.3, 1.8, 2.5)
    )

    fig = Figure(figsize=(8.6, 4.4), dpi=140, facecolor="#ffffff")
    ax = fig.add_subplot(111)
    _style_axes(ax)
    rows = [
        ("Microgravity (0g)", micro_lo, micro_hi, _MICRO_COLOR),
        ("Earth gravity (1g)", earth, earth, _EARTH_COLOR),
    ]
    for row, (_label, lo, hi, color) in enumerate(rows):
        ax.plot([lo, hi], [row, row], color=color, linewidth=12,
                solid_capstyle="round", alpha=0.9)
        ax.plot([(lo + hi) / 2], [row], marker="o", color="#ffffff",
                markeredgecolor=color, markeredgewidth=2, markersize=8, zorder=3)
        label = f"{lo:g}–{hi:g} mm/s" if lo != hi else f"{lo:g} mm/s"
        ax.annotate(label, xy=((lo + hi) / 2, row), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", color=_TEXT_COLOR)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=_TEXT_COLOR)
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(0, max(micro_hi, earth) * 1.3)
    ax.set_xlabel("Flame spread rate (mm/s)", fontsize=10, color=_MUTED_COLOR)

    fig.suptitle("Concurrent flame spread — microgravity vs Earth", x=0.05, y=0.96,
                 ha="left", fontsize=14, fontweight="bold", color=_TEXT_COLOR)
    fig.text(0.05, 0.88, "Solid fuels · BASS findings · 30–60% reduction in microgravity",
             fontsize=9, color=_MUTED_COLOR, ha="left")
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.85))

    micro_mid = (micro_lo + micro_hi) / 2
    reduction = (1 - micro_mid / earth) * 100 if earth else 0
    answer = (
        "Concurrent flame spread in microgravity (grounded in NASA findings):\n\n"
        f"- Thin solid fuels spread at {micro_lo:g}–{micro_hi:g} mm/s in microgravity versus "
        f"about {earth:g} mm/s in 1g upward spread.\n"
        f"- Range midpoints are roughly {reduction:.0f}% lower in microgravity, consistent with "
        "the BASS 30–60% reduction.\n"
        "- Slower spread extends crew egress time but can hide internal thermal accumulation "
        "in confined spaces.\n\n"
        "Chart generated from local NASA knowledge-base findings."
    )
    return {
        "answer": answer,
        "sources": [_source_label(finding, "Concurrent Solid Fuel Flame Spread Reduction")],
        "chart": _fig_to_data_url(fig),
    }


def _materials_payload() -> dict[str, Any]:
    materials = load_materials()
    if not materials:
        raise ValueError("no material records available")
    ranked = sorted(
        materials,
        key=lambda m: m.get("flammability_rating", 0),
        reverse=True,
    )[:10]

    names = [
        re.sub(r"\s*\(.*?\)\s*", "", m.get("name") or "Material")[:26]
        for m in ranked
    ]
    ratings = [float(m.get("flammability_rating", 0)) for m in ranked]
    colors = [
        "#EF4444" if r >= 8 else "#F59E0B" if r >= 5 else "#22C55E"
        for r in ratings
    ]

    fig = Figure(figsize=(8.8, 5.2), dpi=140, facecolor="#ffffff")
    ax = fig.add_subplot(111)
    _style_axes(ax)
    y_positions = list(range(len(ranked)))[::-1]
    ax.barh(y_positions, ratings, color=colors, height=0.6)
    for y, rating in zip(y_positions, ratings):
        ax.annotate(f"{rating:g}", xy=(rating, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=10,
                    fontweight="bold", color=_TEXT_COLOR)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names, fontsize=9, color=_TEXT_COLOR)
    ax.set_xlim(0, 11)
    ax.set_xlabel("Flammability rating (1 = resistant, 10 = highly flammable)",
                  fontsize=10, color=_MUTED_COLOR)

    fig.suptitle("Spacecraft materials ranked by flammability", x=0.05, y=0.96,
                 ha="left", fontsize=14, fontweight="bold", color=_TEXT_COLOR)
    fig.text(0.05, 0.89, "Top 10 materials in the local NASA material database",
             fontsize=9, color=_MUTED_COLOR, ha="left")
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.85))

    top = ", ".join(f"{n} ({r:g})" for n, r in zip(names[:3], ratings[:3]))
    answer = (
        "Spacecraft material flammability (grounded in the local NASA materials database):\n\n"
        f"- Highest flammability ratings: {top}.\n"
        "- Ratings run from 1 (fire-resistant) to 10 (highly flammable); microgravity can "
        "reverse 1g behavior when melt drips no longer remove fuel from the flame zone.\n"
        "- Use this ranking for material selection, then verify against experiment-specific "
        "limiting-oxygen-index data.\n\n"
        "Chart generated from local NASA knowledge-base materials."
    )
    return {
        "answer": answer,
        "sources": ["Materials Flammability Database (Materials)"],
        "chart": _fig_to_data_url(fig),
    }


# ── Data extraction ───────────────────────────────────────────────────────────

def _parse_range(value: Any) -> Optional[tuple[float, float]]:
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)", value)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    if match:
        return (float(match.group(1)), float(match.group(1)))
    return None


def _param_value(param: Any) -> Optional[tuple[float, float, str]]:
    if not isinstance(param, dict):
        return None
    unit = str(param.get("unit") or "")
    if "value" in param and isinstance(param["value"], (int, float)):
        v = float(param["value"])
        return (v, v, unit)
    if "min" in param or "max" in param:
        lo = param.get("min", param.get("max"))
        hi = param.get("max", param.get("min"))
        try:
            return (float(lo), float(hi), unit)
        except (TypeError, ValueError):
            return None
    return None


def _conditions_text(params: dict[str, Any]) -> str:
    parts: list[str] = []
    o2 = params.get("oxygen_concentration")
    if isinstance(o2, dict) and o2.get("value") is not None:
        parts.append(f"{o2['value']}{o2.get('unit', '')} O2")
    pressure = params.get("pressure")
    if isinstance(pressure, dict) and pressure.get("value") is not None:
        parts.append(f"{pressure['value']} {pressure.get('unit', '')}".strip())
    flow = _param_value(params.get("flow_velocity"))
    if flow:
        lo, hi, unit = flow
        if lo == hi:
            parts.append(f"{lo:g} {unit} airflow".strip())
        else:
            parts.append(f"{lo:g}-{hi:g} {unit} airflow".strip())
    sample = params.get("sample_size")
    if isinstance(sample, dict):
        width, length = sample.get("width"), sample.get("length")
        if width and length:
            parts.append(f"{width} x {length} m sample")
    return " · ".join(parts)


def _grounded_summary(exp: dict[str, Any]) -> str:
    name = exp.get("name") or exp.get("acronym") or "Experiment"
    meta = ", ".join(x for x in (exp.get("years"), exp.get("platform")) if x)
    lines = [f"{name}" + (f" ({meta})" if meta else "")]

    params = exp.get("parameters") or {}
    conditions = _conditions_text(params)
    if conditions:
        lines.append(f"Conditions: {conditions}.")

    chars = exp.get("flame_characteristics") or {}
    micro = _parse_range(chars.get("spread_rate_microgravity"))
    earth = _parse_range(chars.get("spread_rate_earth"))
    if micro and earth:
        micro_mid = sum(micro) / 2
        earth_mid = sum(earth) / 2
        change = (1 - micro_mid / earth_mid) * 100 if earth_mid else 0
        lines.append(
            f"Flame spread rate: {micro[0]:g}-{micro[1]:g} cm/s in microgravity vs "
            f"{earth[0]:g}-{earth[1]:g} cm/s at 1g (range midpoints ~{change:.0f}% lower)."
        )
    elif micro:
        lines.append(f"Flame spread rate in microgravity: {micro[0]:g}-{micro[1]:g} cm/s.")

    findings = exp.get("key_findings") or []
    for finding in findings[:2]:
        lines.append(f"- {finding}")

    lines.append("Chart generated from the experiment record in the local NASA knowledge base.")
    return "\n".join(lines)


# ── Rendering ─────────────────────────────────────────────────────────────────

def _style_axes(ax) -> None:
    ax.set_facecolor("#ffffff")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors=_MUTED_COLOR, labelsize=10)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def _render_chart(exp: dict[str, Any]) -> str:
    chars = exp.get("flame_characteristics") or {}
    params = exp.get("parameters") or {}
    micro = _parse_range(chars.get("spread_rate_microgravity"))
    earth = _parse_range(chars.get("spread_rate_earth"))

    fig = Figure(figsize=(8, 4.6), dpi=140, facecolor="#ffffff")

    if micro and earth:
        ax = fig.add_subplot(111)
        _style_axes(ax)
        rows = [("Microgravity (0g)", micro, _MICRO_COLOR), ("Earth gravity (1g)", earth, _EARTH_COLOR)]
        for row, (_label, (lo, hi), color) in enumerate(rows):
            ax.plot([lo, hi], [row, row], color=color, linewidth=12, solid_capstyle="round", alpha=0.9)
            ax.plot([(lo + hi) / 2], [row], marker="o", color="#ffffff", markeredgecolor=color,
                    markeredgewidth=2, markersize=8, zorder=3)
            ax.annotate(f"{lo:g}–{hi:g} cm/s", xy=((lo + hi) / 2, row), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=10,
                        color=_TEXT_COLOR, fontweight="bold")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=_TEXT_COLOR)
        ax.set_ylim(-0.6, 1.7)
        top = max(micro[1], earth[1])
        ax.set_xlim(0, max(top * 1.25, 1))
        ax.set_xlabel("Flame spread rate (cm/s)", fontsize=10, color=_MUTED_COLOR)
    else:
        numeric: list[tuple[str, tuple[float, float, str]]] = []
        for key, label in (
            ("oxygen_concentration", "Oxygen"),
            ("pressure", "Pressure"),
            ("flow_velocity", "Airflow"),
        ):
            parsed = _param_value(params.get(key))
            if parsed:
                numeric.append((label, parsed))
        if not numeric:
            raise ValueError("no numeric parameters to chart")

        axes = fig.subplots(1, len(numeric))
        if len(numeric) == 1:
            axes = [axes]
        for ax, (label, (lo, hi, unit)) in zip(axes, numeric):
            _style_axes(ax)
            val = hi if hi != lo else lo
            ax.bar([0], [val], color=_MICRO_COLOR, width=0.5, alpha=0.9)
            text = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
            ax.annotate(f"{text} {unit}".strip(), xy=(0, val), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=10,
                        color=_TEXT_COLOR, fontweight="bold")
            ax.set_xticks([])
            ax.set_title(label, fontsize=11, color=_TEXT_COLOR, fontweight="bold")
            ax.set_ylim(0, max(val * 1.35, 1))
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            ax.grid(axis="x", visible=False)

    name = exp.get("acronym") or exp.get("name") or "Experiment"
    fig.suptitle(f"{name} — combustion conditions", x=0.04, y=0.97, ha="left",
                 fontsize=14, fontweight="bold", color=_TEXT_COLOR)
    conditions = _conditions_text(params)
    if conditions:
        fig.text(0.04, 0.89, conditions, fontsize=9, color=_MUTED_COLOR, ha="left")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
