"""KSMI Schema loader and retrieval module.

Loads ksmi_schema.yaml and provides topic-based retrieval for the
knowledge_traversal tool. The YAML file is the single source of truth for
KSMI domain reasoning — definitions, rules, transitions, formulas,
and entity relationships.

NOTE: The primary retrieval path is now via KSMIGraphManager (LadybugDB),
which provides FTS search and Cypher graph traversal. This module serves as
a fallback when LadybugDB is unavailable. See ksmi_graph_manager.py.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "ksmi_schema.yaml"

_TOPICS: dict[str, list[str]] = {
    "definition": [
        "WaktuAcuanPelaporan",
        "Risk",
        "Uncertainty",
        "Petroleum",
        "InitialPetroleumInPlace",
        "SumberDaya",
        "CumulativeProduction",
        "Unrecoverable",
        "EstimatedUltimateRecovery",
        "Abandoned",
        "DokumenPenentuanStatusEksplorasi",
        "ClassificationHierarchy",
        "ProjectClassification",
        "Reserves",
        "ContingentResources",
        "ProspectiveResources",
        "CommercialFactors",
        "UncertaintyIncremental",
        "Play",
        "GeologicalChanceFactor",
        "WorkingArea",
        "Field",
        "Zone",
        "Project",
        "ProducingLicense",
        "GROOVY",
        "VolumeReporting",
        "Kodifikasi",
        "EntityRelationships",
    ],
    "level": [
        "ProjectLevel",
    ],
    "transition": [
        "LevelTransitionRules",
    ],
    "formula": [
        "VolumeFormulas",
        "CommercialFactors",
    ],
    "hierarchy": [
        "ClassificationHierarchy",
    ],
    "entity": [
        "WorkingArea",
        "Field",
        "Zone",
        "Project",
        "ProducingLicense",
        "GROOVY",
        "DokumenPenentuanStatusEksplorasi",
    ],
    "document": [
        "DokumenPenentuanStatusEksplorasi",
        "ProducingLicense",
        "GROOVY",
    ],
    "commercial": [
        "CommercialFactors",
    ],
    "all": [
        "_ALL_",
    ],
}


@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    """Load and cache the KSMI schema YAML."""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"KSMI schema not found: {_SCHEMA_PATH}")
    with open(_SCHEMA_PATH) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"KSMI schema is empty: {_SCHEMA_PATH}")
    return data


def _get_level_keys() -> list[str]:
    """Return all project level keys from ProjectLevel section."""
    schema = _load_schema()
    project_level = schema.get("ProjectLevel", {})
    return [
        k
        for k in project_level
        if k != "name"
        and k != "definition_code"
        and k != "aliases"
        and k != "description"
        and not k.startswith("_")
        and isinstance(project_level[k], dict)
        and "code" in project_level[k]
    ]


def _format_level_summary(level_data: dict, code: str) -> str:
    """Format a project level entry as concise summary."""
    name = level_data.get("name", "")
    proj_class = level_data.get("project_classification", "")
    is_pod = level_data.get("is_pod_approved")
    is_pse = level_data.get("is_pse_approved")
    key_concepts = level_data.get("key_concepts", [])

    is_pod_str = "true" if is_pod is True else "false" if is_pod is False else "null"
    is_pse_str = "true" if is_pse is True else "false" if is_pse is False else "null"

    concept_str = (
        "; ".join(
            list(kc.values())[0] if isinstance(kc, dict) else str(kc)
            for kc in key_concepts[:3]
        )
        if key_concepts
        else ""
    )

    result = f"**{code} — {name}** ({proj_class})"
    result += f"\n  is_pod_approved: {is_pod_str} | is_pse_approved: {is_pse_str}"
    if concept_str:
        result += f"\n  Key: {concept_str}"
    rule_note = level_data.get("rule_note")
    if rule_note:
        result += f"\n  Rule: {rule_note.strip()}"
    desc = level_data.get("description")
    if desc and isinstance(desc, str):
        result += f"\n  Desc: {desc.strip()[:200]}"
    return result


def _format_reachability(matrix: dict) -> str:
    """Format reachability matrix as readable text."""
    lines = ["### Reachability Matrix (from → to)"]
    for from_level, targets in matrix.items():
        if isinstance(targets, dict):
            to_list = targets.get("to_levels", targets.get("reachable", []))
            if isinstance(to_list, list):
                lines.append(f"  {from_level} → {', '.join(to_list)}")
    return "\n".join(lines)


def _format_wap_constraints(constraints: list) -> str:
    """Format WAP constraints as readable text."""
    lines = ["### WAP Constraints"]
    for c in constraints:
        if isinstance(c, dict):
            code = c.get("code", "")
            name = c.get("name", "")
            max_wap = c.get("max_wap_without_groovy")
            max_wap_g = c.get("max_wap_with_groovy")
            dest = c.get("destination_after", "")
            lines.append(
                f"  {code} ({name}): max {max_wap} WAP"
                f"{' (' + str(max_wap_g) + ' with GROOVY)' if max_wap_g else ''}"
                f"{', then → ' + dest if dest else ''}"
            )
    return "\n".join(lines)


def ksmi_retrieve(topic: str, entity: str | None = None) -> str:
    """Retrieve KSMI domain knowledge by topic and optional entity.

    Args:
        topic: One of 'definition', 'level', 'transition', 'formula',
            'hierarchy', 'entity', 'document', 'commercial', 'all'.
        entity: Optional entity name to narrow results (e.g. 'E0', 'GRR',
            'PSE', 'GROOVY', 'DokumenPenentuanStatusEksplorasi').

    Returns:
        Formatted text with the requested knowledge.
    """
    schema = _load_schema()

    topic_lower = topic.lower().strip()
    if topic_lower not in _TOPICS:
        valid = ", ".join(sorted(_TOPICS.keys()))
        return (
            f"Unknown topic '{topic}'. Valid topics: {valid}"
            f"\nUse 'all' for the entire knowledge base."
        )

    if topic_lower == "all":
        return _retrieve_all(schema)

    section_keys = _TOPICS[topic_lower]
    if section_keys == ["_ALL_"]:
        return _retrieve_all(schema)

    if entity:
        return _retrieve_entity(schema, entity)

    return _retrieve_topic(schema, topic_lower, section_keys)


def _ci_match(haystack: str, needle: str) -> bool:
    """Case-insensitive exact string match."""
    return haystack.lower() == needle.lower()


def _ci_in(needle: str, haystack: list[str]) -> bool:
    """Case-insensitive membership check: needle in haystack."""
    return needle.lower() in [h.lower() for h in haystack]


def _ci_contains(haystack: str, needle: str) -> bool:
    """Case-insensitive substring check: needle in haystack."""
    return needle.lower() in haystack.lower()


def _retrieve_entity(schema: dict[str, Any], entity: str) -> str:
    """Retrieve a specific entity from the schema."""
    entity_norm = entity.strip()

    for key in schema:
        if _ci_match(key, entity_norm):
            return _format_entity(key, schema[key])

    for key in schema:
        inner = schema[key]
        if not isinstance(inner, dict):
            continue
        code = inner.get("code", "")
        name = inner.get("name", "")
        aliases = inner.get("aliases", [])
        if (
            _ci_match(code, entity_norm)
            or _ci_match(name, entity_norm)
            or _ci_in(entity_norm, aliases)
        ):
            return _format_entity(key, inner)

    level_keys = _get_level_keys()
    project_level = schema.get("ProjectLevel", {})
    for lk in level_keys:
        ld = project_level[lk]
        if _ci_match(ld.get("code", ""), entity_norm) or _ci_in(
            entity_norm, ld.get("aliases", [])
        ):
            return _format_level_detail(ld)

    nested_sections = [
        "ProjectClassification",
        "CommercialFactors",
        "VolumeFormulas",
        "VolumeReporting",
        "KSMIFramework",
    ]
    for section_name in nested_sections:
        section = schema.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for sub_key, sub_val in section.items():
            if not isinstance(sub_val, dict):
                continue
            sub_name = sub_val.get("name", "")
            sub_aliases = sub_val.get("aliases", [])
            sub_code = sub_val.get("code", "")
            if (
                _ci_match(sub_key, entity_norm)
                or _ci_match(sub_code, entity_norm)
                or _ci_match(sub_name, entity_norm)
                or _ci_in(entity_norm, sub_aliases)
                or _ci_contains(sub_name, entity_norm)
            ):
                return _format_entity(f"{section_name}.{sub_key}", sub_val)

    matching = []
    for key in schema:
        if _ci_contains(key, entity_norm):
            matching.append(key)
        inner = schema[key]
        if isinstance(inner, dict):
            if _ci_contains(inner.get("name", ""), entity_norm):
                matching.append(key)
            if _ci_in(entity_norm, inner.get("aliases", [])):
                matching.append(key)

    if matching:
        results = []
        for m in matching[:5]:
            results.append(_format_entity(m, schema[m]))
        return "\n---\n".join(results)

    return _entity_not_found(entity_norm)


def _entity_not_found(entity: str) -> str:
    """Return helpful not-found message with available entities."""
    schema = _load_schema()
    top_level = sorted(k for k in schema if isinstance(schema[k], dict))
    level_keys = _get_level_keys()
    level_codes = [
        schema["ProjectLevel"][lk]["code"]
        for lk in level_keys
        if lk in schema.get("ProjectLevel", {})
    ]
    return (
        f"Entity '{entity}' not found in KSMI schema.\n\n"
        f"Available top-level sections: {', '.join(top_level)}\n"
        f"Available level codes: {', '.join(level_codes)}"
    )


def _retrieve_topic(schema: dict[str, Any], topic: str, section_keys: list[str]) -> str:
    """Retrieve all sections for a topic."""
    parts = [f"## KSMI Knowledge: {topic.upper()}\n"]
    for key in section_keys:
        if key not in schema:
            continue
        data = schema[key]

        if key == "ProjectLevel":
            parts.append(_format_levels_summary(schema))
        elif key == "LevelTransitionRules":
            parts.append(_format_transition_rules(data))
        elif key == "ClassificationHierarchy":
            parts.append(_format_hierarchy(data))
        else:
            parts.append(_format_entity(key, data))

    return "\n\n".join(parts)


def _retrieve_all(schema: dict[str, Any]) -> str:
    """Retrieve the entire knowledge base."""
    parts = ["## KSMI Complete Knowledge Base\n"]
    for key in schema:
        data = schema[key]
        if not isinstance(data, dict):
            continue
        if key == "ProjectLevel":
            parts.append(_format_levels_summary(schema))
            parts.append(_format_level_details(schema))
        elif key == "LevelTransitionRules":
            parts.append(_format_transition_rules(data))
        elif key == "ClassificationHierarchy":
            parts.append(_format_hierarchy(data))
        else:
            parts.append(_format_entity(key, data))
    return "\n\n---\n\n".join(parts)


def _format_entity(key: str, data: dict[str, Any]) -> str:
    """Format a single entity as readable text."""
    lines = [f"### {key}"]

    name = data.get("name")
    if name:
        lines.append(f"**Name:** {name}")

    code = data.get("code")
    if code:
        lines.append(f"**Code:** {code}")

    def_code = data.get("definition_code")
    if def_code:
        lines.append(f"**Definition Code:** {def_code}")

    aliases = data.get("aliases")
    if aliases:
        lines.append(f"**Aliases:** {', '.join(str(a) for a in aliases)}")

    definition = data.get("definition")
    if definition:
        def_text = definition.strip()
        if len(def_text) > 500:
            def_text = def_text[:500] + "..."
        lines.append(f"**Definition:** {def_text}")

    description = data.get("description")
    if description and isinstance(description, str):
        desc_text = description.strip()
        if len(desc_text) > 300:
            desc_text = desc_text[:300] + "..."
        lines.append(f"**Description:** {desc_text}")

    key_concepts = data.get("key_concepts")
    if key_concepts:
        lines.append("**Key Concepts:**")
        for kc in key_concepts:
            if isinstance(kc, dict):
                for k, v in kc.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"  - {kc}")

    is_pod = data.get("is_pod_approved")
    if is_pod is not None:
        lines.append(f"**is_pod_approved:** {is_pod}")

    is_pse = data.get("is_pse_approved")
    if is_pse is not None:
        lines.append(f"**is_pse_approved:** {is_pse}")

    commerciality_rank = data.get("commerciality_rank")
    if commerciality_rank is not None:
        lines.append(f"**Commerciality Rank:** {commerciality_rank}")

    db_column = data.get("db_column")
    if db_column:
        lines.append(f"**DB Column:** {db_column}")

    project_class = data.get("project_classification")
    if project_class:
        lines.append(f"**Classification:** {project_class}")

    return "\n".join(lines)


def _format_level_detail(level_data: dict) -> str:
    """Format a single project level with full detail."""
    code = level_data.get("code", "")
    name = level_data.get("name", "")
    proj_class = level_data.get("project_classification", "")
    definition = level_data.get("definition", "")
    key_concepts = level_data.get("key_concepts", [])
    rule_note = level_data.get("rule_note", "")
    is_pod = level_data.get("is_pod_approved")
    is_pse = level_data.get("is_pse_approved")

    lines = [
        f"### {code} — {name}",
        f"**Classification:** {proj_class}",
    ]

    if definition:
        def_text = definition.strip()
        if len(def_text) > 400:
            def_text = def_text[:400] + "..."
        lines.append(f"**Definition:** {def_text}")

    if key_concepts:
        lines.append("**Key Concepts:**")
        for kc in key_concepts:
            lines.append(f"  - {kc}")

    if rule_note:
        lines.append(f"**Rule:** {rule_note.strip()}")

    is_pod_str = "true" if is_pod is True else "false" if is_pod is False else "null"
    is_pse_str = "true" if is_pse is True else "false" if is_pse is False else "null"
    lines.append(
        f"**is_pod_approved:** {is_pod_str} | **is_pse_approved:** {is_pse_str}"
    )

    return "\n".join(lines)


def _format_levels_summary(schema: dict) -> str:
    """Format a concise summary table of all project levels."""
    project_level = schema.get("ProjectLevel", {})
    level_keys = _get_level_keys()

    lines = ["### Project Maturity Levels (E0–X6, A1–A2)", ""]

    for lk in level_keys:
        ld = project_level.get(lk, {})
        if not isinstance(ld, dict) or "code" not in ld:
            continue
        lines.append(_format_level_summary(ld, ld["code"]))

    return "\n\n".join(lines)


def _format_level_details(schema: dict) -> str:
    """Format full details of all project levels."""
    project_level = schema.get("ProjectLevel", {})
    level_keys = _get_level_keys()

    lines = ["### Project Level Details", ""]
    for lk in level_keys:
        ld = project_level.get(lk, {})
        if not isinstance(ld, dict) or "code" not in ld:
            continue
        lines.append(_format_level_detail(ld))
        lines.append("")

    return "\n".join(lines)


def _format_transition_rules(data: dict) -> str:
    """Format level transition rules."""
    lines = ["### Level Transition Rules"]

    desc = data.get("description")
    if desc:
        lines.append(desc.strip())
        lines.append("")

    matrix = data.get("reachability_matrix")
    if matrix and isinstance(matrix, dict):
        rm = matrix if "description" not in matrix else matrix
        lines.append(_format_reachability(rm))

    wap = data.get("wap_constraints")
    if wap and isinstance(wap, list):
        lines.append("")
        lines.append(_format_wap_constraints(wap))

    return "\n".join(lines)


def _format_hierarchy(data: dict) -> str:
    """Format classification hierarchy as indented tree."""
    lines = ["### Classification Hierarchy"]

    def _walk(node: Any, indent: int = 0) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                prefix = "  " * indent
                if isinstance(v, dict):
                    if "uncertainty_levels" in v:
                        lines.append(f"{prefix}- {k}")
                        uncertainty = v.get("uncertainty_levels", [])
                        if isinstance(uncertainty, list):
                            for u in uncertainty:
                                code = u.get("code", "")
                                name = u.get("name", "")
                                pct = u.get("percentile", "")
                                lines.append(f"{prefix}  · {code}: {name} ({pct})")
                    else:
                        lines.append(f"{prefix}- {k}")
                        _walk(v, indent + 1)
                elif v is None:
                    lines.append(f"{prefix}- {k}")
                elif isinstance(v, str):
                    lines.append(f"{prefix}- {k}: {v}")
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    code = item.get("code", "")
                    name = item.get("name", "")
                    lines.append(f"{'  ' * indent}· {code}: {name}")

    tree = data.get("tree")
    if tree:
        _walk(tree)

    return "\n".join(lines)
