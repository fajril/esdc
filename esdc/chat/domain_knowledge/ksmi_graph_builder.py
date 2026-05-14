"""Build KSMI domain knowledge graph in LadybugDB from YAML schema.

Parses ksmi_schema.yaml and creates nodes, relationships, and FTS indexes
in an in-memory LadybugDB database for rich Cypher queries and BM25 search.

The YAML file remains the single source of truth. This module rebuilds
the graph from YAML on every initialization.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import real_ladybug as lb
import yaml

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "ksmi_schema.yaml"


class KSMIGraphBuilder:
    """Parse KSMI YAML schema and populate LadybugDB graph."""

    def __init__(self) -> None:
        self._db: lb.Database | None = None
        self._conn: lb.Connection | None = None
        self._node_counts: dict[str, int] = {}
        self._edge_counts: dict[str, int] = {}
        self._class_map: dict[str, str] = {}
        self._built = False

    @property
    def conn(self) -> lb.Connection:
        if not self._built or self._conn is None:
            msg = "Graph not built. Call build_from_yaml() first."
            raise RuntimeError(msg)
        return self._conn

    @property
    def is_built(self) -> bool:
        return self._built

    def node_count(self, label: str) -> int:
        return self._node_counts.get(label, 0)

    def edge_count(self, label: str) -> int:
        return self._edge_counts.get(label, 0)

    @staticmethod
    def _cypher_escape(s: str) -> str:
        s = s.replace("\\", "\\\\")
        s = s.replace("'", "\\'")
        s = s.replace('"', '\\"')
        return s

    def _build_classification_map(self, schema: dict[str, Any]) -> None:
        pc = schema.get("ProjectClassification", {})
        for key, val in pc.items():
            if not isinstance(val, dict):
                continue
            name = val.get("name", "")
            code = val.get("code", key)
            if name and code:
                self._class_map[name] = code

    def build_from_yaml(self, schema_path: Path | str | None = None) -> None:
        if self._built:
            return

        path = Path(schema_path) if schema_path else _SCHEMA_PATH
        with open(path) as f:
            schema = yaml.safe_load(f)

        self._db = lb.Database(":memory:")
        self._conn = lb.Connection(self._db)

        self._load_extensions()
        self._create_schema()
        self._build_classification_map(schema)
        self._create_nodes(schema)
        self._create_relationships(schema)
        self._create_fts_indexes()

        self._built = True
        logger.info(
            "[KSMI-KG] graph_built | nodes=%s edges=%s",
            self._node_counts,
            self._edge_counts,
        )

    def _load_extensions(self) -> None:
        assert self._conn is not None
        self._conn.execute("INSTALL FTS")
        self._conn.execute("LOAD EXTENSION FTS")

    def _create_schema(self) -> None:
        assert self._conn is not None
        stmts = [
            (
                "CREATE NODE TABLE KSMIEntity "
                "(code STRING PRIMARY KEY, name STRING, "
                "aliases STRING, definition STRING, entity_type STRING)"
            ),
            (
                "CREATE NODE TABLE ProjectLevel "
                "(code STRING PRIMARY KEY, name STRING, "
                "is_pod_approved BOOLEAN, is_pse_approved BOOLEAN, "
                "project_classification STRING, definition STRING, "
                "rule_note STRING)"
            ),
            (
                "CREATE NODE TABLE VolumeType "
                "(code STRING PRIMARY KEY, name STRING, description STRING)"
            ),
            (
                "CREATE NODE TABLE VolumeSubstance "
                "(code STRING PRIMARY KEY, name STRING, unit STRING, "
                "substance STRING, volume_category STRING)"
            ),
            (
                "CREATE NODE TABLE KSMIFrameworkNode "
                "(code STRING PRIMARY KEY, name STRING, "
                "aliases STRING, definition STRING, source STRING)"
            ),
            (
                "CREATE REL TABLE CLASSIFIED_AS "
                "(FROM ProjectLevel TO KSMIEntity)"
            ),
            (
                "CREATE REL TABLE REPORTED_AS "
                "(FROM KSMIEntity TO VolumeType)"
            ),
            (
                "CREATE REL TABLE CAN_TRANSITION_TO "
                "(FROM ProjectLevel TO ProjectLevel, "
                "condition STRING, rule_note STRING)"
            ),
            (
                "CREATE REL TABLE BELONGS_TO_FRAMEWORK "
                "(FROM KSMIEntity TO KSMIFrameworkNode)"
            ),
            (
                "CREATE REL TABLE HAS_LEVEL "
                "(FROM KSMIFrameworkNode TO ProjectLevel)"
            ),
            (
                "CREATE REL TABLE HAS_SUBSTANCE "
                "(FROM KSMIEntity TO VolumeSubstance)"
            ),
        ]
        for stmt in stmts:
            try:
                self._conn.execute(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("[KSMI-KG] schema_error | %s", e)

    def _create_nodes(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        self._create_framework_node(schema)
        self._create_top_level_entities(schema)
        self._create_project_levels(schema)
        self._create_project_classifications(schema)
        self._create_volume_types(schema)
        self._create_volume_substances(schema)
        self._create_commercial_factors(schema)

    def _create_framework_node(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        fw = schema.get("KSMIFramework", {})
        if not fw:
            return
        code = fw.get("code", "KSMI")
        name = fw.get("name", "")
        aliases = self._cypher_escape("||".join(fw.get("aliases", [])))
        definition = self._cypher_escape((fw.get("definition") or "")[:2000])
        source = self._cypher_escape((fw.get("source") or "")[:500])
        try:
            self._conn.execute(
                f"CREATE (f:KSMIFrameworkNode "
                f"{{code: '{self._cypher_escape(code)}', "
                f"name: '{self._cypher_escape(name)}', "
                f"aliases: '{aliases}', "
                f"definition: '{definition}', "
                f"source: '{source}'}})"
            )
            self._node_counts["KSMIFrameworkNode"] = (
                self._node_counts.get("KSMIFrameworkNode", 0) + 1
            )
        except Exception as e:
            logger.warning("[KSMI-KG] framework_node_error | %s", e)

    def _create_top_level_entities(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        skip_sections = {
            "KSMIFramework",
            "ProjectLevel",
            "ProjectClassification",
            "ClassificationHierarchy",
            "LevelTransitionRules",
            "VolumeReporting",
            "CommercialFactors",
            "VolumeFormulas",
            "EntityRelationships",
        }
        count = 0
        for key, val in schema.items():
            if key in skip_sections or not isinstance(val, dict):
                continue
            code = val.get("code", key)
            name = val.get("name", key)
            aliases = self._cypher_escape("||".join(val.get("aliases", [])))
            definition = self._cypher_escape((val.get("definition") or "")[:2000])
            entity_type = self._classify_entity(key, val)
            try:
                self._conn.execute(
                    f"CREATE (e:KSMIEntity "
                    f"{{code: '{self._cypher_escape(code)}', "
                    f"name: '{self._cypher_escape(name)}', "
                    f"aliases: '{aliases}', "
                    f"definition: '{definition}', "
                    f"entity_type: '{self._cypher_escape(entity_type)}'}})"
                )
                count += 1
            except Exception as e:
                logger.warning("[KSMI-KG] entity_error | key=%s err=%s", key, e)
        self._node_counts["KSMIEntity"] = (
            self._node_counts.get("KSMIEntity", 0) + count
        )

    def _classify_entity(self, key: str, val: dict[str, Any]) -> str:
        concepts = val.get("key_concepts", [])
        if any("commercial" in str(c).lower() for c in concepts):
            return "commercial"
        if any(
            "risk" in str(c).lower() or "chance" in str(c).lower()
            for c in concepts
        ):
            return "risk"
        if "unit" in val or "units" in val:
            return "volume"
        return "concept"

    def _create_project_levels(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        pl = schema.get("ProjectLevel", {})
        level_keys = [
            k
            for k in pl
            if isinstance(pl[k], dict) and "code" in pl[k]
        ]
        count = 0
        for lk in level_keys:
            ld = pl[lk]
            code = ld.get("code", "")
            name = ld.get("name", "")
            is_pod = ld.get("is_pod_approved", False)
            is_pse = ld.get("is_pse_approved", False)
            proj_class = ld.get("project_classification", "")
            definition = self._cypher_escape((ld.get("definition") or "")[:1000])
            rule_note = self._cypher_escape((ld.get("rule_note") or "")[:500])
            try:
                self._conn.execute(
                    f"CREATE (l:ProjectLevel "
                    f"{{code: '{self._cypher_escape(code)}', "
                    f"name: '{self._cypher_escape(name)}', "
                    f"is_pod_approved: {'true' if is_pod else 'false'}, "
                    f"is_pse_approved: {'true' if is_pse else 'false'}, "
                    f"project_classification: "
                    f"'{self._cypher_escape(proj_class)}', "
                    f"definition: '{definition}', "
                    f"rule_note: '{rule_note}'}})"
                )
                count += 1
            except Exception as e:
                logger.warning("[KSMI-KG] level_error | code=%s err=%s", code, e)
        self._node_counts["ProjectLevel"] = (
            self._node_counts.get("ProjectLevel", 0) + count
        )

    def _create_project_classifications(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        pc = schema.get("ProjectClassification", {})
        count = 0
        for key, val in pc.items():
            if not isinstance(val, dict):
                continue
            code = val.get("code", key)
            name = val.get("name", key)
            aliases = self._cypher_escape("||".join(val.get("aliases", [])))
            definition = ""
            defs = val.get("definitions", {})
            if isinstance(defs, dict):
                parts = [f"{k}: {v[:200]}" for k, v in defs.items()]
                definition = self._cypher_escape(" | ".join(parts))
            entity_type = "classification"
            try:
                self._conn.execute(
                    f"CREATE (e:KSMIEntity "
                    f"{{code: '{self._cypher_escape(code)}', "
                    f"name: '{self._cypher_escape(name)}', "
                    f"aliases: '{aliases}', "
                    f"definition: '{definition}', "
                    f"entity_type: '{self._cypher_escape(entity_type)}'}})"
                )
                count += 1
            except Exception as e:
                logger.warning("[KSMI-KG] class_error | key=%s err=%s", key, e)
        self._node_counts["KSMIEntity"] = (
            self._node_counts.get("KSMIEntity", 0) + count
        )

    def _create_volume_types(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        vr = schema.get("VolumeReporting", {})
        vth = vr.get("volume_type_hierarchy", [])
        if not isinstance(vth, list):
            return
        count = 0
        for vt in vth:
            if not isinstance(vt, dict):
                continue
            code = vt.get("code", "")
            name = vt.get("name", "")
            description = self._cypher_escape((vt.get("description") or "")[:500])
            try:
                self._conn.execute(
                    f"CREATE (v:VolumeType "
                    f"{{code: '{self._cypher_escape(code)}', "
                    f"name: '{self._cypher_escape(name)}', "
                    f"description: '{description}'}})"
                )
                count += 1
            except Exception as e:
                logger.warning("[KSMI-KG] volume_type_error | code=%s err=%s", code, e)
        self._node_counts["VolumeType"] = (
            self._node_counts.get("VolumeType", 0) + count
        )

    def _create_volume_substances(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        vr = schema.get("VolumeReporting", {})
        categories = [
            "reserves", "grr",
            "contingent_resources", "prospective_resources",
        ]
        seen_codes: set[str] = set()
        count = 0
        for cat in categories:
            cat_data = vr.get(cat, {})
            if not isinstance(cat_data, dict):
                continue
            items = cat_data.get("items", [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                code = item.get("code", "")
                if code in seen_codes:
                    continue
                seen_codes.add(code)
                name = item.get("name", "")
                unit = item.get("unit", "")
                substance = item.get("substance", "")
                try:
                    self._conn.execute(
                        f"CREATE (vs:VolumeSubstance "
                        f"{{code: '{self._cypher_escape(code)}', "
                        f"name: '{self._cypher_escape(name)}', "
                        f"unit: '{self._cypher_escape(unit)}', "
                        f"substance: "
                        f"'{self._cypher_escape(substance)}', "
                        f"volume_category: "
                        f"'{self._cypher_escape(cat)}'}})"
                    )
                    count += 1
                except Exception as e:
                    logger.warning(
                        "[KSMI-KG] substance_error | code=%s err=%s",
                        code,
                        e,
                    )
        self._node_counts["VolumeSubstance"] = (
            self._node_counts.get("VolumeSubstance", 0) + count
        )

    def _create_commercial_factors(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        cf = schema.get("CommercialFactors", {})
        factors = cf.get("factors", {})
        if not isinstance(factors, dict):
            return
        count = 0
        for key, val in factors.items():
            if not isinstance(val, dict):
                continue
            code = val.get("code", key)
            name = val.get("name", key)
            aliases = self._cypher_escape("||".join(val.get("aliases", [])))
            definition = self._cypher_escape((val.get("definition") or "")[:500])
            entity_type = "commercial_factor"
            try:
                self._conn.execute(
                    f"CREATE (e:KSMIEntity "
                    f"{{code: '{self._cypher_escape(code)}', "
                    f"name: '{self._cypher_escape(name)}', "
                    f"aliases: '{aliases}', "
                    f"definition: '{definition}', "
                    f"entity_type: "
                    f"'{self._cypher_escape(entity_type)}'}})"
                )
                count += 1
            except Exception as e:
                logger.warning("[KSMI-KG] commercial_error | key=%s err=%s", key, e)
        self._node_counts["KSMIEntity"] = (
            self._node_counts.get("KSMIEntity", 0) + count
        )

    def _create_relationships(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        self._link_levels_to_classifications(schema)
        self._link_classifications_to_volume_types(schema)
        self._link_level_transitions(schema)
        self._link_framework_to_levels(schema)
        self._link_entities_to_framework(schema)
        # TODO: HAS_SUBSTANCE edges require entity→substance mapping
        # not yet available in ksmi_schema.yaml

    def _link_levels_to_classifications(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        pl = schema.get("ProjectLevel", {})
        level_keys = [k for k in pl if isinstance(pl[k], dict) and "code" in pl[k]]
        count = 0
        for lk in level_keys:
            ld = pl[lk]
            proj_class = ld.get("project_classification", "")
            level_code = ld.get("code", "")
            class_code = self._classification_name_to_code(proj_class)
            if not class_code:
                continue
            try:
                self._conn.execute(
                    f"MATCH (l:ProjectLevel "
                    f"{{code: '{self._cypher_escape(level_code)}'}}), "
                    f"(c:KSMIEntity "
                    f"{{code: '{self._cypher_escape(class_code)}'}}) "
                    f"CREATE (l)-[:CLASSIFIED_AS]->(c)"
                )
                count += 1
            except Exception as e:
                logger.warning("[KSMI-KG] class_link_error | %s", e)
        self._edge_counts["CLASSIFIED_AS"] = (
            self._edge_counts.get("CLASSIFIED_AS", 0) + count
        )

    def _classification_name_to_code(self, name: str) -> str:
        return self._class_map.get(name, "")

    def _link_classifications_to_volume_types(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        class_to_volumes = {
            "1. Reserves & GRR": ["gross", "net", "sales"],
            "2. Contingent Resources": ["gross", "net", "sales"],
            "3. Prospective Resources": ["gross", "net", "sales"],
        }
        count = 0
        for class_code, volumes in class_to_volumes.items():
            for vol_code in volumes:
                try:
                    self._conn.execute(
                        f"MATCH (c:KSMIEntity "
                        f"{{code: '{self._cypher_escape(class_code)}'}}), "
                        f"(v:VolumeType "
                        f"{{code: '{self._cypher_escape(vol_code)}'}}) "
                        f"CREATE (c)-[:REPORTED_AS]->(v)"
                    )
                    count += 1
                except Exception as e:
                    logger.warning("[KSMI-KG] vol_link_error | %s", e)
        self._edge_counts["REPORTED_AS"] = (
            self._edge_counts.get("REPORTED_AS", 0) + count
        )

    def _link_level_transitions(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        ltr = schema.get("LevelTransitionRules", {})
        matrix = ltr.get("reachability_matrix", {})
        if not isinstance(matrix, dict):
            return
        count = 0
        for from_level, targets in matrix.items():
            if not isinstance(targets, dict):
                continue
            reachable = targets.get("reachable", [])
            if not isinstance(reachable, list):
                continue
            for to_level in reachable:
                try:
                    self._conn.execute(
                        f"MATCH (from:ProjectLevel "
                        f"{{code: "
                        f"'{self._cypher_escape(from_level)}'}}), "
                        f"(to:ProjectLevel "
                        f"{{code: '{self._cypher_escape(to_level)}'}}) "
                        f"CREATE (from)-[:CAN_TRANSITION_TO "
                        f"{{condition: 'reachable', "
                        f"rule_note: ''}}]->(to)"
                    )
                    count += 1
                except Exception as e:
                    logger.warning("[KSMI-KG] transition_error | %s", e)
        self._edge_counts["CAN_TRANSITION_TO"] = (
            self._edge_counts.get("CAN_TRANSITION_TO", 0) + count
        )

    def _link_framework_to_levels(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        fw_code = "KSMI"
        pl = schema.get("ProjectLevel", {})
        level_keys = [k for k in pl if isinstance(pl[k], dict) and "code" in pl[k]]
        count = 0
        for lk in level_keys:
            code = pl[lk].get("code", "")
            try:
                self._conn.execute(
                    f"MATCH (f:KSMIFrameworkNode "
                    f"{{code: '{self._cypher_escape(fw_code)}'}}), "
                    f"(l:ProjectLevel "
                    f"{{code: '{self._cypher_escape(code)}'}}) "
                    f"CREATE (f)-[:HAS_LEVEL]->(l)"
                )
                count += 1
            except Exception as e:
                logger.warning("[KSMI-KG] fw_level_error | %s", e)
        self._edge_counts["HAS_LEVEL"] = (
            self._edge_counts.get("HAS_LEVEL", 0) + count
        )

    def _link_entities_to_framework(self, schema: dict[str, Any]) -> None:
        assert self._conn is not None
        fw = schema.get("KSMIFramework", {})
        if not fw:
            return
        fw_code = fw.get("code", "KSMI")
        fw_code_esc = self._cypher_escape(fw_code)
        result = self._conn.execute("MATCH (e:KSMIEntity) RETURN e.code")
        count = 0
        while result.has_next():  # type: ignore[union-attr]
            code = result.get_next()[0]  # type: ignore[union-attr]
            with contextlib.suppress(Exception):
                self._conn.execute(
                    f"MATCH (e:KSMIEntity {{code: '{self._cypher_escape(code)}'}}), "
                    f"(f:KSMIFrameworkNode {{code: '{fw_code_esc}'}}) "
                    f"CREATE (e)-[:BELONGS_TO_FRAMEWORK]->(f)"
                )
                count += 1
        self._edge_counts["BELONGS_TO_FRAMEWORK"] = (
            self._edge_counts.get("BELONGS_TO_FRAMEWORK", 0) + count
        )

    def _create_fts_indexes(self) -> None:
        assert self._conn is not None
        indexes = [
            (
                "KSMIEntity",
                "entity_fts",
                ["code", "name", "aliases", "definition", "entity_type"],
            ),
            (
                "ProjectLevel",
                "level_fts",
                ["code", "name", "project_classification", "definition"],
            ),
            ("VolumeType", "volume_fts", ["code", "name", "description"]),
            (
                "VolumeSubstance",
                "substance_fts",
                ["code", "name", "substance"],
            ),
            (
                "KSMIFrameworkNode",
                "framework_fts",
                ["code", "name", "aliases", "definition"],
            ),
        ]
        for table, idx_name, props in indexes:
            try:
                props_literal = (
                    "[" + ", ".join(f"'{p}'" for p in props) + "]"
                )
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"{props_literal}, stemmer := 'indonesian')"
                )
                logger.debug("[KSMI-KG] fts_created | %s.%s", table, idx_name)
            except Exception as e:
                logger.warning(
                    "[KSMI-KG] fts_error | table=%s idx=%s err=%s",
                    table,
                    idx_name,
                    e,
                )

    def close(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
        if self._db is not None:
            with contextlib.suppress(Exception):
                self._db.close()
            self._db = None
        self._built = False
