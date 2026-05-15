"""Tests for KSMI knowledge graph modules (builder + manager)."""

import threading

import pytest

from esdc.chat.domain_knowledge.ksmi_graph_builder import KSMIGraphBuilder
from esdc.chat.domain_knowledge.ksmi_graph_manager import KSMIGraphManager


@pytest.fixture(autouse=True)
def reset_singleton():
    yield
    KSMIGraphManager._instance = None


@pytest.fixture
def builder():
    b = KSMIGraphBuilder()
    b.build_from_yaml()
    yield b
    b.close()


@pytest.fixture
def manager():
    mgr = KSMIGraphManager()
    yield mgr
    mgr.close()


class TestKSMIGraphBuilder:
    """Tests for KSMIGraphBuilder — YAML → LadybugDB graph construction."""

    def test_build_from_yaml_succeeds(self, builder):
        assert builder.is_built

    def test_idempotent_build(self, builder):
        builder.build_from_yaml()
        assert builder.is_built

    def test_node_counts_populated(self, builder):
        counts = builder._node_counts
        assert counts.get("KSMIFrameworkNode", 0) >= 1
        assert counts.get("ProjectLevel", 0) >= 5
        assert counts.get("VolumeType", 0) >= 3
        assert counts.get("VolumeSubstance", 0) >= 3

    def test_edge_counts_populated(self, builder):
        counts = builder._edge_counts
        assert counts.get("HAS_LEVEL", 0) >= 5
        assert counts.get("CLASSIFIED_AS", 0) >= 3

    def test_builder_levels_exist(self, builder):
        result = builder.conn.execute("MATCH (l:ProjectLevel) RETURN l.code, l.name")
        levels = {row[0]: row[1] for row in result}
        assert "E0" in levels or any("E" in k for k in levels)

    def test_builder_transitions_exist(self, builder):
        result = builder.conn.execute(
            "MATCH (from:ProjectLevel)-[r:CAN_TRANSITION_TO]->(to:ProjectLevel) "
            "RETURN from.code, to.code LIMIT 5"
        )
        transitions = [(row[0], row[1]) for row in result]
        assert len(transitions) >= 1

    def test_classifications_exist(self, builder):
        result = builder.conn.execute(
            "MATCH (e:KSMIEntity) WHERE e.entity_type = 'classification' "
            "RETURN e.code, e.name"
        )
        classifications = {row[0]: row[1] for row in result}
        assert len(classifications) >= 1

    def test_volume_types_created(self, builder):
        result = builder.conn.execute("MATCH (v:VolumeType) RETURN v.code, v.name")
        vols = {row[0]: row[1] for row in result}
        assert "gross" in vols or len(vols) >= 1


class TestKSMIGraphManagerFind:
    """Tests for KSMIGraphManager.find() — BM25 FTS entity resolution."""

    def test_find_reserves(self, manager):
        results = manager.find("reserves")
        assert len(results) >= 1
        codes = [r["code"] for r in results]
        assert any("Reserve" in c or "reserve" in c.lower() for c in codes)

    def test_find_reserves_case_insensitive(self, manager):
        lower = manager.find("reserves")
        upper = manager.find("RESERVES")
        assert len(lower) >= 1
        assert len(upper) >= 1

    def test_find_cadangan_indonesian(self, manager):
        results = manager.find("cadangan")
        assert len(results) >= 1

    def test_find_e0_level_via_fts_name(self, manager):
        """FTS works better with natural language names than short codes."""
        results = manager.find("On Production", table="ProjectLevel")
        assert len(results) >= 1
        assert any(r["code"] == "E0" for r in results)

    def test_find_returns_score(self, manager):
        results = manager.find("reserves")
        assert len(results) >= 1
        assert "score" in results[0]
        assert results[0]["score"] > 0

    def test_find_nonexistent_returns_empty(self, manager):
        results = manager.find("xyznonexistent12345")
        assert len(results) == 0

    def test_find_top_k_limits_results(self, manager):
        all_results = manager.find("resources", top_k=2)
        assert len(all_results) <= 2


class TestKSMIGraphManagerFindAll:
    """Tests for KSMIGraphManager.find_all() — cross-table search."""

    def test_find_all_reserves(self, manager):
        results = manager.find_all("reserves")
        assert len(results) >= 1
        assert "source_table" in results[0]

    def test_find_all_deduplicates(self, manager):
        results = manager.find_all("reserves")
        codes = [r["code"] for r in results]
        assert len(codes) == len(set(codes))

    def test_find_all_sorted_by_score(self, manager):
        results = manager.find_all("reserves")
        scores = [r.get("score", 0) for r in results]
        assert scores == sorted(scores, reverse=True)


class TestKSMIGraphManagerTraverse:
    """Tests for KSMIGraphManager.traverse() — relationship traversal."""

    def test_traverse_classified_as(self, manager):
        results = manager.traverse("E0", "CLASSIFIED_AS")
        assert len(results) >= 1
        assert any(
            "Reserve" in r["name"] or "reserve" in r["name"].lower() for r in results
        )

    def test_traverse_can_transition_to(self, manager):
        results = manager.traverse("E0", "CAN_TRANSITION_TO")
        assert len(results) >= 1
        target_codes = [r["code"] for r in results]
        assert any(code.startswith("E") for code in target_codes)

    def test_traverse_reported_as(self, manager):
        results = manager.traverse("1. Reserves & GRR", "REPORTED_AS")
        assert len(results) >= 1

    def test_traverse_nonexistent_entity_returns_empty(self, manager):
        results = manager.traverse("NONEXISTENT", "CLASSIFIED_AS")
        assert len(results) == 0

    def test_traverse_nonexistent_rel_returns_empty(self, manager):
        results = manager.traverse("E0", "NONEXISTENT_REL")
        assert len(results) == 0


class TestKSMIGraphManagerTraverseReverse:
    """Tests for KSMIGraphManager.traverse_reverse() — backward traversal."""

    def test_traverse_reverse_classified_as(self, manager):
        results = manager.traverse_reverse("1. Reserves & GRR", "CLASSIFIED_AS")
        assert len(results) >= 1
        codes = [r["code"] for r in results]
        assert "E0" in codes or any(c.startswith("E") for c in codes)

    def test_traverse_reverse_has_level(self, manager):
        results = manager.traverse_reverse("E0", "HAS_LEVEL")
        assert len(results) >= 1


class TestKSMIGraphManagerQuery:
    """Tests for KSMIGraphManager.query() — raw Cypher."""

    def test_query_count_all_nodes(self, manager):
        results = manager.query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
        )
        assert len(results) >= 1

    def test_query_project_levels(self, manager):
        results = manager.query("MATCH (l:ProjectLevel) RETURN l.code, l.name LIMIT 3")
        assert len(results) >= 1

    def test_query_invalid_cypher_returns_empty(self, manager):
        results = manager.query("MATCH INVALID CYPHER HERE")
        assert len(results) == 0


class TestKSMIGraphManagerFindLevels:
    """Tests for KSMIGraphManager.find_levels()."""

    def test_find_all_levels(self, manager):
        results = manager.find_levels()
        assert len(results) >= 5
        assert "code" in results[0]
        assert "name" in results[0]

    def test_find_levels_by_class(self, manager):
        results = manager.find_levels("Reserves & GRR")
        assert len(results) >= 1
        assert all(
            "Reserve" in r.get("project_classification", "")
            or "reserve" in r.get("project_classification", "").lower()
            for r in results
        )


class TestKSMIGraphManagerGetTransitions:
    """Tests for KSMIGraphManager.get_transitions()."""

    def test_get_all_transitions(self, manager):
        results = manager.get_transitions()
        assert len(results) >= 1
        assert "from" in results[0]
        assert "to" in results[0]

    def test_get_transitions_from_e0(self, manager):
        results = manager.get_transitions("E0")
        assert len(results) >= 1
        assert all(r["from"] == "E0" for r in results)


class TestKSMIGraphManagerIsAvailable:
    """Tests for KSMIGraphManager.is_available property."""

    def test_is_available_after_init(self, manager):
        manager.find("test")
        assert manager.is_available


class TestKSMIGraphManagerClose:
    """Tests for KSMIGraphManager lifecycle."""

    def test_close_and_reinitialize(self):
        mgr = KSMIGraphManager()
        mgr.find("reserves")
        assert mgr.is_available
        mgr.close()
        assert not mgr.is_available

    def test_singleton_pattern(self):
        mgr1 = KSMIGraphManager()
        mgr2 = KSMIGraphManager()
        assert mgr1 is mgr2
        mgr1.close()


class TestCypherEscaping:
    """Test that single quotes and special chars are properly escaped in Cypher."""

    def test_escape_method(self):
        assert KSMIGraphBuilder._cypher_escape("Indonesia's") == "Indonesia\\'s"
        assert KSMIGraphBuilder._cypher_escape('say "hello"') == 'say \\"hello\\"'
        assert KSMIGraphBuilder._cypher_escape("back\\slash") == "back\\\\slash"
        assert KSMIGraphBuilder._cypher_escape("normal") == "normal"
        assert KSMIGraphBuilder._cypher_escape("") == ""

    def test_entity_with_single_quote_stored_and_retrievable(self):
        builder = KSMIGraphBuilder()
        builder.build_from_yaml()
        builder._conn.execute(
            "CREATE (e:KSMIEntity {code: 'QUOTE_TEST', "
            "name: 'Indonesia\\'s Reserve', "
            "aliases: 'Indonesia\\'s oil', "
            "definition: 'It\\'s a test of Singapore\\'s data', "
            "entity_type: 'test'})"
        )
        result = builder.conn.execute(
            "MATCH (e:KSMIEntity {code: 'QUOTE_TEST'}) "
            "RETURN e.name, e.aliases, e.definition"
        )
        rows = list(result)
        assert len(rows) == 1
        assert rows[0][0] == "Indonesia's Reserve"
        assert rows[0][1] == "Indonesia's oil"
        assert rows[0][2] == "It's a test of Singapore's data"
        builder.close()

    def test_project_level_with_quotes_stored_and_retrievable(self):
        builder = KSMIGraphBuilder()
        builder.build_from_yaml()
        builder._conn.execute(
            "CREATE (l:ProjectLevel {code: 'Q_LEVEL', "
            "name: 'Level\\'s Name', "
            "is_pod_approved: false, "
            "is_pse_approved: true, "
            "project_classification: 'Reserve\\'s Class', "
            "definition: 'It\\'s defined here', "
            "rule_note: 'Note\\'s rule'})"
        )
        result = builder.conn.execute(
            "MATCH (l:ProjectLevel {code: 'Q_LEVEL'}) "
            "RETURN l.name, l.project_classification, "
            "l.definition, l.rule_note"
        )
        rows = list(result)
        assert len(rows) == 1
        assert rows[0][0] == "Level's Name"
        assert rows[0][1] == "Reserve's Class"
        assert rows[0][2] == "It's defined here"
        assert rows[0][3] == "Note's rule"
        builder.close()


class TestThreadSafety:
    def test_concurrent_initialization(self):
        KSMIGraphManager._instance = None
        results = []
        errors = []

        def init_and_find():
            try:
                mgr = KSMIGraphManager()
                mgr.find("reserves")
                results.append(mgr.is_available)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=init_and_find) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)
