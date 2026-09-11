"""Tests for knowledge_traversal tool reachability matrix injection.

Verifies that knowledge_traversal auto-includes the reachability matrix
when topic is 'transition' or 'level', and that E3 specifically does
NOT list E4 as a valid target (regression test for the original
hallucination issue).
"""

from __future__ import annotations

from esdc.chat.tools import knowledge_traversal


class TestReachabilityInjection:
    """Verify the reachability matrix is included for transition/level topics."""

    def test_transition_topic_includes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "transition", "entity": "E3"})
        assert "Reachability Matrix" in result
        assert "E3 →" in result

    def test_transition_topic_with_entity_highlights_row(self):
        result = knowledge_traversal.invoke({"topic": "transition", "entity": "E3"})
        assert ">>>" in result
        highlighted = [line for line in result.split("\n") if ">>>" in line]
        assert any("E3 →" in line for line in highlighted)

    def test_level_topic_includes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "level", "entity": "E3"})
        assert "Reachability Matrix" in result

    def test_transition_topic_without_entity_includes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "transition"})
        assert "Reachability Matrix" in result
        assert ">>>" not in result

    def test_definition_topic_excludes_matrix(self):
        result = knowledge_traversal.invoke(
            {"topic": "definition", "entity": "Reserves"}
        )
        assert "Reachability Matrix" not in result

    def test_formula_topic_excludes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "formula"})
        assert "Reachability Matrix" not in result

    def test_hierarchy_topic_excludes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "hierarchy"})
        assert "Reachability Matrix" not in result

    def test_entity_topic_excludes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "entity", "entity": "GROOVY"})
        assert "Reachability Matrix" not in result

    def test_document_topic_excludes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "document", "entity": "GROOVY"})
        assert "Reachability Matrix" not in result

    def test_commercial_topic_excludes_matrix(self):
        result = knowledge_traversal.invoke({"topic": "commercial"})
        assert "Reachability Matrix" not in result


class TestE3ReachabilityRegression:
    """Regression tests for the original E3→E4 hallucination bug."""

    def test_e3_targets_do_not_include_e4(self):
        result = knowledge_traversal.invoke({"topic": "transition", "entity": "E3"})
        for line in result.split("\n"):
            if "E3 →" in line:
                assert "E4" not in line, f"E3 should not list E4: {line}"
                break

    def test_e3_targets_include_e0_e2_e5(self):
        result = knowledge_traversal.invoke({"topic": "transition", "entity": "E3"})
        for line in result.split("\n"):
            if "E3 →" in line:
                assert "E0" in line
                assert "E2" in line
                assert "E5" in line
                break

    def test_level_topic_e3_also_shows_correct_targets(self):
        result = knowledge_traversal.invoke({"topic": "level", "entity": "E3"})
        assert "Reachability Matrix" in result
        for line in result.split("\n"):
            if "E3 →" in line:
                assert "E4" not in line
                assert "E0" in line
                assert "E2" in line
                assert "E5" in line
                break


class TestIncludeReachabilityFalse:
    """Verify the include_reachability=False flag suppresses the matrix."""

    def test_transition_with_flag_false_omits_matrix(self):
        result = knowledge_traversal.invoke(
            {
                "topic": "transition",
                "entity": "E3",
                "include_reachability": False,
            }
        )
        assert "Reachability Matrix" not in result

    def test_level_with_flag_false_omits_matrix(self):
        result = knowledge_traversal.invoke(
            {
                "topic": "level",
                "entity": "E3",
                "include_reachability": False,
            }
        )
        assert "Reachability Matrix" not in result


class TestMatrixCompleteness:
    """Verify the matrix covers all 18 levels."""

    def test_matrix_includes_all_levels(self):
        result = knowledge_traversal.invoke({"topic": "transition", "entity": "E0"})
        for level in [
            "E0",
            "E1",
            "E2",
            "E3",
            "E4",
            "E5",
            "E6",
            "E7",
            "E8",
            "X0",
            "X1",
            "X2",
            "X3",
            "X4",
            "X5",
            "X6",
            "A1",
            "A2",
        ]:
            assert f"{level} →" in result, f"Missing level {level}"

    def test_e1_can_transition_to_e4(self):
        """E1 → E4 is valid (Production on Hold → Production Pending)."""
        result = knowledge_traversal.invoke({"topic": "transition", "entity": "E1"})
        for line in result.split("\n"):
            if "E1 →" in line:
                assert "E4" in line
                break

    def test_e7_can_transition_to_e4(self):
        """E7 → E4 is valid (Prod Not Viable → Production Pending)."""
        result = knowledge_traversal.invoke({"topic": "transition", "entity": "E7"})
        for line in result.split("\n"):
            if "E7 →" in line:
                assert "E4" in line
                break


class TestHighlightedEntity:
    """Verify highlight marker behavior."""

    def test_highlight_for_different_entities(self):
        for code in ["E0", "E5", "X1", "A1"]:
            result = knowledge_traversal.invoke({"topic": "transition", "entity": code})
            highlighted = [line for line in result.split("\n") if ">>>" in line]
            assert len(highlighted) == 1
            assert f"{code} →" in highlighted[0]


class TestRelationshipWithMatrix:
    """Verify matrix injection works when relationship is also provided."""

    def test_relationship_transition_topic_includes_matrix(self):
        result = knowledge_traversal.invoke(
            {
                "topic": "transition",
                "entity": "E3",
                "relationship": "CAN_TRANSITION_TO",
            }
        )
        assert "Reachability Matrix" in result
        assert ">>>" in result
        assert "E3 →" in result
        for line in result.split("\n"):
            if "E3 →" in line and ">>>" in line:
                assert "E4" not in line
                break

    def test_relationship_level_topic_includes_matrix(self):
        result = knowledge_traversal.invoke(
            {
                "topic": "level",
                "entity": "E3",
                "relationship": "CAN_TRANSITION_TO",
            }
        )
        assert "Reachability Matrix" in result

    def test_relationship_with_flag_false_omits_matrix(self):
        result = knowledge_traversal.invoke(
            {
                "topic": "transition",
                "entity": "E3",
                "relationship": "CAN_TRANSITION_TO",
                "include_reachability": False,
            }
        )
        assert "Reachability Matrix" not in result
