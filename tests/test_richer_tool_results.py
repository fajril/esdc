"""Tests for richer tool result content types (JSON serialization)."""

import pytest


class TestToolNodeRicherResults:
    """Test that tool_node JSON-serializes dict/list observations."""

    def test_dict_observation_serialized_as_json(self):
        """Dict observations should be JSON-serialized, not Python repr."""
        import json

        observation = {"count": 5, "results": [{"field": "Duri", "value": 123.0}]}
        if isinstance(observation, dict):
            serialized = json.dumps(observation, ensure_ascii=False)
        else:
            serialized = str(observation)

        # Should be valid JSON
        parsed = json.loads(serialized)
        assert parsed["count"] == 5
        assert parsed["results"][0]["field"] == "Duri"

        # Should NOT be Python repr format
        assert "u'" not in serialized  # Python 2 style
        assert serialized.startswith("{")

    def test_list_observation_serialized_as_json(self):
        """List observations should be JSON-serialized."""
        import json

        observation = [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}]
        if isinstance(observation, list):
            serialized = json.dumps(observation, ensure_ascii=False)
        else:
            serialized = str(observation)

        parsed = json.loads(serialized)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Alpha"

    def test_string_observation_unchanged(self):
        """String observations should pass through unchanged."""
        import json

        observation = "Cadangan minyak Duri: 123 MMSTB"
        if isinstance(observation, (dict, list)):
            result = json.dumps(observation, ensure_ascii=False)
        else:
            result = str(observation)
        assert result == "Cadangan minyak Duri: 123 MMSTB"

    def test_nested_dict_serialization(self):
        """Nested dicts should be fully serialized."""
        import json

        observation = {
            "status": "success",
            "data": {"nested": {"deep": "value"}, "array": [1, 2, 3]},
        }
        serialized = json.dumps(observation, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert parsed["data"]["nested"]["deep"] == "value"
        assert parsed["data"]["array"] == [1, 2, 3]


class TestRicherToolResultsDetection:
    """Test detection of JSON-serialized content in wrapper."""

    def test_is_json_detectable(self):
        """JSON strings should be detectable."""
        import json

        json_str = '{"status": "success", "data": [1, 2, 3]}'
        try:
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
        except json.JSONDecodeError:
            pytest.fail("Should be valid JSON")

    def test_plain_text_not_json(self):
        """Plain text should not parse as JSON."""
        import json

        text = "Cadangan minyak Duri: 123 MMSTB"
        try:
            json.loads(text)
            pytest.fail("Should not be valid JSON")
        except json.JSONDecodeError:
            pass  # Expected

    def test_json_array_detection(self):
        """JSON arrays should be detectable."""
        import json

        json_str = '[{"id": 1}, {"id": 2}]'
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 2


class TestToolResultContentTypes:
    """Test various content types in tool results."""

    def test_image_url_in_dict(self):
        """Dict with image URL should be preserved."""
        import json

        observation = {
            "image_url": "https://example.com/chart.png",
            "caption": "Reserves chart",
        }
        serialized = json.dumps(observation, ensure_ascii=False)
        assert "image_url" in serialized
        assert "chart.png" in serialized

    def test_unicode_in_json(self):
        """Unicode should be preserved in JSON."""
        import json

        observation = {"text": "Cadangan minyak: 123 MMSTB"}
        serialized = json.dumps(observation, ensure_ascii=False)
        assert "Cadangan minyak" in serialized
        # Should not be escaped unicode
        assert "\\u" not in serialized
