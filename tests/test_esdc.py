from unittest.mock import patch, MagicMock

import pytest

from esdc.esdc import (
    _read_csv,
    esdc_url_builder,
    _load_file_as_csv,
    _load_file_as_json,
    esdc_downloader,
)
from esdc.selection import TableName, ApiVer, FileType


class TestEsdcUrlBuilder:
    """Tests for esdc_url_builder()."""

    def test_url_builder_default(self):
        """Test URL builder with default parameters."""
        with patch(
            "esdc.esdc.Config.get_api_url", return_value="https://esdc.skkmigas.go.id/"
        ):
            url = esdc_url_builder(TableName.PROJECT_RESOURCES)
            assert "esdc.skkmigas.go.id" in url
            assert "project-resources" in url
            assert "api/v2" in url
            assert "verbose=3" in url

    def test_url_builder_with_year(self):
        """Test URL builder with report year."""
        with patch(
            "esdc.esdc.Config.get_api_url", return_value="https://esdc.skkmigas.go.id/"
        ):
            url = esdc_url_builder(TableName.PROJECT_RESOURCES, report_year=2024)
            assert "report-year=2024" in url

    def test_url_builder_json(self):
        """Test URL builder with JSON file type."""
        with patch(
            "esdc.esdc.Config.get_api_url", return_value="https://esdc.skkmigas.go.id/"
        ):
            url = esdc_url_builder(TableName.PROJECT_RESOURCES, file_type=FileType.JSON)
            assert "output=json" in url

    def test_url_builder_csv(self):
        """Test URL builder with CSV file type."""
        with patch(
            "esdc.esdc.Config.get_api_url", return_value="https://esdc.skkmigas.go.id/"
        ):
            url = esdc_url_builder(TableName.PROJECT_RESOURCES, file_type=FileType.CSV)
            assert "output=csv" in url

    def test_url_builder_timeseries(self):
        """Test URL builder for timeseries table."""
        with patch(
            "esdc.esdc.Config.get_api_url", return_value="https://esdc.skkmigas.go.id/"
        ):
            url = esdc_url_builder(TableName.PROJECT_TIMESERIES)
            assert "project-timeseries" in url


class TestReadCsv:
    """Tests for _read_csv()."""

    def test_read_csv_with_stringio(self):
        """Test reading CSV from StringIO."""
        csv_data = """id;name;value
1;test;100
2;example;200
"""
        result_data, result_header = _read_csv(csv_data.splitlines())
        assert result_header == ["id", "name", "value"]
        assert result_data == [["1", "test", "100"], ["2", "example", "200"]]

    def test_read_csv_custom_dialect(self):
        """Test CSV uses custom semicolon delimiter."""
        csv_data = """a;b;c
1;2;3
"""
        result_data, result_header = _read_csv(csv_data.splitlines())
        assert result_header == ["a", "b", "c"]
        assert result_data == [["1", "2", "3"]]

    def test_read_csv_quoted_fields(self):
        """Test CSV handles quoted fields with semicolons."""
        csv_data = """id;description
1;"field;with;semicolons"
2;normal
"""
        result_data, result_header = _read_csv(csv_data.splitlines())
        assert result_header == ["id", "description"]
        assert result_data[0] == ["1", "field;with;semicolons"]
        assert result_data[1] == ["2", "normal"]

    def test_read_csv_empty(self):
        """Test reading empty CSV."""
        csv_data = """a;b;c
"""
        result_data, result_header = _read_csv(csv_data.splitlines())
        assert result_header == ["a", "b", "c"]
        assert result_data == []


class TestLoadFileAsCsv:
    """Tests for _load_file_as_csv()."""

    def test_load_file_as_csv(self, tmp_path):
        """Test loading CSV file into database."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id;name\n1;test\n")

        with patch("esdc.esdc.load_data_to_db") as mock_load:
            _load_file_as_csv(str(csv_file), "test_table")
            mock_load.assert_called_once()
            args = mock_load.call_args[0]
            assert args[1] == ["id", "name"]


class TestLoadFileAsJson:
    """Tests for _load_file_as_json()."""

    def test_load_file_as_json(self, tmp_path):
        """Test loading JSON file into database."""
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"id": 1, "name": "test"}]')

        with patch("esdc.esdc.load_data_to_db") as mock_load:
            _load_file_as_json(str(json_file), "test_table")
            mock_load.assert_called_once()
            args = mock_load.call_args[0]
            assert list(args[1]) == ["id", "name"]
            assert args[2] == "test_table"

    def test_load_file_as_json_empty(self, tmp_path):
        """Test loading empty JSON file returns early without crashing."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("[]")

        with patch("esdc.esdc.load_data_to_db") as mock_load:
            result = _load_file_as_json(str(json_file), "test_table")
            mock_load.assert_not_called()  # Should not call load_data_to_db


class TestEsdcDownloader:
    """Tests for esdc_downloader (mocked)."""

    def test_downloader_success(self, mocker):
        """Test successful download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": "100", "Content-Encoding": None}
        mock_response.iter_content = lambda chunk_size: [b"test data"]

        mocker.patch("requests.get", return_value=mock_response)

        result = esdc_downloader("https://example.com/file", "user", "pass")
        assert result is not None

    def test_downloader_failure(self, mocker):
        """Test failed download returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mocker.patch("requests.get", return_value=mock_response)

        result = esdc_downloader("https://example.com/file", "user", "pass")
        assert result is None

    def test_downloader_exception(self, mocker):
        """Test download with request exception."""
        import requests

        mocker.patch(
            "requests.get", side_effect=requests.RequestException("Connection error")
        )

        result = esdc_downloader("https://example.com/file", "user", "pass")
        assert result is None
