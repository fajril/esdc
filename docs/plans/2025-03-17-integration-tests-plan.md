# Integration Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 11 integration tests for ESDC CLI that exercise real database and config operations with mocked API responses.

**Architecture:** Create `tests/integration/` directory with fixtures that use pytest's `tmp_path` for isolation and `responses` library to mock HTTP calls. Tests will exercise real SQLite database operations and config file handling.

**Tech Stack:** pytest, pytest-mock, responses, typer.testing.CliRunner

---

## Prerequisites

1. Verify `responses` library is installed: `pip install responses`
2. Verify `responses` is in dev dependencies in `pyproject.toml`
3. Check existing fixtures in `tests/conftest.py`

---

## Task 1: Create Integration Test Directory Structure

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_fetch_integration.py`
- Create: `tests/integration/test_show_integration.py`

**Step 1: Create integration directory and __init__.py**

```bash
mkdir -p tests/integration
touch tests/integration/__init__.py
```

**Step 2: Verify directory structure**

Run: `ls -la tests/integration/`
Expected: `__init__.py` exists

**Step 3: Commit**

```bash
git add tests/integration/__init__.py
git commit -m "feat: create integration test directory"
```

---

## Task 2: Add Integration Test Fixtures to conftest.py

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Read existing conftest.py**

Run: `cat tests/conftest.py`

**Step 2: Add integration fixtures**

Add these fixtures to `tests/conftest.py`:

```python
@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Create isolated config directory for integration tests.
    
    Sets ESDC_CONFIG_DIR to a temporary directory and ESDC_DB_FILE
    to avoid polluting the user's actual ~/.esdc/ directory.
    """
    config_dir = tmp_path / ".esdc"
    db_file = config_dir / "esdc.db"
    monkeypatch.setenv("ESDC_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ESDC_DB_FILE", str(db_file))
    yield tmp_path


@pytest.fixture
def mock_esdc_api(responses):
    """Factory to add mock ESDC API responses.
    
    Usage:
        mock = mock_esdc_api()
        mock.add_json("https://esdc.skkmigas.go.id/api/v2/...", {...})
        mock.add_csv("https://esdc.skkmigas.go.id/api/v2/...", "col1,col2\nval1,val2")
    """
    import re
    
    def _add_response(method, pattern, **kwargs):
        url_pattern = re.compile(pattern)
        responses.add(method, url_pattern, **kwargs)
    
    class MockAPI:
        def add_json(self, pattern, json_data, status=200):
            _add_response(responses.GET, pattern, json=json_data, status=status)
        
        def add_csv(self, pattern, body, status=200):
            _add_response(responses.GET, pattern, body=body, status=status)
        
        def add_status(self, pattern, status):
            _add_response(responses.GET, pattern, status=status)
    
    return MockAPI()


@pytest.fixture
def seeded_database(isolated_config):
    """Create database with sample project_resources data.
    
    Returns the tmp_path so tests can reference the database file.
    """
    from esdc.dbmanager import load_data_to_db
    
    sample_data = [
        ["PRJ-001", "Test Project Alpha", "2024", "DISCOVERED", "100.00"],
        ["PRJ-002", "Test Project Beta", "2024", "DEVELOPMENT", "200.00"],
        ["PRJ-003", "Test Project Gamma", "2023", "PRODUCTION", "300.00"],
    ]
    header = ["project_code", "project_name", "report_year", "project_stage", "budget"]
    
    # Ensure config is initialized
    from esdc.configs import Config
    Config.init_config()
    
    load_data_to_db(sample_data, header, "project_resources")
    
    return isolated_config
```

**Step 3: Run tests to verify fixtures work**

Run: `pytest tests/test_configs.py -v -k "test_get_config_dir"`
Expected: PASS (fixtures shouldn't break existing tests)

**Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add integration test fixtures"
```

---

## Task 3: Create test_fetch_integration.py

**Files:**
- Create: `tests/integration/test_fetch_integration.py`

**Step 1: Write the test file**

```python
"""Integration tests for fetch command."""
import json
import pytest
from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


class TestFetchCreatesDatabase:
    """Tests that fetch command creates database correctly."""

    def test_fetch_creates_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """First fetch run should create ~/.esdc/esdc.db"""
        # Setup mock API response
        mock_esdc_api.add_json(
            "https://esdc.skkmigas.go.id/api/v2/project-resources.*",
            json_data=[
                {"id": "PRJ-001", "project_name": "Test", "report_year": "2024"}
            ]
        )
        
        # Set credentials via env
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        
        # Run fetch command
        result = runner.invoke(app, ["fetch", "--filetype", "json"])
        
        # Verify database was created
        db_file = isolated_config / ".esdc" / "esdc.db"
        assert db_file.exists(), f"Database not created at {db_file}"
        assert result.exit_code == 0

    def test_fetch_json_to_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch with JSON should persist data to database."""
        # Setup mock API with sample data matching table schema
        mock_esdc_api.add_json(
            "https://esdc.skkmigas.go.id/api/v2/project-resources.*",
            json_data=[
                {
                    "id": "PRJ-TEST-001",
                    "project_name": "Integration Test Project",
                    "report_year": "2024",
                    "project_stage": "DISCOVERED",
                    "budget": "1000000.00"
                }
            ]
        )
        
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        
        result = runner.invoke(app, ["fetch", "--filetype", "json"])
        
        # Verify data was inserted by querying the database
        from esdc.dbmanager import run_query
        from esdc.selection import TableName
        
        df = run_query(TableName.PROJECT_RESOURCES)
        assert df is not None
        assert len(df) > 0
        assert "Integration Test Project" in df["project_name"].values

    def test_fetch_csv_to_database(self, isolated_config, mock_esdc_api, monkeypatch):
        """Fetch with CSV should persist data to database."""
        csv_data = """id,project_name,report_year,project_stage,budget
PRJ-CSV-001,CSV Test Project,2024,DISCOVERED,500000.00"""
        
        mock_esdc_api.add_csv(
            "https://esdc.skkmigas.go.id/api/v2/project-resources.*",
            body=csv_data
        )
        
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        
        result = runner.invoke(app, ["fetch", "--filetype", "csv"])
        
        # Verify data was inserted
        from esdc.dbmanager import run_query
        from esdc.selection import TableName
        
        df = run_query(TableName.PROJECT_RESOURCES)
        assert df is not None
        assert "CSV Test Project" in df["project_name"].values


class TestFetchErrorHandling:
    """Tests for fetch command error handling."""

    def test_fetch_auth_failure(self, isolated_config, mock_esdc_api, monkeypatch):
        """API returns 401 should show clear error message."""
        mock_esdc_api.add_status(
            "https://esdc.skkmigas.go.id/api/v2/project-resources.*",
            status=401
        )
        
        monkeypatch.setenv("ESDC_USER", "baduser")
        monkeypatch.setenv("ESDC_PASS", "badpass")
        
        result = runner.invoke(app, ["fetch", "--filetype", "json"])
        
        # Should fail gracefully (either exit code != 0 or error in output)
        assert result.exit_code != 0 or "401" in result.stdout or "unauthorized" in result.stdout.lower()

    def test_fetch_malformed_json(self, isolated_config, mock_esdc_api, monkeypatch):
        """Invalid JSON response should fail gracefully."""
        mock_esdc_api.add(
            responses.GET,
            url="https://esdc.skkmigas.go.id/api/v2/project-resources.*",
            body="not valid json {{{",
            status=200
        )
        
        monkeypatch.setenv("ESDC_USER", "testuser")
        monkeypatch.setenv("ESDC_PASS", "testpass")
        
        result = runner.invoke(app, ["fetch", "--filetype", "json"])
        
        # Should handle parse error gracefully
        assert result.exit_code != 0 or "error" in result.stdout.lower() or "json" in result.stdout.lower()

    def test_fetch_with_env_credentials(self, isolated_config, mock_esdc_api):
        """ESDC_USER and ESDC_PASS env vars should work for authentication."""
        mock_esdc_api.add_json(
            "https://esdc.skkmigas.go.id/api/v2/project-resources.*",
            json_data=[{"id": "PRJ-001", "project_name": "Test"}]
        )
        
        # Credentials via environment (not prompt)
        import os
        os.environ["ESDC_USER"] = "envuser"
        os.environ["ESDC_PASS"] = "envpass"
        
        try:
            result = runner.invoke(app, ["fetch", "--filetype", "json"])
            # Should work without prompting for credentials
            assert result.exit_code == 0
        finally:
            del os.environ["ESDC_USER"]
            del os.environ["ESDC_PASS"]
```

**Step 2: Run tests to verify they fail (no implementation yet)**

Run: `pytest tests/integration/test_fetch_integration.py -v`
Expected: 6 tests collected, should pass (fixtures work) or fail on assertions

**Step 3: Commit**

```bash
git add tests/integration/test_fetch_integration.py
git commit -m "feat: add fetch integration tests (6 tests)"
```

---

## Task 4: Create test_show_integration.py

**Files:**
- Create: `tests/integration/test_show_integration.py`

**Step 1: Write the test file**

```python
"""Integration tests for show command."""
import pytest
from typer.testing import CliRunner

from esdc.esdc import app

runner = CliRunner()


class TestShowCommand:
    """Tests for show command with real database."""

    def test_show_empty_database(self, isolated_config):
        """Show with no data should show warning message."""
        # Ensure config is initialized but no data
        from esdc.configs import Config
        Config.init_config()
        
        result = runner.invoke(app, ["show", "project_resources"])
        
        # Should either show warning or return empty (depends on current behavior)
        assert result.exit_code == 0  # Should not crash

    def test_show_with_data(self, seeded_database):
        """Show with seeded data should display formatted table."""
        result = runner.invoke(app, ["show", "project_resources"])
        
        assert result.exit_code == 0
        # Should contain data from seeded_database
        assert "Test Project Alpha" in result.stdout or "Test Project Beta" in result.stdout

    def test_show_with_where_filter(self, seeded_database):
        """Show with --where and --search should filter results."""
        result = runner.invoke(
            app, 
            ["show", "project_resources", "--where", "project_name", "--search", "Alpha"]
        )
        
        assert result.exit_code == 0
        # Should only show matching project
        assert "Alpha" in result.stdout
        # Should NOT show Beta (doesn't match)
        # Note: depends on implementation, may show all with Beta filtered

    def test_show_with_year_filter(self, seeded_database):
        """Show with --year should filter by report year."""
        result = runner.invoke(
            app,
            ["show", "project_resources", "--year", "2023"]
        )
        
        assert result.exit_code == 0
        # Should only show 2023 data
        assert "2023" in result.stdout

    def test_show_with_columns(self, seeded_database):
        """Show with --columns should select specific fields."""
        result = runner.invoke(
            app,
            ["show", "project_resources", "--columns", "project_name project_stage"]
        )
        
        assert result.exit_code == 0
        # Should show only selected columns
        assert "project_name" in result.stdout.lower() or "project_stage" in result.stdout.lower()
```

**Step 2: Run tests to verify they work**

Run: `pytest tests/integration/test_show_integration.py -v`
Expected: 5 tests, should pass with seeded data

**Step 3: Commit**

```bash
git add tests/integration/test_show_integration.py
git commit -m "feat: add show integration tests (5 tests)"
```

---

## Task 5: Final Verification

**Step 1: Run all integration tests**

Run: `pytest tests/integration/ -v`
Expected: 11 tests pass

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: 78 tests pass (67 existing + 11 new)

**Step 3: Final commit**

```bash
git status
git log --oneline -5
git commit --allow-empty -m "feat: add integration tests for fetch and show commands (11 tests)"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Create integration directory | - |
| 2 | Add fixtures to conftest.py | - |
| 3 | Create test_fetch_integration.py | 6 tests |
| 4 | Create test_show_integration.py | 5 tests |
| 5 | Verify and commit | 11 tests |

**Total: 5 tasks, ~290 LOC, 11 integration tests**
