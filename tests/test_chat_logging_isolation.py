"""Chat logging honors an isolated ESDC configuration directory."""

import os
import subprocess
import sys
from pathlib import Path


def test_chat_import_writes_log_under_isolated_config_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    env = os.environ.copy()
    env.update(
        {
            "ESDC_CONFIG_DIR": str(config_dir),
            "ESDC_DB_DIR": str(tmp_path / "database-dir"),
            "ESDC_DB_FILE": str(tmp_path / "database.duckdb"),
            "ESDC_LOG_LEVEL": "WARNING",
        }
    )
    code = (
        "import logging; "
        "from esdc.chat.app import log_file; "
        "logging.getLogger('esdc.chat').warning('isolated log check'); "
        "print(log_file)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        cwd=Path.cwd(),
        env=env,
        text=True,
    )

    expected_log = config_dir / "logs" / "esdc_chat.log"
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(expected_log)
    assert expected_log.is_file()
    assert "isolated log check" in expected_log.read_text(encoding="utf-8")
