import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agentic_startup_studio.testing import (
    run_pytest_with_coverage,
    check_coverage_threshold,
)


def test_coverage_above_threshold(tmp_path, monkeypatch):
    mod = tmp_path / "mod.py"
    mod.write_text("def add(a, b):\n    return a + b\n")
    test_file = tmp_path / "test_mod.py"
    test_file.write_text(
        "from mod import add\n\n\n" "def test_add():\n    assert add(1, 2) == 3\n"
    )
    (tmp_path / ".coveragerc").write_text("[run]\nsource = .\n")
    monkeypatch.chdir(tmp_path)
    exit_code, cov_path = run_pytest_with_coverage(Path("."))
    assert exit_code == 0
    percentage = check_coverage_threshold(cov_path, 80.0)
    assert percentage >= 80.0


def test_coverage_below_threshold_fails(tmp_path, monkeypatch):
    mod = tmp_path / "mod.py"
    mod.write_text("def add(a, b):\n    return a + b\n")
    test_file = tmp_path / "test_placeholder.py"
    test_file.write_text("def test_placeholder():\n    assert True\n")
    (tmp_path / ".coveragerc").write_text("[run]\nsource = .\n")
    monkeypatch.chdir(tmp_path)
    exit_code, cov_path = run_pytest_with_coverage(Path("."))
    assert exit_code == 0
    with pytest.raises(ValueError):
        check_coverage_threshold(cov_path, 80.0)
