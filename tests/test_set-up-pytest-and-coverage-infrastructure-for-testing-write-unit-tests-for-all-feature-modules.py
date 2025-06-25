import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agentic_startup_studio.testing import run_pytest_with_coverage


def test_run_pytest_with_coverage_success(tmp_path):
    test_file = tmp_path / "test_success.py"
    test_file.write_text("def test_ok():\n    assert True\n")
    exit_code, cov_path = run_pytest_with_coverage(tmp_path)
    assert exit_code == 0
    assert cov_path.exists()


def test_run_pytest_with_coverage_without_coveragerc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_file = Path("test_edge.py")
    test_file.write_text("def test_ok():\n    assert True\n")
    exit_code, cov_path = run_pytest_with_coverage(Path("."))
    assert exit_code == 0
    assert cov_path.exists()
