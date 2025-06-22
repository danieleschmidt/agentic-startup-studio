import json

from click.testing import CliRunner

from scripts import run_health_checks


def test_results_dir_env(tmp_path, monkeypatch):
    results_file = tmp_path / "results.json"
    monkeypatch.setenv("HEALTH_CHECK_RESULTS_FILE", str(results_file))

    runner = CliRunner()
    result = runner.invoke(run_health_checks.main, [])
    assert result.exit_code == 0

    assert results_file.exists()
    data = json.loads(results_file.read_text())
    assert "overall_status" in data


def test_results_file_cli(tmp_path):
    results_file = tmp_path / "cli_results.json"
    runner = CliRunner()
    result = runner.invoke(run_health_checks.main, ["--results-file", str(results_file)])
    assert result.exit_code == 0
    assert results_file.exists()
    data = json.loads(results_file.read_text())
    assert "overall_status" in data
