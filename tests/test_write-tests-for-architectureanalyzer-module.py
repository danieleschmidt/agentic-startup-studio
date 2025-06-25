from pathlib import Path


from agentic_startup_studio.arch_review import ArchitectureAnalyzer


def create_pkg(tmp_path: Path, files: dict[str, str]) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for name, content in files.items():
        (pkg / name).write_text(content)
    return pkg


def test_generate_markdown_success(tmp_path: Path):
    create_pkg(tmp_path, {"mod.py": "class A:\n    pass\n"})
    analyzer = ArchitectureAnalyzer(tmp_path, ["pkg"])
    modules = analyzer.analyze()
    markdown = analyzer.generate_markdown(modules)
    assert "# Architecture Report" in markdown
    assert "pkg/mod.py" in markdown


def test_handles_unparsable_file(tmp_path: Path):
    create_pkg(
        tmp_path,
        {
            "good.py": "def func():\n    return 1\n",
            "bad.py": "def invalid",
        },
    )
    analyzer = ArchitectureAnalyzer(tmp_path, ["pkg"])
    modules = analyzer.analyze()
    paths = [str(m.path) for m in modules]
    assert "pkg/good.py" in paths
    assert "pkg/bad.py" not in paths
