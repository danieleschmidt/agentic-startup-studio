from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModuleInfo:
    path: Path
    classes: int
    functions: int
    lines: int


class ArchitectureAnalyzer:
    """Analyze repository structure and produce a markdown report."""

    def __init__(self, root: Path, packages: Iterable[str]) -> None:
        self.root = root
        self.packages = [pkg for pkg in packages if (root / pkg).exists()]

    def _analyze_file(self, file_path: Path) -> ModuleInfo:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = sum(isinstance(node, ast.ClassDef) for node in tree.body)
        functions = sum(isinstance(node, ast.FunctionDef) for node in tree.body)
        lines = len(source.splitlines())
        return ModuleInfo(file_path.relative_to(self.root), classes, functions, lines)

    def analyze(self) -> list[ModuleInfo]:
        modules: list[ModuleInfo] = []
        for pkg in self.packages:
            pkg_path = self.root / pkg
            for file in pkg_path.rglob("*.py"):
                if file.name == "__init__.py":
                    continue
                try:
                    modules.append(self._analyze_file(file))
                except Exception:
                    # Skip files that fail to parse
                    continue
        return modules

    def generate_markdown(self, modules: Iterable[ModuleInfo]) -> str:
        lines = ["# Architecture Report", ""]
        by_package: dict[str, list[ModuleInfo]] = {}
        for info in modules:
            pkg = info.path.parts[0]
            by_package.setdefault(pkg, []).append(info)
        for pkg, infos in by_package.items():
            lines.append(f"## {pkg}")
            lines.append("| Module | Classes | Functions | Lines |")
            lines.append("| ------ | ------- | --------- | ----- |")
            for info in sorted(infos, key=lambda i: str(i.path)):
                lines.append(
                    f"| {info.path} | {info.classes} | {info.functions} | {info.lines} |"
                )
            lines.append("")
        return "\n".join(lines)

    def write_report(self, output: Path) -> None:
        modules = self.analyze()
        markdown = self.generate_markdown(modules)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(f"Architecture report written to {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate architecture report")
    parser.add_argument(
        "--packages",
        nargs="+",
        default=["pipeline", "agents", "core", "tools", "config"],
        help="Packages to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/architecture_report.md"),
        help="Output markdown file",
    )
    args = parser.parse_args(argv)
    analyzer = ArchitectureAnalyzer(Path.cwd(), args.packages)
    analyzer.write_report(args.output)


if __name__ == "__main__":
    main()
