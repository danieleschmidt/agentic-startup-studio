#!/usr/bin/env python3
"""
Code Quality Monitoring System
Continuously monitors and reports on code quality metrics.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import re


class CodeQualityMonitor:
    """Automated code quality monitoring and analysis."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.timestamp = datetime.now().isoformat()
        self.quality_report = {}
        
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Perform comprehensive code quality analysis."""
        print("🔍 Analyzing code quality metrics...")
        
        self.quality_report = {
            "timestamp": self.timestamp,
            "repository_stats": self._get_repository_stats(),
            "python_analysis": self._analyze_python_code(),
            "documentation_quality": self._analyze_documentation(),
            "test_coverage": self._analyze_test_coverage(),
            "complexity_analysis": self._analyze_complexity(),
            "security_analysis": self._analyze_security_patterns(),
            "maintainability": self._analyze_maintainability(),
            "code_style": self._analyze_code_style()
        }
        
        # Calculate overall quality score
        self.quality_report["overall_score"] = self._calculate_quality_score()
        self.quality_report["recommendations"] = self._generate_recommendations()
        
        return self.quality_report
    
    def _get_repository_stats(self) -> Dict[str, Any]:
        """Get basic repository statistics."""
        stats = {}
        
        try:
            # Count different file types
            file_types = {
                ".py": "python_files",
                ".js": "javascript_files", 
                ".ts": "typescript_files",
                ".md": "markdown_files",
                ".yml": "yaml_files",
                ".json": "json_files"
            }
            
            for ext, key in file_types.items():
                files = list(self.repo_path.rglob(f"*{ext}"))
                stats[key] = len(files)
            
            # Total lines of code
            python_files = list(self.repo_path.rglob("*.py"))
            total_lines = 0
            for file in python_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except UnicodeDecodeError:
                    continue
            
            stats["total_lines_of_code"] = total_lines
            stats["average_file_size"] = total_lines / max(stats["python_files"], 1)
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def _analyze_python_code(self) -> Dict[str, Any]:
        """Analyze Python code quality."""
        analysis = {
            "files_analyzed": 0,
            "syntax_errors": [],
            "import_analysis": {},
            "function_analysis": {},
            "class_analysis": {},
            "issues": []
        }
        
        python_files = list(self.repo_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                try:
                    tree = ast.parse(content)
                    analysis["files_analyzed"] += 1
                    
                    # Analyze imports
                    imports = self._extract_imports(tree)
                    analysis["import_analysis"][str(file_path)] = imports
                    
                    # Analyze functions and classes
                    functions, classes = self._extract_functions_and_classes(tree)
                    analysis["function_analysis"][str(file_path)] = functions
                    analysis["class_analysis"][str(file_path)] = classes
                    
                except SyntaxError as e:
                    analysis["syntax_errors"].append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                analysis["issues"].append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        return analysis
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract import information from AST."""
        imports = {"standard": [], "third_party": [], "local": []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    if self._is_standard_library(module_name):
                        imports["standard"].append(module_name)
                    elif self._is_local_import(module_name):
                        imports["local"].append(module_name)
                    else:
                        imports["third_party"].append(module_name)
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if self._is_standard_library(module_name):
                    imports["standard"].append(module_name)
                elif self._is_local_import(module_name):
                    imports["local"].append(module_name)
                else:
                    imports["third_party"].append(module_name)
        
        return imports
    
    def _extract_functions_and_classes(self, tree: ast.AST) -> tuple:
        """Extract function and class information from AST."""
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args_count": len(node.args.args),
                    "has_docstring": ast.get_docstring(node) is not None,
                    "is_private": node.name.startswith("_"),
                    "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                })
            
            elif isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "methods_count": len(methods),
                    "has_docstring": ast.get_docstring(node) is not None,
                    "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    "decorators": [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                })
        
        return functions, classes
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if module is part of Python standard library."""
        standard_modules = {
            'os', 'sys', 'json', 'datetime', 'pathlib', 'subprocess', 're', 
            'collections', 'itertools', 'functools', 'typing', 'dataclasses',
            'unittest', 'pytest', 'logging', 'argparse', 'configparser'
        }
        return module_name.split('.')[0] in standard_modules
    
    def _is_local_import(self, module_name: str) -> bool:
        """Check if module is a local import."""
        if not module_name:
            return True
        return module_name.startswith('.') or not ('.' in module_name and module_name.split('.')[0] not in ['src', 'app', 'core', 'utils'])
    
    def _analyze_documentation(self) -> Dict[str, Any]:
        """Analyze documentation quality."""
        analysis = {
            "markdown_files": 0,
            "readme_exists": False,
            "changelog_exists": False,
            "api_docs_exists": False,
            "docstring_coverage": 0,
            "documentation_issues": []
        }
        
        # Check for key documentation files
        key_docs = {
            "README.md": "readme_exists",
            "CHANGELOG.md": "changelog_exists",
            "docs/api.md": "api_docs_exists"
        }
        
        for file_name, key in key_docs.items():
            file_path = self.repo_path / file_name
            analysis[key] = file_path.exists()
        
        # Count markdown files
        md_files = list(self.repo_path.rglob("*.md"))
        analysis["markdown_files"] = len(md_files)
        
        # Analyze docstring coverage
        python_files = list(self.repo_path.rglob("*.py"))
        total_functions = 0
        documented_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if total_functions > 0:
            analysis["docstring_coverage"] = (documented_functions / total_functions) * 100
        
        return analysis
    
    def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage."""
        analysis = {
            "test_files": 0,
            "has_pytest_config": False,
            "has_coverage_config": False,
            "test_patterns_found": []
        }
        
        # Count test files
        test_patterns = ["**/test_*.py", "**/tests/*.py", "**/*_test.py"]
        test_files = set()
        
        for pattern in test_patterns:
            files = list(self.repo_path.rglob(pattern))
            if files:
                analysis["test_patterns_found"].append(pattern)
                test_files.update(files)
        
        analysis["test_files"] = len(test_files)
        
        # Check for test configuration
        config_files = {
            "pytest.ini": "has_pytest_config",
            ".coveragerc": "has_coverage_config",
            "pyproject.toml": "has_pytest_config"  # might contain pytest config
        }
        
        for file_name, key in config_files.items():
            file_path = self.repo_path / file_name
            if file_path.exists():
                analysis[key] = True
        
        return analysis
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity."""
        analysis = {
            "high_complexity_functions": [],
            "average_complexity": 0,
            "max_complexity": 0
        }
        
        # Simple complexity analysis based on control structures
        python_files = list(self.repo_path.rglob("*.py"))
        complexities = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        complexities.append(complexity)
                        
                        if complexity > 10:  # High complexity threshold
                            analysis["high_complexity_functions"].append({
                                "file": str(file_path),
                                "function": node.name,
                                "complexity": complexity,
                                "line": node.lineno
                            })
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if complexities:
            analysis["average_complexity"] = sum(complexities) / len(complexities)
            analysis["max_complexity"] = max(complexities)
        
        return analysis
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity
    
    def _analyze_security_patterns(self) -> Dict[str, Any]:
        """Analyze for common security anti-patterns."""
        analysis = {
            "potential_issues": [],
            "hardcoded_secrets": [],
            "sql_injection_risks": [],
            "xss_risks": []
        }
        
        python_files = list(self.repo_path.rglob("*.py"))
        
        # Security patterns to look for
        security_patterns = {
            r'password\s*=\s*["\'][^"\']+["\']': "hardcoded_password",
            r'api_key\s*=\s*["\'][^"\']+["\']': "hardcoded_api_key",
            r'secret\s*=\s*["\'][^"\']+["\']': "hardcoded_secret",
            r'\.execute\s*\([^)]*%': "sql_injection_risk",
            r'eval\s*\(': "eval_usage",
            r'exec\s*\(': "exec_usage"
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, issue_type in security_patterns.items():
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        analysis["potential_issues"].append({
                            "file": str(file_path),
                            "line": line_num,
                            "type": issue_type,
                            "pattern": match.group()
                        })
                        
            except UnicodeDecodeError:
                continue
        
        return analysis
    
    def _analyze_maintainability(self) -> Dict[str, Any]:
        """Analyze code maintainability factors."""
        analysis = {
            "large_files": [],
            "long_functions": [],
            "god_classes": [],
            "duplicate_code_blocks": []
        }
        
        python_files = list(self.repo_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check for large files (>500 lines)
                if len(lines) > 500:
                    analysis["large_files"].append({
                        "file": str(file_path),
                        "lines": len(lines)
                    })
                
                # Parse and analyze functions/classes
                try:
                    tree = ast.parse(''.join(lines))
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for long functions (>50 lines)
                            if hasattr(node, 'end_lineno') and node.end_lineno:
                                func_length = node.end_lineno - node.lineno
                                if func_length > 50:
                                    analysis["long_functions"].append({
                                        "file": str(file_path),
                                        "function": node.name,
                                        "lines": func_length,
                                        "start_line": node.lineno
                                    })
                        
                        elif isinstance(node, ast.ClassDef):
                            # Check for god classes (>20 methods)
                            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                            if len(methods) > 20:
                                analysis["god_classes"].append({
                                    "file": str(file_path),
                                    "class": node.name,
                                    "methods": len(methods),
                                    "line": node.lineno
                                })
                
                except SyntaxError:
                    continue
                    
            except UnicodeDecodeError:
                continue
        
        return analysis
    
    def _analyze_code_style(self) -> Dict[str, Any]:
        """Analyze code style compliance."""
        analysis = {
            "style_violations": [],
            "naming_issues": [],
            "formatting_issues": []
        }
        
        # Check for common style issues using simple pattern matching
        python_files = list(self.repo_path.rglob("*.py"))
        
        style_patterns = {
            r'^\s*#.*TODO': "todo_comment",
            r'^\s*#.*FIXME': "fixme_comment", 
            r'^\s*#.*HACK': "hack_comment",
            r'print\s*\(': "print_statement",
            r'import \*': "wildcard_import"
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, issue_type in style_patterns.items():
                        if re.search(pattern, line):
                            analysis["style_violations"].append({
                                "file": str(file_path),
                                "line": line_num,
                                "type": issue_type,
                                "content": line.strip()
                            })
                            
            except UnicodeDecodeError:
                continue
        
        return analysis
    
    def _calculate_quality_score(self) -> Dict[str, float]:
        """Calculate overall quality score based on various metrics."""
        scores = {}
        
        # Documentation score (0-100)
        doc_analysis = self.quality_report["documentation_quality"]
        doc_score = 0
        if doc_analysis["readme_exists"]:
            doc_score += 30
        if doc_analysis["changelog_exists"]:
            doc_score += 20
        doc_score += min(doc_analysis["docstring_coverage"], 50)  # Max 50 points
        scores["documentation"] = doc_score
        
        # Test coverage score (0-100)
        test_analysis = self.quality_report["test_coverage"]
        test_score = min(test_analysis["test_files"] * 10, 50)  # 10 points per test file, max 50
        if test_analysis["has_pytest_config"]:
            test_score += 25
        if test_analysis["has_coverage_config"]:
            test_score += 25
        scores["testing"] = test_score
        
        # Complexity score (0-100, inverted - lower complexity is better)
        complexity_analysis = self.quality_report["complexity_analysis"]
        if complexity_analysis["average_complexity"] > 0:
            complexity_score = max(0, 100 - (complexity_analysis["average_complexity"] * 5))
        else:
            complexity_score = 100
        scores["complexity"] = complexity_score
        
        # Security score (0-100, inverted - fewer issues is better)
        security_analysis = self.quality_report["security_analysis"]
        security_issues = len(security_analysis["potential_issues"])
        security_score = max(0, 100 - (security_issues * 10))
        scores["security"] = security_score
        
        # Maintainability score (0-100, inverted - fewer issues is better)
        maint_analysis = self.quality_report["maintainability"]
        maint_issues = (len(maint_analysis["large_files"]) + 
                       len(maint_analysis["long_functions"]) + 
                       len(maint_analysis["god_classes"]))
        maint_score = max(0, 100 - (maint_issues * 5))
        scores["maintainability"] = maint_score
        
        # Overall score (weighted average)
        weights = {
            "documentation": 0.2,
            "testing": 0.25,
            "complexity": 0.2,
            "security": 0.2,
            "maintainability": 0.15
        }
        
        overall = sum(scores[category] * weights[category] for category in scores)
        scores["overall"] = overall
        
        return scores
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on analysis."""
        recommendations = []
        
        scores = self.quality_report["overall_score"]
        
        # Documentation recommendations
        if scores["documentation"] < 70:
            recommendations.append({
                "category": "documentation",
                "priority": "high",
                "title": "Improve Documentation Coverage",
                "description": "Add missing documentation files and increase docstring coverage",
                "actions": [
                    "Add README.md if missing",
                    "Create CHANGELOG.md",
                    "Add docstrings to functions and classes",
                    "Create API documentation"
                ]
            })
        
        # Testing recommendations
        if scores["testing"] < 70:
            recommendations.append({
                "category": "testing",
                "priority": "high",
                "title": "Enhance Test Coverage",
                "description": "Increase test coverage and improve testing infrastructure",
                "actions": [
                    "Write more unit tests",
                    "Add pytest configuration",
                    "Set up coverage reporting",
                    "Add integration tests"
                ]
            })
        
        # Complexity recommendations
        if scores["complexity"] < 70:
            complexity_analysis = self.quality_report["complexity_analysis"]
            if complexity_analysis["high_complexity_functions"]:
                recommendations.append({
                    "category": "complexity",
                    "priority": "medium",
                    "title": "Reduce Code Complexity",
                    "description": f"Refactor {len(complexity_analysis['high_complexity_functions'])} high-complexity functions",
                    "actions": [
                        "Break down complex functions",
                        "Extract helper methods",
                        "Simplify conditional logic",
                        "Use early returns"
                    ]
                })
        
        # Security recommendations
        if scores["security"] < 80:
            security_analysis = self.quality_report["security_analysis"]
            if security_analysis["potential_issues"]:
                recommendations.append({
                    "category": "security",
                    "priority": "high",
                    "title": "Address Security Issues",
                    "description": f"Fix {len(security_analysis['potential_issues'])} potential security issues",
                    "actions": [
                        "Remove hardcoded secrets",
                        "Use parameterized queries",
                        "Avoid eval() and exec()",
                        "Add security scanning to CI"
                    ]
                })
        
        # Maintainability recommendations
        if scores["maintainability"] < 70:
            recommendations.append({
                "category": "maintainability",
                "priority": "medium",
                "title": "Improve Code Maintainability",
                "description": "Address large files, long functions, and god classes",
                "actions": [
                    "Split large files into modules",
                    "Refactor long functions",
                    "Break down large classes",
                    "Remove duplicate code"
                ]
            })
        
        return recommendations
    
    def save_report(self, output_file: str = ".github/code-quality-report.json"):
        """Save quality report to JSON file."""
        output_path = self.repo_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2)
        
        print(f"📊 Code quality report saved to {output_path}")
    
    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        report = []
        report.append("# Code Quality Report")
        report.append(f"Generated: {self.timestamp}")
        report.append("")
        
        # Overall scores
        scores = self.quality_report["overall_score"]
        report.append("## Overall Quality Scores")
        for category, score in scores.items():
            emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            report.append(f"- {category.title()}: {score:.1f}/100 {emoji}")
        report.append("")
        
        # Key metrics
        repo_stats = self.quality_report["repository_stats"]
        report.append("## Repository Statistics")
        report.append(f"- Python Files: {repo_stats.get('python_files', 0)}")
        report.append(f"- Total Lines of Code: {repo_stats.get('total_lines_of_code', 0)}")
        report.append(f"- Average File Size: {repo_stats.get('average_file_size', 0):.1f} lines")
        report.append("")
        
        # Documentation
        doc_analysis = self.quality_report["documentation_quality"]
        report.append("## Documentation")
        report.append(f"- Docstring Coverage: {doc_analysis['docstring_coverage']:.1f}%")
        report.append(f"- README Exists: {'✅' if doc_analysis['readme_exists'] else '❌'}")
        report.append(f"- Changelog Exists: {'✅' if doc_analysis['changelog_exists'] else '❌'}")
        report.append("")
        
        # Testing
        test_analysis = self.quality_report["test_coverage"]
        report.append("## Testing")
        report.append(f"- Test Files: {test_analysis['test_files']}")
        report.append(f"- Pytest Configured: {'✅' if test_analysis['has_pytest_config'] else '❌'}")
        report.append("")
        
        # Issues summary
        complexity_analysis = self.quality_report["complexity_analysis"]
        security_analysis = self.quality_report["security_analysis"]
        
        report.append("## Issues Summary")
        report.append(f"- High Complexity Functions: {len(complexity_analysis['high_complexity_functions'])}")
        report.append(f"- Potential Security Issues: {len(security_analysis['potential_issues'])}")
        report.append("")
        
        # Recommendations
        recommendations = self.quality_report["recommendations"]
        if recommendations:
            report.append("## Top Recommendations")
            for rec in recommendations[:3]:  # Show top 3
                report.append(f"### {rec['title']} ({rec['priority']} priority)")
                report.append(f"{rec['description']}")
                report.append("")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor and analyze code quality")
    parser.add_argument("--output", "-o", default=".github/code-quality-report.json",
                       help="Output file for quality report")
    parser.add_argument("--summary", "-s", help="Generate summary report to file")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    monitor = CodeQualityMonitor(args.repo_path)
    report = monitor.analyze_code_quality()
    
    # Save detailed report
    monitor.save_report(args.output)
    
    # Generate summary if requested
    if args.summary:
        summary = monitor.generate_summary_report()
        with open(args.summary, 'w') as f:
            f.write(summary)
        print(f"📋 Summary report saved to {args.summary}")
    
    # Print summary to console
    overall_score = report["overall_score"]["overall"]
    print(f"\n📈 Code Quality Analysis Complete")
    print(f"Overall Score: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print("🟢 Excellent code quality!")
    elif overall_score >= 60:
        print("🟡 Good code quality with room for improvement")
    else:
        print("🔴 Code quality needs significant improvement")
    
    # Show key metrics
    scores = report["overall_score"]
    print(f"\nCategory Scores:")
    for category, score in scores.items():
        if category != "overall":
            print(f"  {category.title()}: {score:.1f}/100")


if __name__ == "__main__":
    main()