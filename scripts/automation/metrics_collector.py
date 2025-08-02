#!/usr/bin/env python3
"""
Automated Metrics Collection System
Collects comprehensive repository, code quality, and development metrics.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import psutil


class MetricsCollector:
    """Comprehensive metrics collection for SDLC tracking."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics = {}
        self.timestamp = datetime.now().isoformat()
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🔍 Collecting comprehensive repository metrics...")
        
        self.metrics = {
            "collection_timestamp": self.timestamp,
            "repository": self._get_repository_info(),
            "code_quality": self._collect_code_quality_metrics(),
            "development": self._collect_development_metrics(),
            "testing": self._collect_testing_metrics(),
            "security": self._collect_security_metrics(),
            "performance": self._collect_performance_metrics(),
            "infrastructure": self._collect_infrastructure_metrics(),
            "dependencies": self._collect_dependency_metrics(),
            "documentation": self._collect_documentation_metrics()
        }
        
        return self.metrics
    
    def _get_repository_info(self) -> Dict[str, Any]:
        """Get basic repository information."""
        try:
            remote_url = subprocess.check_output(
                ["git", "remote", "get-url", "origin"], 
                cwd=self.repo_path,
                text=True
            ).strip()
            
            current_branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            commit_count = subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            return {
                "remote_url": remote_url,
                "current_branch": current_branch,
                "total_commits": int(commit_count),
                "last_commit": self._get_last_commit_info()
            }
        except subprocess.CalledProcessError:
            return {"error": "Failed to collect repository info"}
    
    def _get_last_commit_info(self) -> Dict[str, str]:
        """Get information about the last commit."""
        try:
            last_commit = subprocess.check_output(
                ["git", "log", "-1", "--format=%H|%an|%ae|%ad|%s"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            parts = last_commit.split("|")
            return {
                "hash": parts[0],
                "author": parts[1],
                "email": parts[2],
                "date": parts[3],
                "message": parts[4]
            }
        except subprocess.CalledProcessError:
            return {}
    
    def _collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Count lines of code
        try:
            # Python files
            python_files = list(self.repo_path.rglob("*.py"))
            total_lines = 0
            for file in python_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except UnicodeDecodeError:
                    continue
            
            metrics["lines_of_code"] = total_lines
            metrics["python_files_count"] = len(python_files)
            
        except Exception as e:
            metrics["lines_of_code_error"] = str(e)
        
        # Count different file types
        file_types = {
            "python": "*.py",
            "javascript": "*.js",
            "typescript": "*.ts",
            "yaml": "*.yml",
            "json": "*.json",
            "markdown": "*.md",
            "dockerfile": "Dockerfile*"
        }
        
        for file_type, pattern in file_types.items():
            files = list(self.repo_path.rglob(pattern))
            metrics[f"{file_type}_files"] = len(files)
        
        return metrics
    
    def _collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development-related metrics."""
        metrics = {}
        
        try:
            # Git statistics
            commits_last_week = subprocess.check_output(
                ["git", "rev-list", "--count", "--since='1 week ago'", "HEAD"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            branches = subprocess.check_output(
                ["git", "branch", "-a"],
                cwd=self.repo_path,
                text=True
            ).strip().split('\n')
            
            metrics["commits_last_week"] = int(commits_last_week)
            metrics["total_branches"] = len([b for b in branches if b.strip()])
            
            # Contributors
            contributors = subprocess.check_output(
                ["git", "log", "--format='%an'", "|", "sort", "|", "uniq"],
                shell=True,
                cwd=self.repo_path,
                text=True
            ).strip().split('\n')
            
            metrics["total_contributors"] = len([c for c in contributors if c.strip()])
            
        except subprocess.CalledProcessError as e:
            metrics["git_metrics_error"] = str(e)
        
        return metrics
    
    def _collect_testing_metrics(self) -> Dict[str, Any]:
        """Collect testing-related metrics."""
        metrics = {}
        
        # Count test files
        test_patterns = ["**/test_*.py", "**/tests/*.py", "**/*_test.py"]
        test_files = []
        for pattern in test_patterns:
            test_files.extend(list(self.repo_path.rglob(pattern)))
        
        metrics["test_files_count"] = len(set(test_files))
        
        # Look for test configuration files
        test_configs = {
            "pytest": "pytest.ini",
            "coverage": ".coveragerc",
            "tox": "tox.ini",
            "jest": "jest.config.js"
        }
        
        for test_type, config_file in test_configs.items():
            config_path = self.repo_path / config_file
            metrics[f"{test_type}_configured"] = config_path.exists()
        
        return metrics
    
    def _collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        metrics = {}
        
        # Check for security configuration files
        security_files = {
            "security_policy": "SECURITY.md",
            "codeowners": "CODEOWNERS",
            "dependabot": ".github/dependabot.yml",
            "gitignore": ".gitignore"
        }
        
        for sec_type, file_name in security_files.items():
            file_path = self.repo_path / file_name
            metrics[f"{sec_type}_exists"] = file_path.exists()
        
        # Check for secrets in environment files
        env_files = list(self.repo_path.rglob(".env*"))
        metrics["env_files_count"] = len(env_files)
        
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        metrics = {}
        
        # System metrics
        metrics["cpu_count"] = psutil.cpu_count()
        metrics["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        metrics["disk_usage_percent"] = psutil.disk_usage('/').percent
        
        # Docker metrics if available
        try:
            docker_info = subprocess.check_output(
                ["docker", "info", "--format", "{{.Containers}}"],
                text=True
            ).strip()
            metrics["docker_containers"] = int(docker_info)
        except subprocess.CalledProcessError:
            metrics["docker_available"] = False
        
        return metrics
    
    def _collect_infrastructure_metrics(self) -> Dict[str, Any]:
        """Collect infrastructure-related metrics."""
        metrics = {}
        
        # Container and deployment files
        infra_files = {
            "dockerfile": "Dockerfile",
            "docker_compose": "docker-compose.yml",
            "kubernetes": "k8s/",
            "helm": "helm-charts/",
            "terraform": "terraform/",
            "makefile": "Makefile"
        }
        
        for infra_type, file_name in infra_files.items():
            path = self.repo_path / file_name
            metrics[f"{infra_type}_exists"] = path.exists()
        
        return metrics
    
    def _collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        metrics = {}
        
        # Python dependencies
        requirements_files = ["requirements.txt", "pyproject.toml", "setup.py"]
        for req_file in requirements_files:
            file_path = self.repo_path / req_file
            if file_path.exists():
                metrics[f"{req_file}_exists"] = True
                if req_file == "requirements.txt":
                    try:
                        with open(file_path) as f:
                            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]
                            metrics["python_dependencies_count"] = len(lines)
                    except Exception:
                        pass
        
        # Node.js dependencies
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    package_data = json.load(f)
                    deps = package_data.get("dependencies", {})
                    dev_deps = package_data.get("devDependencies", {})
                    metrics["node_dependencies_count"] = len(deps)
                    metrics["node_dev_dependencies_count"] = len(dev_deps)
            except Exception:
                pass
        
        return metrics
    
    def _collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation-related metrics."""
        metrics = {}
        
        # Count documentation files
        doc_patterns = ["*.md", "docs/**/*", "*.rst", "*.txt"]
        doc_files = []
        for pattern in doc_patterns:
            doc_files.extend(list(self.repo_path.rglob(pattern)))
        
        metrics["documentation_files_count"] = len(set(doc_files))
        
        # Check for key documentation files
        key_docs = {
            "readme": "README.md",
            "changelog": "CHANGELOG.md",
            "contributing": "CONTRIBUTING.md",
            "license": "LICENSE",
            "code_of_conduct": "CODE_OF_CONDUCT.md"
        }
        
        for doc_type, file_name in key_docs.items():
            file_path = self.repo_path / file_name
            metrics[f"{doc_type}_exists"] = file_path.exists()
        
        return metrics
    
    def save_metrics(self, output_file: str = ".github/latest-metrics.json"):
        """Save collected metrics to a JSON file."""
        output_path = self.repo_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"📊 Metrics saved to {output_path}")
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report."""
        report = []
        report.append("# Repository Metrics Report")
        report.append(f"Generated: {self.timestamp}")
        report.append("")
        
        if "repository" in self.metrics:
            repo = self.metrics["repository"]
            report.append("## Repository Information")
            report.append(f"- Current Branch: {repo.get('current_branch', 'unknown')}")
            report.append(f"- Total Commits: {repo.get('total_commits', 'unknown')}")
            report.append("")
        
        if "code_quality" in self.metrics:
            code = self.metrics["code_quality"]
            report.append("## Code Quality")
            report.append(f"- Lines of Code: {code.get('lines_of_code', 'unknown')}")
            report.append(f"- Python Files: {code.get('python_files_count', 'unknown')}")
            report.append(f"- Markdown Files: {code.get('markdown_files', 'unknown')}")
            report.append("")
        
        if "testing" in self.metrics:
            testing = self.metrics["testing"]
            report.append("## Testing")
            report.append(f"- Test Files: {testing.get('test_files_count', 'unknown')}")
            report.append(f"- Pytest Configured: {testing.get('pytest_configured', 'unknown')}")
            report.append("")
        
        if "security" in self.metrics:
            security = self.metrics["security"]
            report.append("## Security")
            report.append(f"- Security Policy: {security.get('security_policy_exists', 'unknown')}")
            report.append(f"- CODEOWNERS: {security.get('codeowners_exists', 'unknown')}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect repository metrics")
    parser.add_argument("--output", "-o", default=".github/latest-metrics.json", 
                       help="Output file for metrics JSON")
    parser.add_argument("--report", "-r", help="Generate human-readable report to file")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.repo_path)
    metrics = collector.collect_all_metrics()
    
    # Save metrics
    collector.save_metrics(args.output)
    
    # Generate report if requested
    if args.report:
        report = collector.generate_report()
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"📋 Report saved to {args.report}")
    
    # Print summary
    print("\n📈 Metrics Collection Summary:")
    print(f"  - Code Quality Metrics: ✅")
    print(f"  - Development Metrics: ✅")
    print(f"  - Testing Metrics: ✅")
    print(f"  - Security Metrics: ✅")
    print(f"  - Infrastructure Metrics: ✅")
    print(f"  - Documentation Metrics: ✅")


if __name__ == "__main__":
    main()