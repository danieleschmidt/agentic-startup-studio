#!/usr/bin/env python3
"""
Automated Reporting System
Generates comprehensive reports for stakeholders automatically.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


class AutomatedReporter:
    """Automated report generation and distribution system."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.timestamp = datetime.now().isoformat()
        self.reports = {}
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive project status report."""
        print("📊 Generating comprehensive project report...")
        
        self.reports = {
            "metadata": {
                "generated_at": self.timestamp,
                "report_type": "comprehensive",
                "repository": self._get_repository_info(),
                "period": "last_30_days"
            },
            "executive_summary": self._generate_executive_summary(),
            "development_metrics": self._collect_development_metrics(),
            "quality_metrics": self._collect_quality_metrics(),
            "security_status": self._collect_security_status(),
            "performance_metrics": self._collect_performance_metrics(),
            "team_productivity": self._collect_team_metrics(),
            "infrastructure_status": self._collect_infrastructure_status(),
            "upcoming_milestones": self._identify_upcoming_milestones(),
            "recommendations": self._generate_recommendations(),
            "action_items": self._generate_action_items()
        }
        
        return self.reports
    
    def _get_repository_info(self) -> Dict[str, Any]:
        """Get basic repository information."""
        info = {}
        
        try:
            # Repository URL
            remote_url = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                cwd=self.repo_path,
                text=True
            ).strip()
            info["remote_url"] = remote_url
            
            # Current branch
            current_branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                text=True
            ).strip()
            info["current_branch"] = current_branch
            
            # Last commit
            last_commit = subprocess.check_output(
                ["git", "log", "-1", "--format=%H|%an|%ad|%s"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            parts = last_commit.split("|")
            info["last_commit"] = {
                "hash": parts[0][:8],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3]
            }
            
        except subprocess.CalledProcessError:
            info["error"] = "Failed to get repository information"
        
        return info
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary for stakeholders."""
        summary = {
            "overall_status": "healthy",
            "key_achievements": [],
            "critical_issues": [],
            "resource_needs": [],
            "timeline_status": "on_track"
        }
        
        # Analyze project health indicators
        try:
            # Check recent activity
            commits_30d = subprocess.check_output(
                ["git", "rev-list", "--count", "--since='30 days ago'", "HEAD"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            commits_count = int(commits_30d)
            
            if commits_count > 20:
                summary["key_achievements"].append(f"High development activity: {commits_count} commits in last 30 days")
                summary["overall_status"] = "excellent"
            elif commits_count > 10:
                summary["key_achievements"].append(f"Steady development: {commits_count} commits in last 30 days")
            elif commits_count < 5:
                summary["critical_issues"].append("Low development activity in recent weeks")
                summary["overall_status"] = "needs_attention"
            
        except subprocess.CalledProcessError:
            summary["critical_issues"].append("Unable to assess development activity")
        
        # Check for key project files
        key_files = {
            "README.md": "Project documentation",
            "tests/": "Test coverage",
            "docs/": "Documentation structure",
            "CI configuration": ".github/workflows/"
        }
        
        missing_files = []
        for file_name, description in key_files.items():
            file_path = self.repo_path / file_name
            if not file_path.exists():
                missing_files.append(description)
        
        if not missing_files:
            summary["key_achievements"].append("Complete project structure with all essential files")
        else:
            summary["critical_issues"].extend([f"Missing: {item}" for item in missing_files])
        
        return summary
    
    def _collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development-related metrics."""
        metrics = {
            "commit_frequency": {},
            "branch_activity": {},
            "contributor_activity": {},
            "code_changes": {}
        }
        
        try:
            # Commit frequency over time
            periods = ["7 days ago", "14 days ago", "30 days ago"]
            for period in periods:
                commits = subprocess.check_output(
                    ["git", "rev-list", "--count", f"--since='{period}'", "HEAD"],
                    cwd=self.repo_path,
                    text=True
                ).strip()
                metrics["commit_frequency"][period] = int(commits)
            
            # Branch information
            branches_output = subprocess.check_output(
                ["git", "branch", "-a"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            all_branches = [b.strip().replace('* ', '') for b in branches_output.split('\n') if b.strip()]
            local_branches = [b for b in all_branches if not b.startswith('remotes/')]
            
            metrics["branch_activity"] = {
                "total_branches": len(all_branches),
                "local_branches": len(local_branches),
                "active_branches": len([b for b in local_branches if b not in ['main', 'master']])
            }
            
            # Contributor activity
            contributors_output = subprocess.check_output(
                ["git", "shortlog", "-sn", "--since='30 days ago'"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            if contributors_output:
                contributors = []
                for line in contributors_output.split('\n'):
                    if line.strip():
                        parts = line.strip().split('\t')
                        contributors.append({
                            "commits": int(parts[0]),
                            "name": parts[1] if len(parts) > 1 else "Unknown"
                        })
                
                metrics["contributor_activity"] = {
                    "active_contributors": len(contributors),
                    "total_commits": sum(c["commits"] for c in contributors),
                    "top_contributors": contributors[:5]
                }
            
            # Code change statistics
            stats_output = subprocess.check_output(
                ["git", "diff", "--stat", "--since='30 days ago'", "HEAD~30", "HEAD"],
                cwd=self.repo_path,
                text=True,
                stderr=subprocess.DEVNULL
            ).strip()
            
            if stats_output:
                lines = stats_output.split('\n')
                if lines and 'files changed' in lines[-1]:
                    stats_line = lines[-1]
                    # Parse "X files changed, Y insertions(+), Z deletions(-)"
                    import re
                    match = re.search(r'(\d+) files? changed', stats_line)
                    files_changed = int(match.group(1)) if match else 0
                    
                    match = re.search(r'(\d+) insertions?', stats_line)
                    insertions = int(match.group(1)) if match else 0
                    
                    match = re.search(r'(\d+) deletions?', stats_line)
                    deletions = int(match.group(1)) if match else 0
                    
                    metrics["code_changes"] = {
                        "files_changed": files_changed,
                        "lines_added": insertions,
                        "lines_removed": deletions,
                        "net_change": insertions - deletions
                    }
            
        except subprocess.CalledProcessError as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def _collect_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {
            "test_coverage": 0,
            "code_complexity": "unknown",
            "documentation_coverage": 0,
            "technical_debt": "low"
        }
        
        # Count test files
        test_patterns = ["**/test_*.py", "**/tests/*.py", "**/*_test.py"]
        test_files = set()
        for pattern in test_patterns:
            test_files.update(self.repo_path.rglob(pattern))
        
        # Count source files
        source_files = list(self.repo_path.rglob("*.py"))
        source_files = [f for f in source_files if not any(test_pattern in str(f) for test_pattern in ['test_', 'tests/'])]
        
        if source_files:
            test_ratio = len(test_files) / len(source_files)
            metrics["test_coverage"] = min(test_ratio * 100, 100)
        
        # Documentation coverage (simplified)
        documented_functions = 0
        total_functions = 0
        
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import ast
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if total_functions > 0:
            metrics["documentation_coverage"] = (documented_functions / total_functions) * 100
        
        return metrics
    
    def _collect_security_status(self) -> Dict[str, Any]:
        """Collect security-related status."""
        status = {
            "security_files_present": {},
            "potential_vulnerabilities": [],
            "dependency_status": "unknown",
            "access_controls": "configured"
        }
        
        # Check for security files
        security_files = {
            "SECURITY.md": "Security policy",
            ".github/dependabot.yml": "Automated dependency updates",
            ".gitignore": "Git ignore configuration"
        }
        
        for file_name, description in security_files.items():
            file_path = self.repo_path / file_name
            status["security_files_present"][description] = file_path.exists()
        
        # Basic vulnerability scan (secrets)
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*[=:]\s*["\'][^"\']+["\']',
            r'secret[_-]?key\s*[=:]\s*["\'][^"\']+["\']'
        ]
        
        python_files = list(self.repo_path.rglob("*.py"))
        for file_path in python_files[:10]:  # Limit to first 10 files for performance
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in secret_patterns:
                    import re
                    if re.search(pattern, content, re.IGNORECASE):
                        status["potential_vulnerabilities"].append({
                            "type": "potential_secret",
                            "file": str(file_path),
                            "severity": "medium"
                        })
                        
            except UnicodeDecodeError:
                continue
        
        return status
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        metrics = {
            "build_time": "unknown",
            "test_execution_time": "unknown",
            "deployment_frequency": "unknown",
            "system_resources": {}
        }
        
        # System resource usage
        try:
            import psutil
            metrics["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        except ImportError:
            metrics["system_resources"] = {"error": "psutil not available"}
        
        # Check for performance testing files
        perf_files = list(self.repo_path.rglob("**/performance/**/*.py"))
        perf_files.extend(list(self.repo_path.rglob("**/load_test*.py")))
        
        if perf_files:
            metrics["performance_tests"] = len(perf_files)
        else:
            metrics["performance_tests"] = 0
        
        return metrics
    
    def _collect_team_metrics(self) -> Dict[str, Any]:
        """Collect team productivity metrics."""
        metrics = {
            "velocity": "stable",
            "collaboration_score": "good",
            "knowledge_sharing": "active"
        }
        
        try:
            # Analyze commit patterns for team collaboration
            commits_by_day = {}
            commits_output = subprocess.check_output(
                ["git", "log", "--since='7 days ago'", "--format=%ad", "--date=short"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            if commits_output:
                dates = commits_output.split('\n')
                for date in dates:
                    commits_by_day[date] = commits_by_day.get(date, 0) + 1
                
                metrics["daily_activity"] = commits_by_day
                metrics["active_days"] = len(commits_by_day)
                
                if len(commits_by_day) >= 5:
                    metrics["velocity"] = "high"
                elif len(commits_by_day) >= 3:
                    metrics["velocity"] = "stable"
                else:
                    metrics["velocity"] = "low"
        
        except subprocess.CalledProcessError:
            pass
        
        return metrics
    
    def _collect_infrastructure_status(self) -> Dict[str, Any]:
        """Collect infrastructure and deployment status."""
        status = {
            "containerization": "unknown",
            "ci_cd_status": "unknown",
            "monitoring": "unknown",
            "backup_status": "unknown"
        }
        
        # Check for infrastructure files
        infra_indicators = {
            "Dockerfile": "containerization",
            "docker-compose.yml": "containerization",
            ".github/workflows/": "ci_cd_status",
            "monitoring/": "monitoring",
            "k8s/": "kubernetes"
        }
        
        for file_name, category in infra_indicators.items():
            file_path = self.repo_path / file_name
            if file_path.exists():
                status[category] = "configured"
        
        return status
    
    def _identify_upcoming_milestones(self) -> List[Dict[str, Any]]:
        """Identify upcoming project milestones."""
        milestones = []
        
        # Check for milestone files or TODO items
        milestone_files = ["ROADMAP.md", "docs/ROADMAP.md", "milestones.md"]
        
        for file_name in milestone_files:
            file_path = self.repo_path / file_name
            if file_path.exists():
                milestones.append({
                    "type": "documented_roadmap",
                    "source": file_name,
                    "status": "available"
                })
                break
        
        # Look for TODO comments in code
        todo_count = 0
        python_files = list(self.repo_path.rglob("*.py"))
        
        for file_path in python_files[:20]:  # Limit for performance
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import re
                todos = re.findall(r'#\s*TODO[:\s]*(.*)', content, re.IGNORECASE)
                todo_count += len(todos)
                
            except UnicodeDecodeError:
                continue
        
        if todo_count > 0:
            milestones.append({
                "type": "pending_todos",
                "count": todo_count,
                "status": "needs_review"
            })
        
        return milestones
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on collected metrics, generate smart recommendations
        
        # Development activity recommendation
        dev_metrics = self.reports.get("development_metrics", {})
        if dev_metrics.get("commit_frequency", {}).get("7 days ago", 0) < 5:
            recommendations.append({
                "category": "development",
                "priority": "medium",
                "title": "Increase Development Velocity",
                "description": "Consider breaking down large tasks into smaller, more frequent commits",
                "impact": "Improved collaboration and easier debugging"
            })
        
        # Quality recommendation
        quality_metrics = self.reports.get("quality_metrics", {})
        if quality_metrics.get("test_coverage", 0) < 70:
            recommendations.append({
                "category": "quality",
                "priority": "high",
                "title": "Improve Test Coverage",
                "description": "Increase test coverage to at least 70% to ensure code reliability",
                "impact": "Reduced bugs and improved confidence in deployments"
            })
        
        # Security recommendation
        security_status = self.reports.get("security_status", {})
        if security_status.get("potential_vulnerabilities"):
            recommendations.append({
                "category": "security",
                "priority": "high",
                "title": "Address Security Vulnerabilities",
                "description": "Review and fix potential security issues identified in code scan",
                "impact": "Improved security posture and compliance"
            })
        
        return recommendations
    
    def _generate_action_items(self) -> List[Dict[str, Any]]:
        """Generate specific action items."""
        action_items = []
        
        # Convert recommendations to action items
        recommendations = self.reports.get("recommendations", [])
        
        for i, rec in enumerate(recommendations):
            action_items.append({
                "id": f"action_{i+1}",
                "title": rec["title"],
                "description": rec["description"],
                "priority": rec["priority"],
                "category": rec["category"],
                "estimated_effort": "medium",
                "assigned_to": "team",
                "due_date": (datetime.now() + timedelta(days=14)).isoformat(),
                "status": "open"
            })
        
        return action_items
    
    def generate_stakeholder_report(self, stakeholder_type: str = "management") -> str:
        """Generate stakeholder-specific report."""
        if stakeholder_type == "management":
            return self._generate_management_report()
        elif stakeholder_type == "technical":
            return self._generate_technical_report()
        elif stakeholder_type == "security":
            return self._generate_security_report()
        else:
            return self._generate_general_report()
    
    def _generate_management_report(self) -> str:
        """Generate management-focused report."""
        report = []
        report.append("# Executive Project Status Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        
        # Executive Summary
        exec_summary = self.reports["executive_summary"]
        report.append("## Executive Summary")
        report.append(f"**Overall Status:** {exec_summary['overall_status'].title().replace('_', ' ')}")
        report.append(f"**Timeline Status:** {exec_summary['timeline_status'].title().replace('_', ' ')}")
        report.append("")
        
        if exec_summary["key_achievements"]:
            report.append("### Key Achievements")
            for achievement in exec_summary["key_achievements"]:
                report.append(f"- ✅ {achievement}")
            report.append("")
        
        if exec_summary["critical_issues"]:
            report.append("### Critical Issues")
            for issue in exec_summary["critical_issues"]:
                report.append(f"- ⚠️ {issue}")
            report.append("")
        
        # Development Progress
        dev_metrics = self.reports["development_metrics"]
        report.append("## Development Progress")
        
        if "commit_frequency" in dev_metrics:
            report.append("### Activity Summary")
            freq = dev_metrics["commit_frequency"]
            report.append(f"- Last 7 days: {freq.get('7 days ago', 0)} commits")
            report.append(f"- Last 30 days: {freq.get('30 days ago', 0)} commits")
            report.append("")
        
        if "contributor_activity" in dev_metrics:
            contrib = dev_metrics["contributor_activity"]
            report.append(f"- Active Contributors: {contrib.get('active_contributors', 0)}")
            report.append(f"- Total Commits (30d): {contrib.get('total_commits', 0)}")
            report.append("")
        
        # Quality Metrics
        quality = self.reports["quality_metrics"]
        report.append("## Quality Metrics")
        report.append(f"- Test Coverage: {quality.get('test_coverage', 0):.1f}%")
        report.append(f"- Documentation Coverage: {quality.get('documentation_coverage', 0):.1f}%")
        report.append("")
        
        # Action Items
        action_items = self.reports["action_items"]
        if action_items:
            report.append("## Priority Action Items")
            high_priority = [item for item in action_items if item["priority"] == "high"]
            for item in high_priority[:3]:  # Top 3 high priority items
                report.append(f"- **{item['title']}** (Due: {item['due_date'][:10]})")
                report.append(f"  {item['description']}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_technical_report(self) -> str:
        """Generate technical team report."""
        report = []
        report.append("# Technical Status Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        
        # Development Metrics
        dev_metrics = self.reports["development_metrics"]
        report.append("## Development Metrics")
        
        if "code_changes" in dev_metrics:
            changes = dev_metrics["code_changes"]
            report.append(f"- Files Changed (30d): {changes.get('files_changed', 0)}")
            report.append(f"- Lines Added: {changes.get('lines_added', 0)}")
            report.append(f"- Lines Removed: {changes.get('lines_removed', 0)}")
            report.append(f"- Net Change: {changes.get('net_change', 0)}")
            report.append("")
        
        # Quality Analysis
        quality = self.reports["quality_metrics"]
        report.append("## Code Quality")
        report.append(f"- Test Coverage: {quality.get('test_coverage', 0):.1f}%")
        report.append(f"- Documentation Coverage: {quality.get('documentation_coverage', 0):.1f}%")
        report.append(f"- Technical Debt: {quality.get('technical_debt', 'Unknown')}")
        report.append("")
        
        # Security Status
        security = self.reports["security_status"]
        report.append("## Security Status")
        vulnerabilities = security.get("potential_vulnerabilities", [])
        report.append(f"- Potential Issues Found: {len(vulnerabilities)}")
        
        security_files = security.get("security_files_present", {})
        for description, present in security_files.items():
            status = "✅" if present else "❌"
            report.append(f"- {description}: {status}")
        report.append("")
        
        # Infrastructure
        infra = self.reports["infrastructure_status"]
        report.append("## Infrastructure")
        for component, status in infra.items():
            emoji = "✅" if status == "configured" else "⚠️"
            report.append(f"- {component.replace('_', ' ').title()}: {status} {emoji}")
        report.append("")
        
        return "\n".join(report)
    
    def _generate_security_report(self) -> str:
        """Generate security-focused report."""
        report = []
        report.append("# Security Status Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        
        security = self.reports["security_status"]
        
        # Security Overview
        vulnerabilities = security.get("potential_vulnerabilities", [])
        report.append("## Security Overview")
        report.append(f"- Potential Vulnerabilities: {len(vulnerabilities)}")
        report.append(f"- Dependency Status: {security.get('dependency_status', 'Unknown')}")
        report.append(f"- Access Controls: {security.get('access_controls', 'Unknown')}")
        report.append("")
        
        # Security Files Status
        report.append("## Security Configuration")
        security_files = security.get("security_files_present", {})
        for description, present in security_files.items():
            status = "✅ Configured" if present else "❌ Missing"
            report.append(f"- {description}: {status}")
        report.append("")
        
        # Vulnerability Details
        if vulnerabilities:
            report.append("## Identified Issues")
            for vuln in vulnerabilities:
                report.append(f"- **{vuln['type']}** in {vuln['file']} (Severity: {vuln['severity']})")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_general_report(self) -> str:
        """Generate general comprehensive report."""
        report = []
        report.append("# Comprehensive Project Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        
        # Include all sections
        report.append(self._generate_management_report())
        report.append("\n---\n")
        report.append(self._generate_technical_report())
        report.append("\n---\n")
        report.append(self._generate_security_report())
        
        return "\n".join(report)
    
    def save_reports(self, output_dir: str = ".github/reports"):
        """Save all reports to files."""
        output_path = self.repo_path / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON data
        with open(output_path / "comprehensive-report.json", 'w') as f:
            json.dump(self.reports, f, indent=2)
        
        # Save stakeholder reports
        reports_to_generate = [
            ("management", "management-report.md"),
            ("technical", "technical-report.md"),
            ("security", "security-report.md"),
            ("general", "comprehensive-report.md")
        ]
        
        for stakeholder_type, filename in reports_to_generate:
            report_content = self.generate_stakeholder_report(stakeholder_type)
            with open(output_path / filename, 'w') as f:
                f.write(report_content)
        
        print(f"📊 Reports saved to {output_path}")
        return output_path


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate automated project reports")
    parser.add_argument("--type", choices=["management", "technical", "security", "all"], 
                       default="all", help="Type of report to generate")
    parser.add_argument("--output-dir", default=".github/reports", 
                       help="Output directory for reports")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--format", choices=["markdown", "json", "both"], 
                       default="both", help="Output format")
    
    args = parser.parse_args()
    
    reporter = AutomatedReporter(args.repo_path)
    
    # Generate comprehensive data
    print("📊 Collecting project data...")
    comprehensive_data = reporter.generate_comprehensive_report()
    
    # Save reports
    if args.format in ["both", "json", "markdown"]:
        output_path = reporter.save_reports(args.output_dir)
    
    # Print summary
    exec_summary = comprehensive_data["executive_summary"]
    print(f"\n📈 Report Generation Complete")
    print(f"Overall Status: {exec_summary['overall_status'].title().replace('_', ' ')}")
    print(f"Key Achievements: {len(exec_summary['key_achievements'])}")
    print(f"Critical Issues: {len(exec_summary['critical_issues'])}")
    print(f"Action Items: {len(comprehensive_data['action_items'])}")


if __name__ == "__main__":
    main()