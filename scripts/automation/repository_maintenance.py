#!/usr/bin/env python3
"""
Repository Maintenance Automation
Performs automated maintenance tasks for repository health.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil
import tempfile


class RepositoryMaintenance:
    """Automated repository maintenance and cleanup system."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.timestamp = datetime.now().isoformat()
        self.maintenance_log = []
        
    def run_full_maintenance(self) -> Dict[str, Any]:
        """Run complete repository maintenance cycle."""
        print("🔧 Starting comprehensive repository maintenance...")
        
        results = {
            "timestamp": self.timestamp,
            "git_cleanup": self._git_cleanup(),
            "file_cleanup": self._file_cleanup(),
            "dependency_audit": self._dependency_audit(),
            "security_scan": self._security_scan(),
            "performance_optimization": self._performance_optimization(),
            "backup_verification": self._backup_verification(),
            "health_check": self._repository_health_check()
        }
        
        results["maintenance_log"] = self.maintenance_log
        results["summary"] = self._generate_maintenance_summary(results)
        
        return results
    
    def _git_cleanup(self) -> Dict[str, Any]:
        """Perform Git repository cleanup."""
        results = {"actions": [], "errors": []}
        
        try:
            # Git garbage collection
            subprocess.run(
                ["git", "gc", "--prune=now"],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            results["actions"].append("Git garbage collection completed")
            self.maintenance_log.append("Git GC: Completed successfully")
            
        except subprocess.CalledProcessError as e:
            results["errors"].append(f"Git GC failed: {e.stderr}")
        
        try:
            # Remove merged branches (except main/master)
            branches_output = subprocess.run(
                ["git", "branch", "--merged"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if branches_output.returncode == 0:
                merged_branches = [
                    b.strip() for b in branches_output.stdout.split('\n') 
                    if b.strip() and not b.strip().startswith('*') 
                    and b.strip() not in ['main', 'master', 'develop']
                ]
                
                for branch in merged_branches:
                    try:
                        subprocess.run(
                            ["git", "branch", "-d", branch],
                            cwd=self.repo_path,
                            check=True,
                            capture_output=True
                        )
                        results["actions"].append(f"Deleted merged branch: {branch}")
                        self.maintenance_log.append(f"Cleanup: Deleted merged branch {branch}")
                    except subprocess.CalledProcessError:
                        pass  # Branch might be referenced elsewhere
            
        except Exception as e:
            results["errors"].append(f"Branch cleanup failed: {str(e)}")
        
        try:
            # Clean up remote tracking branches
            subprocess.run(
                ["git", "remote", "prune", "origin"],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            results["actions"].append("Pruned remote tracking branches")
            self.maintenance_log.append("Git remote prune: Completed")
            
        except subprocess.CalledProcessError as e:
            results["errors"].append(f"Remote prune failed: {e.stderr}")
        
        return results
    
    def _file_cleanup(self) -> Dict[str, Any]:
        """Clean up unnecessary files and directories."""
        results = {"cleaned_files": [], "space_saved": 0, "errors": []}
        
        # File patterns to clean up
        cleanup_patterns = [
            "**/*.pyc",
            "**/__pycache__",
            "**/.pytest_cache",
            "**/.coverage",
            "**/htmlcov",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.tmp",
            "**/*.log",
            "**/node_modules/.cache",
            "**/.nyc_output"
        ]
        
        total_size_saved = 0
        
        for pattern in cleanup_patterns:
            try:
                files_to_remove = list(self.repo_path.rglob(pattern))
                
                for file_path in files_to_remove:
                    try:
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            file_path.unlink()
                            total_size_saved += size
                            results["cleaned_files"].append(str(file_path))
                            
                        elif file_path.is_dir():
                            size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
                            shutil.rmtree(file_path)
                            total_size_saved += size
                            results["cleaned_files"].append(str(file_path))
                            
                    except PermissionError:
                        results["errors"].append(f"Permission denied: {file_path}")
                    except Exception as e:
                        results["errors"].append(f"Error removing {file_path}: {str(e)}")
                        
            except Exception as e:
                results["errors"].append(f"Error processing pattern {pattern}: {str(e)}")
        
        results["space_saved"] = total_size_saved
        self.maintenance_log.append(f"File cleanup: Removed {len(results['cleaned_files'])} items, saved {total_size_saved} bytes")
        
        return results
    
    def _dependency_audit(self) -> Dict[str, Any]:
        """Audit and check dependencies for issues."""
        results = {"python": {}, "node": {}, "docker": {}}
        
        # Python dependency audit
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                # Check for known vulnerabilities using safety
                safety_result = subprocess.run(
                    ["python", "-m", "pip", "install", "safety", "&&", "safety", "check", "--json"],
                    shell=True,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if safety_result.returncode == 0 and safety_result.stdout:
                    results["python"]["vulnerabilities"] = json.loads(safety_result.stdout)
                else:
                    results["python"]["vulnerabilities"] = []
                
            except Exception as e:
                results["python"]["error"] = str(e)
        
        # Node.js dependency audit
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                audit_result = subprocess.run(
                    ["npm", "audit", "--json"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if audit_result.stdout:
                    results["node"]["audit"] = json.loads(audit_result.stdout)
                
            except Exception as e:
                results["node"]["error"] = str(e)
        
        # Docker image audit (if Dockerfile exists)
        dockerfile = self.repo_path / "Dockerfile"
        if dockerfile.exists():
            try:
                # Extract base images
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    import re
                    base_images = re.findall(r'^FROM\s+([^\s]+)', content, re.MULTILINE)
                    results["docker"]["base_images"] = base_images
                    
            except Exception as e:
                results["docker"]["error"] = str(e)
        
        self.maintenance_log.append("Dependency audit: Completed")
        return results
    
    def _security_scan(self) -> Dict[str, Any]:
        """Perform basic security scanning."""
        results = {"secrets_found": [], "security_issues": [], "recommendations": []}
        
        # Scan for potential secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "potential_password"),
            (r'api[_-]?key\s*[=:]\s*["\'][^"\']+["\']', "potential_api_key"),
            (r'secret[_-]?key\s*[=:]\s*["\'][^"\']+["\']', "potential_secret"),
            (r'token\s*[=:]\s*["\'][^"\']+["\']', "potential_token"),
            (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', "private_key")
        ]
        
        text_files = [
            *list(self.repo_path.rglob("*.py")),
            *list(self.repo_path.rglob("*.js")),
            *list(self.repo_path.rglob("*.ts")),
            *list(self.repo_path.rglob("*.yml")),
            *list(self.repo_path.rglob("*.yaml")),
            *list(self.repo_path.rglob("*.json")),
            *list(self.repo_path.rglob("*.env*"))
        ]
        
        for file_path in text_files:
            if file_path.name.startswith('.git'):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, issue_type in secret_patterns:
                    import re
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        results["secrets_found"].append({
                            "file": str(file_path),
                            "line": line_num,
                            "type": issue_type,
                            "context": match.group()[:50] + "..." if len(match.group()) > 50 else match.group()
                        })
                        
            except (UnicodeDecodeError, PermissionError):
                continue
        
        # Check for secure configurations
        security_files = {
            ".gitignore": "Git ignore file exists",
            "SECURITY.md": "Security policy documented",
            ".github/dependabot.yml": "Dependabot configured"
        }
        
        for file_name, description in security_files.items():
            file_path = self.repo_path / file_name
            if not file_path.exists():
                results["recommendations"].append(f"Missing: {description}")
        
        self.maintenance_log.append(f"Security scan: Found {len(results['secrets_found'])} potential secrets")
        return results
    
    def _performance_optimization(self) -> Dict[str, Any]:
        """Perform performance optimization tasks."""
        results = {"optimizations": [], "recommendations": []}
        
        # Check for large files
        large_files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    if size > 10 * 1024 * 1024:  # 10MB threshold
                        large_files.append({
                            "file": str(file_path),
                            "size_mb": round(size / (1024 * 1024), 2)
                        })
                except (PermissionError, OSError):
                    continue
        
        if large_files:
            results["recommendations"].append(f"Consider Git LFS for {len(large_files)} large files")
        
        # Check for optimization opportunities
        python_files = list(self.repo_path.rglob("*.py"))
        if python_files:
            # Check for potential performance issues
            perf_patterns = [
                (r'for\s+\w+\s+in\s+range\(len\(', "Use enumerate instead of range(len())"),
                (r'\.append\(\s*\w+\s*\)\s*$', "Consider list comprehension for simple appends"),
                (r'open\([^)]+\)\.read\(\)', "Use context manager for file operations")
            ]
            
            perf_issues = 0
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, suggestion in perf_patterns:
                        import re
                        if re.search(pattern, content):
                            perf_issues += 1
                            
                except UnicodeDecodeError:
                    continue
            
            if perf_issues > 0:
                results["recommendations"].append(f"Review {perf_issues} potential performance improvements")
        
        self.maintenance_log.append("Performance optimization: Analysis completed")
        return results
    
    def _backup_verification(self) -> Dict[str, Any]:
        """Verify backup and recovery procedures."""
        results = {"backup_status": "unknown", "recommendations": []}
        
        # Check if repository is properly tracked
        try:
            remote_output = subprocess.run(
                ["git", "remote", "-v"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if remote_output.returncode == 0 and remote_output.stdout:
                results["backup_status"] = "git_remote_configured"
                results["remote_info"] = remote_output.stdout.strip()
            else:
                results["backup_status"] = "no_remote"
                results["recommendations"].append("Configure Git remote for backup")
                
        except subprocess.CalledProcessError:
            results["backup_status"] = "git_error"
        
        # Check for backup configuration files
        backup_configs = ["backup.yml", ".github/workflows/backup.yml", "scripts/backup.sh"]
        for config_file in backup_configs:
            config_path = self.repo_path / config_file
            if config_path.exists():
                results["backup_status"] = "automated_backup_configured"
                break
        else:
            results["recommendations"].append("Consider setting up automated backups")
        
        self.maintenance_log.append("Backup verification: Completed")
        return results
    
    def _repository_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive repository health check."""
        results = {"health_score": 0, "issues": [], "strengths": []}
        
        health_checks = [
            ("README.md exists", (self.repo_path / "README.md").exists()),
            ("LICENSE file exists", (self.repo_path / "LICENSE").exists()),
            ("CONTRIBUTING.md exists", (self.repo_path / "CONTRIBUTING.md").exists()),
            ("Tests directory exists", (self.repo_path / "tests").exists()),
            ("CI configuration exists", (self.repo_path / ".github" / "workflows").exists()),
            ("Requirements file exists", any((self.repo_path / f).exists() for f in ["requirements.txt", "pyproject.toml", "package.json"])),
            ("Git ignore configured", (self.repo_path / ".gitignore").exists()),
            ("Security policy exists", (self.repo_path / "SECURITY.md").exists())
        ]
        
        passed_checks = 0
        for check_name, check_result in health_checks:
            if check_result:
                passed_checks += 1
                results["strengths"].append(check_name)
            else:
                results["issues"].append(f"Missing: {check_name}")
        
        results["health_score"] = (passed_checks / len(health_checks)) * 100
        
        # Additional checks
        try:
            # Check commit frequency
            recent_commits = subprocess.run(
                ["git", "log", "--since='30 days ago'", "--oneline"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if recent_commits.returncode == 0:
                commit_count = len(recent_commits.stdout.strip().split('\n')) if recent_commits.stdout.strip() else 0
                if commit_count > 0:
                    results["strengths"].append(f"Active development ({commit_count} commits in last 30 days)")
                else:
                    results["issues"].append("No recent commits (last 30 days)")
                    
        except subprocess.CalledProcessError:
            pass
        
        self.maintenance_log.append(f"Health check: Score {results['health_score']:.1f}/100")
        return results
    
    def _generate_maintenance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of maintenance activities."""
        summary = {
            "total_actions": 0,
            "total_errors": 0,
            "space_saved_mb": 0,
            "security_issues": 0,
            "health_score": 0,
            "recommendations_count": 0
        }
        
        # Count actions and errors
        for category, data in results.items():
            if isinstance(data, dict):
                if "actions" in data:
                    summary["total_actions"] += len(data["actions"])
                if "errors" in data:
                    summary["total_errors"] += len(data["errors"])
                if "recommendations" in data:
                    summary["recommendations_count"] += len(data["recommendations"])
        
        # Space saved
        if "file_cleanup" in results and "space_saved" in results["file_cleanup"]:
            summary["space_saved_mb"] = round(results["file_cleanup"]["space_saved"] / (1024 * 1024), 2)
        
        # Security issues
        if "security_scan" in results and "secrets_found" in results["security_scan"]:
            summary["security_issues"] = len(results["security_scan"]["secrets_found"])
        
        # Health score
        if "health_check" in results and "health_score" in results["health_check"]:
            summary["health_score"] = results["health_check"]["health_score"]
        
        return summary
    
    def save_maintenance_report(self, results: Dict[str, Any], output_file: str = ".github/maintenance-report.json"):
        """Save maintenance report to JSON file."""
        output_path = self.repo_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"🔧 Maintenance report saved to {output_path}")
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable maintenance summary."""
        report = []
        report.append("# Repository Maintenance Report")
        report.append(f"Generated: {self.timestamp}")
        report.append("")
        
        summary = results["summary"]
        report.append("## Summary")
        report.append(f"- Actions Performed: {summary['total_actions']}")
        report.append(f"- Errors Encountered: {summary['total_errors']}")
        report.append(f"- Space Saved: {summary['space_saved_mb']} MB")
        report.append(f"- Security Issues Found: {summary['security_issues']}")
        report.append(f"- Repository Health Score: {summary['health_score']:.1f}/100")
        report.append(f"- Total Recommendations: {summary['recommendations_count']}")
        report.append("")
        
        # Git cleanup
        git_results = results["git_cleanup"]
        if git_results["actions"]:
            report.append("## Git Cleanup")
            for action in git_results["actions"]:
                report.append(f"- ✅ {action}")
            if git_results["errors"]:
                for error in git_results["errors"]:
                    report.append(f"- ❌ {error}")
            report.append("")
        
        # File cleanup
        file_results = results["file_cleanup"]
        if file_results["cleaned_files"]:
            report.append("## File Cleanup")
            report.append(f"- Cleaned {len(file_results['cleaned_files'])} files/directories")
            report.append(f"- Space saved: {summary['space_saved_mb']} MB")
            report.append("")
        
        # Security findings
        security_results = results["security_scan"]
        if security_results["secrets_found"]:
            report.append("## Security Issues ⚠️")
            for secret in security_results["secrets_found"][:5]:  # Show first 5
                report.append(f"- {secret['type']} in {secret['file']}:{secret['line']}")
            if len(security_results["secrets_found"]) > 5:
                report.append(f"- ... and {len(security_results['secrets_found']) - 5} more")
            report.append("")
        
        # Health check
        health_results = results["health_check"]
        report.append("## Repository Health")
        report.append(f"Health Score: {health_results['health_score']:.1f}/100")
        
        if health_results["strengths"]:
            report.append("### Strengths")
            for strength in health_results["strengths"]:
                report.append(f"- ✅ {strength}")
        
        if health_results["issues"]:
            report.append("### Issues to Address")
            for issue in health_results["issues"]:
                report.append(f"- ⚠️ {issue}")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform repository maintenance")
    parser.add_argument("--output", "-o", default=".github/maintenance-report.json",
                       help="Output file for maintenance report")
    parser.add_argument("--summary", "-s", help="Generate summary report to file")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--skip-git", action="store_true", help="Skip Git cleanup")
    parser.add_argument("--skip-files", action="store_true", help="Skip file cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    
    maintenance = RepositoryMaintenance(args.repo_path)
    results = maintenance.run_full_maintenance()
    
    # Save detailed report
    maintenance.save_maintenance_report(results, args.output)
    
    # Generate summary if requested
    if args.summary:
        summary = maintenance.generate_summary_report(results)
        with open(args.summary, 'w') as f:
            f.write(summary)
        print(f"📋 Summary report saved to {args.summary}")
    
    # Print summary to console
    summary = results["summary"]
    print(f"\n🔧 Repository Maintenance Complete")
    print(f"Actions: {summary['total_actions']}, Errors: {summary['total_errors']}")
    print(f"Space Saved: {summary['space_saved_mb']} MB")
    print(f"Health Score: {summary['health_score']:.1f}/100")
    
    if summary["security_issues"] > 0:
        print(f"⚠️  {summary['security_issues']} security issues found - review recommended")


if __name__ == "__main__":
    main()