#!/usr/bin/env python3
"""
Automated Dependency Update System
Manages and updates project dependencies across different package managers.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class DependencyUpdater:
    """Automated dependency management and update system."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.update_log = []
        self.timestamp = datetime.now().isoformat()
        
    def check_all_dependencies(self) -> Dict[str, Any]:
        """Check all dependency files and their current status."""
        print("🔍 Checking all project dependencies...")
        
        results = {
            "timestamp": self.timestamp,
            "python": self._check_python_dependencies(),
            "node": self._check_node_dependencies(),
            "docker": self._check_docker_dependencies(),
            "github_actions": self._check_github_actions(),
            "system": self._check_system_dependencies()
        }
        
        return results
    
    def _check_python_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies."""
        result = {"found": False, "files": [], "outdated": [], "security_issues": []}
        
        # Check for Python dependency files
        python_files = [
            "requirements.txt",
            "requirements-dev.txt", 
            "requirements-test.txt",
            "pyproject.toml",
            "setup.py",
            "Pipfile"
        ]
        
        for file_name in python_files:
            file_path = self.repo_path / file_name
            if file_path.exists():
                result["files"].append(file_name)
                result["found"] = True
        
        # Check for outdated packages using pip-outdated
        if result["found"]:
            try:
                # Check for outdated packages
                outdated_output = subprocess.check_output(
                    ["python", "-m", "pip", "list", "--outdated", "--format=json"],
                    text=True,
                    cwd=self.repo_path
                )
                
                outdated_packages = json.loads(outdated_output)
                result["outdated"] = outdated_packages
                
            except subprocess.CalledProcessError:
                result["outdated_check_error"] = "Failed to check outdated packages"
            
            # Check for security vulnerabilities
            try:
                safety_output = subprocess.check_output(
                    ["python", "-m", "pip", "install", "safety", "&&", "safety", "check", "--json"],
                    shell=True,
                    text=True,
                    cwd=self.repo_path
                )
                
                if safety_output.strip():
                    security_issues = json.loads(safety_output)
                    result["security_issues"] = security_issues
                    
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                result["security_check_note"] = "Security check skipped (safety not available)"
        
        return result
    
    def _check_node_dependencies(self) -> Dict[str, Any]:
        """Check Node.js dependencies."""
        result = {"found": False, "files": [], "outdated": [], "vulnerabilities": []}
        
        package_json = self.repo_path / "package.json"
        yarn_lock = self.repo_path / "yarn.lock"
        package_lock = self.repo_path / "package-lock.json"
        
        if package_json.exists():
            result["found"] = True
            result["files"].append("package.json")
            
            if yarn_lock.exists():
                result["files"].append("yarn.lock")
                package_manager = "yarn"
            elif package_lock.exists():
                result["files"].append("package-lock.json")
                package_manager = "npm"
            else:
                package_manager = "npm"
            
            # Check for outdated packages
            try:
                if package_manager == "yarn":
                    outdated_cmd = ["yarn", "outdated", "--json"]
                else:
                    outdated_cmd = ["npm", "outdated", "--json"]
                
                outdated_output = subprocess.check_output(
                    outdated_cmd,
                    text=True,
                    cwd=self.repo_path
                )
                
                if outdated_output.strip():
                    result["outdated"] = json.loads(outdated_output)
                    
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                result["outdated_check_note"] = f"Could not check outdated {package_manager} packages"
            
            # Check for security vulnerabilities
            try:
                if package_manager == "yarn":
                    audit_cmd = ["yarn", "audit", "--json"]
                else:
                    audit_cmd = ["npm", "audit", "--json"]
                
                audit_output = subprocess.check_output(
                    audit_cmd,
                    text=True,
                    cwd=self.repo_path
                )
                
                if audit_output.strip():
                    result["vulnerabilities"] = json.loads(audit_output)
                    
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                result["audit_check_note"] = f"Could not run {package_manager} security audit"
        
        return result
    
    def _check_docker_dependencies(self) -> Dict[str, Any]:
        """Check Docker image dependencies."""
        result = {"found": False, "files": [], "base_images": []}
        
        docker_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
        
        for file_name in docker_files:
            file_path = self.repo_path / file_name
            if file_path.exists():
                result["found"] = True
                result["files"].append(file_name)
                
                # Extract base images from Dockerfile
                if file_name == "Dockerfile":
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            from_lines = re.findall(r'^FROM\s+([^\s]+)', content, re.MULTILINE)
                            result["base_images"].extend(from_lines)
                    except Exception:
                        pass
        
        return result
    
    def _check_github_actions(self) -> Dict[str, Any]:
        """Check GitHub Actions workflow dependencies."""
        result = {"found": False, "workflows": [], "actions_used": []}
        
        workflows_dir = self.repo_path / ".github" / "workflows"
        if workflows_dir.exists():
            result["found"] = True
            
            for workflow_file in workflows_dir.glob("*.yml"):
                result["workflows"].append(workflow_file.name)
                
                try:
                    with open(workflow_file, 'r') as f:
                        content = f.read()
                        # Extract action references
                        action_refs = re.findall(r'uses:\s*([^\s]+)', content)
                        result["actions_used"].extend(action_refs)
                except Exception:
                    pass
        
        # Remove duplicates
        result["actions_used"] = list(set(result["actions_used"]))
        
        return result
    
    def _check_system_dependencies(self) -> Dict[str, Any]:
        """Check system-level dependencies."""
        result = {"tools": {}}
        
        # Check for common development tools
        tools_to_check = [
            "git", "docker", "python", "node", "npm", "yarn", 
            "make", "curl", "wget", "jq", "helm", "kubectl"
        ]
        
        for tool in tools_to_check:
            try:
                version_output = subprocess.check_output(
                    [tool, "--version"],
                    text=True,
                    stderr=subprocess.DEVNULL
                )
                result["tools"][tool] = {
                    "available": True,
                    "version": version_output.strip().split('\n')[0]
                }
            except (subprocess.CalledProcessError, FileNotFoundError):
                result["tools"][tool] = {"available": False}
        
        return result
    
    def update_python_dependencies(self, update_type: str = "minor") -> Dict[str, Any]:
        """Update Python dependencies."""
        result = {"success": False, "updates": [], "errors": []}
        
        requirements_file = self.repo_path / "requirements.txt"
        if not requirements_file.exists():
            result["errors"].append("requirements.txt not found")
            return result
        
        try:
            if update_type == "security":
                # Only update packages with security vulnerabilities
                cmd = ["python", "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed"]
            elif update_type == "minor":
                # Update to latest minor versions
                cmd = ["python", "-m", "pip", "install", "--upgrade"]
            else:
                result["errors"].append(f"Unknown update type: {update_type}")
                return result
            
            # Read current requirements
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Update packages
            for requirement in requirements:
                package_name = requirement.split('==')[0].split('>=')[0].split('~=')[0]
                try:
                    subprocess.check_output(
                        cmd + [package_name],
                        cwd=self.repo_path,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    result["updates"].append(package_name)
                except subprocess.CalledProcessError as e:
                    result["errors"].append(f"Failed to update {package_name}: {e.output}")
            
            result["success"] = len(result["updates"]) > 0
            
        except Exception as e:
            result["errors"].append(f"Update process failed: {str(e)}")
        
        return result
    
    def update_node_dependencies(self, update_type: str = "minor") -> Dict[str, Any]:
        """Update Node.js dependencies."""
        result = {"success": False, "updates": [], "errors": []}
        
        package_json = self.repo_path / "package.json"
        if not package_json.exists():
            result["errors"].append("package.json not found")
            return result
        
        # Determine package manager
        if (self.repo_path / "yarn.lock").exists():
            package_manager = "yarn"
        else:
            package_manager = "npm"
        
        try:
            if update_type == "security":
                if package_manager == "yarn":
                    cmd = ["yarn", "audit", "fix"]
                else:
                    cmd = ["npm", "audit", "fix"]
            elif update_type == "minor":
                if package_manager == "yarn":
                    cmd = ["yarn", "upgrade"]
                else:
                    cmd = ["npm", "update"]
            else:
                result["errors"].append(f"Unknown update type: {update_type}")
                return result
            
            output = subprocess.check_output(
                cmd,
                cwd=self.repo_path,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            result["success"] = True
            result["output"] = output
            
        except subprocess.CalledProcessError as e:
            result["errors"].append(f"Update failed: {e.output}")
        
        return result
    
    def generate_update_report(self, check_results: Dict[str, Any]) -> str:
        """Generate a human-readable dependency update report."""
        report = []
        report.append("# Dependency Update Report")
        report.append(f"Generated: {self.timestamp}")
        report.append("")
        
        # Python dependencies
        if check_results["python"]["found"]:
            report.append("## Python Dependencies")
            report.append(f"Files found: {', '.join(check_results['python']['files'])}")
            
            outdated = check_results["python"].get("outdated", [])
            if outdated:
                report.append(f"Outdated packages: {len(outdated)}")
                for pkg in outdated[:5]:  # Show first 5
                    report.append(f"  - {pkg.get('name', 'unknown')}: {pkg.get('version', 'unknown')} → {pkg.get('latest_version', 'unknown')}")
                if len(outdated) > 5:
                    report.append(f"  ... and {len(outdated) - 5} more")
            else:
                report.append("All packages up to date ✅")
            
            security_issues = check_results["python"].get("security_issues", [])
            if security_issues:
                report.append(f"Security issues: {len(security_issues)} ⚠️")
            else:
                report.append("No security issues found ✅")
            
            report.append("")
        
        # Node.js dependencies
        if check_results["node"]["found"]:
            report.append("## Node.js Dependencies")
            report.append(f"Files found: {', '.join(check_results['node']['files'])}")
            
            outdated = check_results["node"].get("outdated", [])
            if outdated:
                report.append(f"Outdated packages found ⚠️")
            else:
                report.append("All packages up to date ✅")
            
            vulnerabilities = check_results["node"].get("vulnerabilities", [])
            if vulnerabilities:
                report.append(f"Security vulnerabilities found ⚠️")
            else:
                report.append("No security vulnerabilities ✅")
            
            report.append("")
        
        # Docker dependencies
        if check_results["docker"]["found"]:
            report.append("## Docker Dependencies")
            report.append(f"Files found: {', '.join(check_results['docker']['files'])}")
            
            base_images = check_results["docker"].get("base_images", [])
            if base_images:
                report.append("Base images:")
                for image in base_images:
                    report.append(f"  - {image}")
            
            report.append("")
        
        # GitHub Actions
        if check_results["github_actions"]["found"]:
            report.append("## GitHub Actions")
            report.append(f"Workflows: {len(check_results['github_actions']['workflows'])}")
            report.append(f"Actions used: {len(check_results['github_actions']['actions_used'])}")
            report.append("")
        
        # System tools
        report.append("## System Tools")
        tools = check_results["system"]["tools"]
        available_tools = [name for name, info in tools.items() if info["available"]]
        missing_tools = [name for name, info in tools.items() if not info["available"]]
        
        report.append(f"Available: {', '.join(available_tools)}")
        if missing_tools:
            report.append(f"Missing: {', '.join(missing_tools)}")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], output_file: str = ".github/dependency-check.json"):
        """Save dependency check results to a JSON file."""
        output_path = self.repo_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Dependency check results saved to {output_path}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and update project dependencies")
    parser.add_argument("--check", action="store_true", help="Check dependency status")
    parser.add_argument("--update", choices=["python", "node", "all"], help="Update dependencies")
    parser.add_argument("--update-type", choices=["minor", "security"], default="minor", 
                       help="Type of updates to apply")
    parser.add_argument("--output", "-o", default=".github/dependency-check.json", 
                       help="Output file for check results")
    parser.add_argument("--report", "-r", help="Generate human-readable report to file")
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    
    args = parser.parse_args()
    
    updater = DependencyUpdater(args.repo_path)
    
    if args.check or not args.update:
        # Run dependency check
        results = updater.check_all_dependencies()
        updater.save_results(results, args.output)
        
        if args.report:
            report = updater.generate_update_report(results)
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"📋 Report saved to {args.report}")
        
        # Print summary
        print("\n📦 Dependency Check Summary:")
        print(f"  - Python: {'✅' if results['python']['found'] else '❌'}")
        print(f"  - Node.js: {'✅' if results['node']['found'] else '❌'}")
        print(f"  - Docker: {'✅' if results['docker']['found'] else '❌'}")
        print(f"  - GitHub Actions: {'✅' if results['github_actions']['found'] else '❌'}")
    
    if args.update:
        # Run dependency updates
        if args.update in ["python", "all"]:
            print("🔄 Updating Python dependencies...")
            python_result = updater.update_python_dependencies(args.update_type)
            if python_result["success"]:
                print(f"✅ Updated {len(python_result['updates'])} Python packages")
            else:
                print(f"❌ Python update failed: {python_result['errors']}")
        
        if args.update in ["node", "all"]:
            print("🔄 Updating Node.js dependencies...")
            node_result = updater.update_node_dependencies(args.update_type)
            if node_result["success"]:
                print("✅ Node.js dependencies updated")
            else:
                print(f"❌ Node.js update failed: {node_result['errors']}")


if __name__ == "__main__":
    main()