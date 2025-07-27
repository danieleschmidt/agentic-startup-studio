#!/usr/bin/env python3
"""
Comprehensive Security Scanning Script for Agentic Startup Studio

This script runs multiple security scans and generates a comprehensive security report.
It includes dependency scanning, static analysis, container security, and more.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class SecurityScanner:
    """Comprehensive security scanner for the application."""
    
    def __init__(self, output_dir: str = "security-reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def run_command(self, command: List[str], description: str) -> Tuple[bool, str, str]:
        """Run a command and return success status, stdout, stderr."""
        try:
            print(f"🔍 Running: {description}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except FileNotFoundError:
            return False, "", f"Command not found: {command[0]}"
        except Exception as e:
            return False, "", f"Error running command: {str(e)}"
    
    def scan_dependencies(self) -> Dict:
        """Scan Python dependencies for known vulnerabilities."""
        print("\n🔒 Scanning Python Dependencies...")
        
        # Safety scan
        safety_success, safety_output, safety_error = self.run_command(
            ["safety", "check", "--json", "--ignore", "51457"],  # Ignore specific non-critical CVE
            "Safety vulnerability scan"
        )
        
        # Bandit scan
        bandit_success, bandit_output, bandit_error = self.run_command(
            ["bandit", "-r", ".", "-f", "json", "-x", "tests,venv,.venv,build,dist"],
            "Bandit static security analysis"
        )
        
        # Pip-audit scan (if available)
        audit_success, audit_output, audit_error = self.run_command(
            ["pip-audit", "--format=json"],
            "pip-audit vulnerability scan"
        )
        
        return {
            "safety": {
                "success": safety_success,
                "output": safety_output,
                "error": safety_error,
                "vulnerabilities_found": "vulnerabilities" in safety_output.lower() if safety_output else False
            },
            "bandit": {
                "success": bandit_success,
                "output": bandit_output,
                "error": bandit_error,
                "issues_found": self._parse_bandit_results(bandit_output) if bandit_success else 0
            },
            "pip_audit": {
                "success": audit_success,
                "output": audit_output,
                "error": audit_error
            }
        }
    
    def _parse_bandit_results(self, bandit_output: str) -> int:
        """Parse Bandit JSON output to count issues."""
        try:
            data = json.loads(bandit_output)
            return len(data.get("results", []))
        except (json.JSONDecodeError, KeyError):
            return 0
    
    def scan_containers(self) -> Dict:
        """Scan container images for vulnerabilities."""
        print("\n🐳 Scanning Container Images...")
        
        # Build the image first
        build_success, build_output, build_error = self.run_command(
            ["docker", "build", "-t", "agentic-studio:security-scan", "."],
            "Building container image for scanning"
        )
        
        if not build_success:
            return {
                "build_failed": True,
                "error": build_error
            }
        
        # Trivy scan
        trivy_success, trivy_output, trivy_error = self.run_command(
            ["trivy", "image", "--format", "json", "--severity", "HIGH,CRITICAL", "agentic-studio:security-scan"],
            "Trivy container vulnerability scan"
        )
        
        # Docker Scout scan (if available)
        scout_success, scout_output, scout_error = self.run_command(
            ["docker", "scout", "cves", "--format", "json", "agentic-studio:security-scan"],
            "Docker Scout vulnerability scan"
        )
        
        return {
            "trivy": {
                "success": trivy_success,
                "output": trivy_output,
                "error": trivy_error,
                "vulnerabilities": self._parse_trivy_results(trivy_output) if trivy_success else {}
            },
            "docker_scout": {
                "success": scout_success,
                "output": scout_output,
                "error": scout_error
            }
        }
    
    def _parse_trivy_results(self, trivy_output: str) -> Dict:
        """Parse Trivy JSON output to extract vulnerability counts."""
        try:
            data = json.loads(trivy_output)
            results = data.get("Results", [])
            total_vulnerabilities = 0
            by_severity = {"HIGH": 0, "CRITICAL": 0}
            
            for result in results:
                vulnerabilities = result.get("Vulnerabilities", [])
                total_vulnerabilities += len(vulnerabilities)
                
                for vuln in vulnerabilities:
                    severity = vuln.get("Severity", "UNKNOWN")
                    if severity in by_severity:
                        by_severity[severity] += 1
            
            return {
                "total": total_vulnerabilities,
                "by_severity": by_severity
            }
        except (json.JSONDecodeError, KeyError):
            return {"total": 0, "by_severity": {}}
    
    def scan_secrets(self) -> Dict:
        """Scan for exposed secrets in the codebase."""
        print("\n🔐 Scanning for Exposed Secrets...")
        
        # TruffleHog scan (if available)
        trufflehog_success, trufflehog_output, trufflehog_error = self.run_command(
            ["trufflehog", "filesystem", ".", "--json"],
            "TruffleHog secrets scan"
        )
        
        # detect-secrets scan
        detect_secrets_success, detect_secrets_output, detect_secrets_error = self.run_command(
            ["detect-secrets", "scan", "--all-files", ".", "--force-use-all-plugins"],
            "detect-secrets scan"
        )
        
        # Custom patterns scan
        custom_scan_results = self._scan_custom_patterns()
        
        return {
            "trufflehog": {
                "success": trufflehog_success,
                "output": trufflehog_output,
                "error": trufflehog_error
            },
            "detect_secrets": {
                "success": detect_secrets_success,
                "output": detect_secrets_output,
                "error": detect_secrets_error
            },
            "custom_patterns": custom_scan_results
        }
    
    def _scan_custom_patterns(self) -> Dict:
        """Scan for custom secret patterns."""
        dangerous_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
        ]
        
        findings = []
        try:
            for pattern in dangerous_patterns:
                grep_success, grep_output, _ = self.run_command(
                    ["grep", "-r", "-n", "-i", "--include=*.py", "--include=*.yml", "--include=*.yaml", pattern, "."],
                    f"Scanning for pattern: {pattern}"
                )
                if grep_success and grep_output:
                    findings.extend(grep_output.strip().split('\n'))
        except Exception as e:
            return {"error": str(e), "findings": []}
        
        return {"findings": findings, "count": len(findings)}
    
    def scan_infrastructure(self) -> Dict:
        """Scan infrastructure as code for security issues."""
        print("\n🏗️ Scanning Infrastructure as Code...")
        
        # Checkov scan
        checkov_success, checkov_output, checkov_error = self.run_command(
            ["checkov", "-f", "docker-compose.yml", "-f", "Dockerfile", "--framework", "dockerfile,docker_compose", "--output", "json"],
            "Checkov infrastructure security scan"
        )
        
        # YAML security scan
        yaml_scan_results = self._scan_yaml_files()
        
        return {
            "checkov": {
                "success": checkov_success,
                "output": checkov_output,
                "error": checkov_error
            },
            "yaml_security": yaml_scan_results
        }
    
    def _scan_yaml_files(self) -> Dict:
        """Scan YAML files for security configurations."""
        yaml_files = list(Path(".").rglob("*.yml")) + list(Path(".").rglob("*.yaml"))
        issues = []
        
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    content = yaml.safe_load(f)
                    
                # Check for common security issues
                if isinstance(content, dict):
                    if self._check_for_plaintext_secrets(content):
                        issues.append(f"{yaml_file}: Potential plaintext secrets found")
                    
                    if self._check_for_privileged_containers(content):
                        issues.append(f"{yaml_file}: Privileged container configuration found")
                        
            except Exception as e:
                issues.append(f"{yaml_file}: Error parsing - {str(e)}")
        
        return {"issues": issues, "files_scanned": len(yaml_files)}
    
    def _check_for_plaintext_secrets(self, data) -> bool:
        """Check for potential plaintext secrets in YAML data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and any(secret_word in key.lower() for secret_word in ["password", "secret", "key", "token"]):
                    if isinstance(value, str) and not value.startswith("${") and len(value) > 5:
                        return True
                if isinstance(value, (dict, list)):
                    if self._check_for_plaintext_secrets(value):
                        return True
        elif isinstance(data, list):
            for item in data:
                if self._check_for_plaintext_secrets(item):
                    return True
        return False
    
    def _check_for_privileged_containers(self, data) -> bool:
        """Check for privileged container configurations."""
        if isinstance(data, dict):
            if data.get("privileged") is True:
                return True
            for value in data.values():
                if isinstance(value, (dict, list)):
                    if self._check_for_privileged_containers(value):
                        return True
        elif isinstance(data, list):
            for item in data:
                if self._check_for_privileged_containers(item):
                    return True
        return False
    
    def generate_report(self) -> str:
        """Generate a comprehensive security report."""
        report_file = self.output_dir / f"security_report_{self.timestamp}.json"
        
        # Calculate overall security score
        score = self._calculate_security_score()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "security_score": score,
            "summary": self._generate_summary(),
            "results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        # Save JSON report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        md_report_file = self.output_dir / f"security_report_{self.timestamp}.md"
        self._generate_markdown_report(report, md_report_file)
        
        print(f"\n📊 Security reports generated:")
        print(f"   JSON: {report_file}")
        print(f"   Markdown: {md_report_file}")
        
        return str(report_file)
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security score (0-100)."""
        score = 100
        
        # Deduct points for various issues
        if "dependencies" in self.results:
            deps = self.results["dependencies"]
            if deps.get("safety", {}).get("vulnerabilities_found", False):
                score -= 20
            bandit_issues = deps.get("bandit", {}).get("issues_found", 0)
            score -= min(bandit_issues * 2, 15)
        
        if "containers" in self.results:
            containers = self.results["containers"]
            trivy_vulns = containers.get("trivy", {}).get("vulnerabilities", {}).get("total", 0)
            score -= min(trivy_vulns * 3, 25)
        
        if "secrets" in self.results:
            secrets = self.results["secrets"]
            custom_findings = secrets.get("custom_patterns", {}).get("count", 0)
            score -= min(custom_findings * 10, 30)
        
        return max(score, 0)
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of findings."""
        summary = {
            "total_scans_run": len(self.results),
            "critical_issues": 0,
            "warnings": 0,
            "info": 0
        }
        
        # Count issues by severity
        for scan_type, scan_results in self.results.items():
            if isinstance(scan_results, dict):
                # This is a simplified counting - in reality, you'd parse each scan's output format
                if "error" in str(scan_results).lower():
                    summary["warnings"] += 1
                if "critical" in str(scan_results).lower():
                    summary["critical_issues"] += 1
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        if "dependencies" in self.results:
            deps = self.results["dependencies"]
            if deps.get("safety", {}).get("vulnerabilities_found", False):
                recommendations.append("Update vulnerable dependencies identified by Safety scan")
            if deps.get("bandit", {}).get("issues_found", 0) > 0:
                recommendations.append("Review and fix security issues identified by Bandit")
        
        if "containers" in self.results:
            containers = self.results["containers"]
            if containers.get("trivy", {}).get("vulnerabilities", {}).get("total", 0) > 0:
                recommendations.append("Update base images and fix container vulnerabilities")
        
        if "secrets" in self.results:
            secrets = self.results["secrets"]
            if secrets.get("custom_patterns", {}).get("count", 0) > 0:
                recommendations.append("Remove hardcoded secrets and use environment variables")
        
        if not recommendations:
            recommendations.append("No critical security issues found. Continue monitoring.")
        
        return recommendations
    
    def _generate_markdown_report(self, report: Dict, output_file: Path):
        """Generate a markdown version of the security report."""
        md_content = f"""# Security Scan Report

**Generated:** {report['timestamp']}  
**Security Score:** {report['security_score']}/100

## Executive Summary

{self._format_summary_for_markdown(report['summary'])}

## Scan Results

### Dependency Scanning
{self._format_dependencies_for_markdown()}

### Container Security
{self._format_containers_for_markdown()}

### Secrets Scanning
{self._format_secrets_for_markdown()}

### Infrastructure Security
{self._format_infrastructure_for_markdown()}

## Recommendations

{chr(10).join(f"- {rec}" for rec in report['recommendations'])}

## Next Steps

1. Review and prioritize the findings above
2. Create tickets for remediation work
3. Schedule regular security scans
4. Update security documentation

---
*This report was generated automatically by the security scanning tool.*
"""
        
        with open(output_file, 'w') as f:
            f.write(md_content)
    
    def _format_summary_for_markdown(self, summary: Dict) -> str:
        """Format summary section for markdown."""
        return f"""
- **Total Scans:** {summary['total_scans_run']}
- **Critical Issues:** {summary['critical_issues']}
- **Warnings:** {summary['warnings']}
- **Info Items:** {summary['info']}
"""
    
    def _format_dependencies_for_markdown(self) -> str:
        """Format dependency scanning results for markdown."""
        if "dependencies" not in self.results:
            return "No dependency scan results available."
        
        deps = self.results["dependencies"]
        content = []
        
        for tool, results in deps.items():
            status = "✅" if results.get("success", False) else "❌"
            content.append(f"- **{tool.title()}:** {status}")
            if results.get("error"):
                content.append(f"  - Error: {results['error']}")
        
        return "\n".join(content)
    
    def _format_containers_for_markdown(self) -> str:
        """Format container scanning results for markdown."""
        if "containers" not in self.results:
            return "No container scan results available."
        
        containers = self.results["containers"]
        if containers.get("build_failed"):
            return f"❌ Container build failed: {containers.get('error', 'Unknown error')}"
        
        trivy = containers.get("trivy", {})
        if trivy.get("success"):
            vulns = trivy.get("vulnerabilities", {})
            total = vulns.get("total", 0)
            by_severity = vulns.get("by_severity", {})
            
            return f"""
- **Trivy Scan:** ✅
  - Total Vulnerabilities: {total}
  - High Severity: {by_severity.get('HIGH', 0)}
  - Critical Severity: {by_severity.get('CRITICAL', 0)}
"""
        else:
            return f"❌ Trivy scan failed: {trivy.get('error', 'Unknown error')}"
    
    def _format_secrets_for_markdown(self) -> str:
        """Format secrets scanning results for markdown."""
        if "secrets" not in self.results:
            return "No secrets scan results available."
        
        secrets = self.results["secrets"]
        content = []
        
        custom = secrets.get("custom_patterns", {})
        if "findings" in custom:
            count = custom.get("count", 0)
            status = "⚠️" if count > 0 else "✅"
            content.append(f"- **Custom Pattern Scan:** {status} ({count} findings)")
        
        return "\n".join(content) if content else "No secrets scanning results available."
    
    def _format_infrastructure_for_markdown(self) -> str:
        """Format infrastructure scanning results for markdown."""
        if "infrastructure" not in self.results:
            return "No infrastructure scan results available."
        
        infra = self.results["infrastructure"]
        content = []
        
        checkov = infra.get("checkov", {})
        status = "✅" if checkov.get("success", False) else "❌"
        content.append(f"- **Checkov:** {status}")
        
        yaml_scan = infra.get("yaml_security", {})
        if "issues" in yaml_scan:
            issue_count = len(yaml_scan.get("issues", []))
            files_scanned = yaml_scan.get("files_scanned", 0)
            status = "⚠️" if issue_count > 0 else "✅"
            content.append(f"- **YAML Security:** {status} ({issue_count} issues in {files_scanned} files)")
        
        return "\n".join(content)
    
    def run_all_scans(self):
        """Run all security scans."""
        print("🔒 Starting Comprehensive Security Scan...")
        print(f"⏰ Timestamp: {datetime.now().isoformat()}")
        
        # Run all scans
        self.results["dependencies"] = self.scan_dependencies()
        self.results["containers"] = self.scan_containers()
        self.results["secrets"] = self.scan_secrets()
        self.results["infrastructure"] = self.scan_infrastructure()
        
        # Generate report
        report_file = self.generate_report()
        
        print(f"\n✅ Security scan completed!")
        print(f"📊 Security Score: {self._calculate_security_score()}/100")
        
        return report_file


def main():
    """Main function to run security scans."""
    parser = argparse.ArgumentParser(description="Comprehensive security scanner for Agentic Startup Studio")
    parser.add_argument("--output-dir", default="security-reports", help="Output directory for reports")
    parser.add_argument("--scan-type", choices=["all", "dependencies", "containers", "secrets", "infrastructure"], 
                       default="all", help="Type of scan to run")
    parser.add_argument("--json-only", action="store_true", help="Only output JSON report")
    parser.add_argument("--ci-mode", action="store_true", help="Run in CI mode (exit with error code if issues found)")
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(args.output_dir)
    
    try:
        if args.scan_type == "all":
            report_file = scanner.run_all_scans()
        elif args.scan_type == "dependencies":
            scanner.results["dependencies"] = scanner.scan_dependencies()
            report_file = scanner.generate_report()
        elif args.scan_type == "containers":
            scanner.results["containers"] = scanner.scan_containers()
            report_file = scanner.generate_report()
        elif args.scan_type == "secrets":
            scanner.results["secrets"] = scanner.scan_secrets()
            report_file = scanner.generate_report()
        elif args.scan_type == "infrastructure":
            scanner.results["infrastructure"] = scanner.scan_infrastructure()
            report_file = scanner.generate_report()
        
        # Check for CI mode
        if args.ci_mode:
            security_score = scanner._calculate_security_score()
            if security_score < 80:  # Threshold for CI failure
                print(f"❌ Security score {security_score} below threshold (80). Failing CI.")
                sys.exit(1)
            else:
                print(f"✅ Security score {security_score} meets threshold.")
        
        print(f"\n🎉 Security scan completed successfully!")
        print(f"📄 Report available at: {report_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Security scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Security scan failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()