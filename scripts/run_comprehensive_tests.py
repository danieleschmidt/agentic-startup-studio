#!/usr/bin/env python3

"""Comprehensive test runner for Agentic Startup Studio."""

import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone


class ComprehensiveTestRunner:
    """Runs all types of tests and generates comprehensive reports."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = datetime.now(timezone.utc)
        self.results = {}
        self.project_root = Path(__file__).parent.parent
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🔍" if level == "INFO" else "⚠️" if level == "WARN" else "❌"
        print(f"{prefix} [{timestamp}] {message}")
    
    def run_command(self, cmd: List[str], description: str, timeout: int = 300) -> Dict[str, Any]:
        """Run a command and capture results."""
        self.log(f"Running {description}...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            status = "✅ PASSED" if success else "❌ FAILED"
            
            self.log(f"{status} {description} ({duration:.1f}s)")
            
            if not success and self.verbose:
                self.log(f"STDOUT: {result.stdout}")
                self.log(f"STDERR: {result.stderr}")
            
            return {
                "success": success,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
            
        except subprocess.TimeoutExpired:
            self.log(f"❌ TIMEOUT {description} (>{timeout}s)", "ERROR")
            return {
                "success": False,
                "duration": timeout,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": -1,
            }
        except Exception as e:
            self.log(f"❌ ERROR {description}: {e}", "ERROR")
            return {
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
            }
    
    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage."""
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "-m", "unit",
            "--cov=pipeline",
            "--cov=core",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=90",
            "--maxfail=10",
            "--tb=short",
            "-v" if self.verbose else "-q",
        ]
        
        result = self.run_command(cmd, "Unit Tests with Coverage")
        self.results["unit_tests"] = result
        return result["success"]
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/",
            "tests/pipeline/",
            "-m", "integration",
            "--maxfail=5",
            "--tb=short",
            "-v" if self.verbose else "-q",
        ]
        
        result = self.run_command(cmd, "Integration Tests")
        self.results["integration_tests"] = result
        return result["success"]
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests with Playwright."""
        # Check if Playwright is available
        if not (self.project_root / "playwright.config.ts").exists():
            self.log("Playwright config not found, skipping E2E tests", "WARN")
            self.results["e2e_tests"] = {"success": True, "duration": 0, "skipped": True}
            return True
        
        cmd = ["npx", "playwright", "test", "--reporter=json"]
        result = self.run_command(cmd, "End-to-End Tests")
        self.results["e2e_tests"] = result
        return result["success"]
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "-m", "performance",
            "--maxfail=3",
            "--tb=short",
            "-v" if self.verbose else "-q",
        ]
        
        result = self.run_command(cmd, "Performance Tests")
        self.results["performance_tests"] = result
        return result["success"]
    
    def run_security_tests(self) -> bool:
        """Run security tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/security/",
            "-m", "security",
            "--maxfail=3",
            "--tb=short",
            "-v" if self.verbose else "-q",
        ]
        
        result = self.run_command(cmd, "Security Tests")
        self.results["security_tests"] = result
        return result["success"]
    
    def run_contract_tests(self) -> bool:
        """Run API contract tests."""
        cmd = [
            "python", "-m", "pytest",
            "tests/contract/",
            "--maxfail=5",
            "--tb=short",
            "-v" if self.verbose else "-q",
        ]
        
        result = self.run_command(cmd, "Contract Tests")
        self.results["contract_tests"] = result
        return result["success"]
    
    def run_lint_checks(self) -> bool:
        """Run code quality checks."""
        checks = [
            (["ruff", "check", "."], "Ruff Linting"),
            (["ruff", "format", "--check", "."], "Ruff Formatting"),
            (["mypy", "pipeline/", "core/"], "MyPy Type Checking"),
            (["bandit", "-r", "pipeline/", "core/", "-f", "json"], "Bandit Security Scan"),
        ]
        
        all_passed = True
        lint_results = {}
        
        for cmd, description in checks:
            result = self.run_command(cmd, description)
            lint_results[description.lower().replace(" ", "_")] = result
            if not result["success"]:
                all_passed = False
        
        self.results["lint_checks"] = {
            "success": all_passed,
            "duration": sum(r["duration"] for r in lint_results.values()),
            "details": lint_results,
        }
        
        return all_passed
    
    def run_mutation_tests(self) -> bool:
        """Run mutation tests (if enabled)."""
        # Only run mutation tests if specifically requested
        # They're typically too slow for regular CI
        if not hasattr(self, "_run_mutation"):
            self.results["mutation_tests"] = {"success": True, "duration": 0, "skipped": True}
            return True
        
        cmd = ["python", "tests/mutation/mutmut_config.py", "--critical-only"]
        result = self.run_command(cmd, "Mutation Tests", timeout=600)
        self.results["mutation_tests"] = result
        return result["success"]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("success", False))
        failed_tests = total_tests - passed_tests
        skipped_tests = sum(1 for r in self.results.values() if r.get("skipped", False))
        
        report = {
            "timestamp": end_time.isoformat(),
            "duration": total_duration,
            "summary": {
                "total_test_suites": total_tests,
                "passed_test_suites": passed_tests,
                "failed_test_suites": failed_tests,
                "skipped_test_suites": skipped_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "overall_status": "PASSED" if failed_tests == 0 else "FAILED",
            },
            "results": self.results,
            "environment": {
                "python_version": sys.version,
                "project_root": str(self.project_root),
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save test report to file."""
        reports_dir = self.project_root / "tests" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"comprehensive_test_report_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log(f"📊 Test report saved to {report_file}")
        return report_file
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print test summary to console."""
        summary = report["summary"]
        
        print("\n" + "=" * 60)
        print("🧪 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary.get('duration', 0):.1f}s")
        print(f"Test Suites: {summary['passed_test_suites']}/{summary['total_test_suites']} passed")
        
        if summary['skipped_test_suites'] > 0:
            print(f"Skipped: {summary['skipped_test_suites']} test suites")
        
        print("\n📋 Test Suite Results:")
        
        for test_name, result in self.results.items():
            status = "SKIPPED" if result.get("skipped") else ("PASSED" if result["success"] else "FAILED")
            duration = result.get("duration", 0)
            emoji = "⏭️" if status == "SKIPPED" else ("✅" if status == "PASSED" else "❌")
            
            print(f"  {emoji} {test_name.replace('_', ' ').title()}: {status} ({duration:.1f}s)")
        
        print("=" * 60)
        
        if summary["overall_status"] == "FAILED":
            print("❌ Some tests failed. Check the detailed report for more information.")
        else:
            print("✅ All tests passed! Your code is ready for deployment.")
    
    def run_all_tests(self, test_types: Optional[List[str]] = None) -> bool:
        """Run all or specified test types."""
        available_tests = {
            "lint": self.run_lint_checks,
            "unit": self.run_unit_tests,
            "integration": self.run_integration_tests,
            "contract": self.run_contract_tests,
            "performance": self.run_performance_tests,
            "security": self.run_security_tests,
            "e2e": self.run_e2e_tests,
            "mutation": self.run_mutation_tests,
        }
        
        if test_types:
            tests_to_run = {k: v for k, v in available_tests.items() if k in test_types}
        else:
            tests_to_run = available_tests
        
        self.log(f"🚀 Starting comprehensive test suite ({len(tests_to_run)} test types)")
        
        overall_success = True
        
        for test_name, test_func in tests_to_run.items():
            if not test_func():
                overall_success = False
        
        # Generate and save report
        report = self.generate_report()
        self.save_report(report)
        self.print_summary(report)
        
        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive tests for Agentic Startup Studio")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["lint", "unit", "integration", "contract", "performance", "security", "e2e", "mutation"],
        help="Specific test types to run (default: all except mutation)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--include-mutation",
        action="store_true", 
        help="Include mutation tests (slow)"
    )
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(verbose=args.verbose)
    
    if args.include_mutation:
        runner._run_mutation = True
    
    test_types = args.types
    if not test_types and args.include_mutation:
        test_types = ["lint", "unit", "integration", "contract", "performance", "security", "e2e", "mutation"]
    
    success = runner.run_all_tests(test_types)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()