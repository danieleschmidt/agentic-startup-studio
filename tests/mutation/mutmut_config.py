"""Mutation testing configuration using mutmut."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def pre_mutation(context):
    """Hook called before each mutation is applied."""
    pass


def post_mutation(context, filename, mutant, index, lines_to_mutate):
    """Hook called after each mutation is applied."""
    pass


# Mutmut configuration
MUTMUT_CONFIG = {
    # Paths to mutate
    "paths_to_mutate": [
        "pipeline/",
        "core/",
        "src/",
    ],
    
    # Paths to exclude from mutation
    "paths_to_exclude": [
        "tests/",
        "docs/",
        "scripts/",
        "__pycache__/",
        ".git/",
        "venv/",
        ".venv/",
        "build/",
        "dist/",
        "*.egg-info/",
    ],
    
    # File patterns to exclude
    "patterns_to_exclude": [
        "*_test.py",
        "test_*.py", 
        "conftest.py",
        "__init__.py",
        "migrations/",
        "alembic/",
    ],
    
    # Test command to run for each mutation
    "test_command": "python -m pytest tests/ -x --tb=no --disable-warnings",
    
    # Timeout for each test run (seconds)
    "timeout": 60,
    
    # Minimum test coverage required before mutation testing
    "minimum_coverage": 90,
    
    # Mutations to apply
    "mutations": [
        "conditional_boundary",  # < to <=, > to >=, etc.
        "math",                  # + to -, * to /, etc.
        "logical_operator",      # and to or, etc.
        "comparison_operator",   # == to !=, etc.
        "unary_operator",        # not to empty, - to +
        "assignment_operator",   # += to -=, etc.
        "constant",              # True to False, 0 to 1, etc.
        "function_call",         # Remove function calls
        "index",                 # Change array/dict indices
        "attribute",             # Change attribute access
    ],
    
    # Mutations to skip (too noisy or irrelevant)
    "skip_mutations": [
        "string",               # String mutations often break imports
        "decorator",            # Decorator mutations can break functionality
        "import",               # Import mutations break execution
    ],
}


class MutationTestRunner:
    """Custom mutation test runner with advanced features."""
    
    def __init__(self, config=None):
        self.config = config or MUTMUT_CONFIG
        self.results = []
    
    def run_mutation_tests(self, target_modules=None):
        """Run mutation tests on specified modules."""
        import subprocess
        import json
        from datetime import datetime
        
        target_modules = target_modules or self.config["paths_to_mutate"]
        
        print("🧬 Starting mutation testing...")
        print(f"Target modules: {target_modules}")
        print(f"Test command: {self.config['test_command']}")
        
        # Run mutmut
        cmd = [
            "mutmut",
            "run",
            "--paths-to-mutate", ",".join(target_modules),
            "--paths-to-exclude", ",".join(self.config["paths_to_exclude"]),
            "--tests-dir", "tests/",
            "--runner", self.config["test_command"],
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout", 300)
            )
            
            # Parse results
            self._parse_mutation_results(result.stdout)
            
            # Generate report
            self._generate_mutation_report()
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("❌ Mutation testing timed out")
            return False
        except Exception as e:
            print(f"❌ Mutation testing failed: {e}")
            return False
    
    def _parse_mutation_results(self, output):
        """Parse mutation test results from output."""
        lines = output.split('\n')
        
        for line in lines:
            if 'mutations:' in line.lower():
                # Parse mutation statistics
                # This is a simplified parser - real implementation would be more robust
                pass
    
    def _generate_mutation_report(self):
        """Generate mutation testing report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": self.results,
            "summary": {
                "total_mutations": len(self.results),
                "killed_mutations": sum(1 for r in self.results if r.get("killed", False)),
                "survived_mutations": sum(1 for r in self.results if not r.get("killed", True)),
            }
        }
        
        # Calculate mutation score
        if report["summary"]["total_mutations"] > 0:
            report["summary"]["mutation_score"] = (
                report["summary"]["killed_mutations"] / 
                report["summary"]["total_mutations"] * 100
            )
        else:
            report["summary"]["mutation_score"] = 0.0
        
        # Save report
        report_path = Path("tests/reports/mutation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Mutation report saved to {report_path}")
        print(f"🎯 Mutation score: {report['summary']['mutation_score']:.1f}%")


def run_targeted_mutation_tests():
    """Run mutation tests on critical modules only."""
    critical_modules = [
        "pipeline/models/",
        "pipeline/ingestion/validators.py",
        "pipeline/storage/idea_repository.py",
        "core/interfaces.py",
    ]
    
    runner = MutationTestRunner()
    return runner.run_mutation_tests(critical_modules)


def run_full_mutation_tests():
    """Run comprehensive mutation tests on all modules."""
    runner = MutationTestRunner()
    return runner.run_mutation_tests()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mutation tests")
    parser.add_argument("--critical-only", action="store_true", 
                       help="Run mutation tests on critical modules only")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout for mutation testing in seconds")
    
    args = parser.parse_args()
    
    if args.critical_only:
        success = run_targeted_mutation_tests()
    else:
        success = run_full_mutation_tests()
    
    sys.exit(0 if success else 1)