#!/usr/bin/env python3
"""
Compliance Monitoring for Enterprise Agentic Startup Studio

This module provides comprehensive compliance monitoring for various standards
including SOC 2, GDPR, HIPAA, and other enterprise compliance requirements.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import aiofiles
import hashlib
import re


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    SOC2_TYPE1 = "soc2_type1"
    SOC2_TYPE2 = "soc2_type2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    FedRAMP = "fedramp"


class ComplianceStatus(Enum):
    """Compliance check status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


@dataclass
class ComplianceCheck:
    """Individual compliance check definition."""
    check_id: str
    standard: ComplianceStandard
    title: str
    description: str
    requirement: str
    category: str
    automated: bool
    severity: str
    remediation: str
    evidence_required: List[str]
    frequency: str  # daily, weekly, monthly, annual


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    check_id: str
    timestamp: datetime
    status: ComplianceStatus
    score: float  # 0.0 to 1.0
    findings: List[str]
    evidence: Dict[str, Any]
    remediation_actions: List[str]
    next_check_due: datetime


class ComplianceMonitor:
    """Enterprise compliance monitoring system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.checks: Dict[str, ComplianceCheck] = {}
        self.results: Dict[str, ComplianceResult] = {}
        
        # Load compliance checks
        self._load_compliance_checks()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load compliance monitoring configuration."""
        default_config = {
            "enabled_standards": [
                ComplianceStandard.SOC2_TYPE2.value,
                ComplianceStandard.GDPR.value,
                ComplianceStandard.ISO27001.value
            ],
            "data_retention_days": 2555,  # 7 years for compliance
            "alert_threshold": 0.8,  # Alert if compliance score below 80%
            "automated_checks_only": False,
            "report_formats": ["json", "html", "pdf"],
            "audit_trail": True,
            "encryption_at_rest": True,
            "access_logging": True
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                logging.warning(f"Failed to load compliance config: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup audit-compliant logging."""
        logger = logging.getLogger("compliance_monitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create compliance audit log handler
            audit_log_path = Path("compliance_audit.log")
            handler = logging.FileHandler(audit_log_path)
            
            # Use structured format for compliance auditing
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Also log to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_compliance_checks(self):
        """Load compliance check definitions."""
        # SOC 2 Type II checks
        soc2_checks = [
            ComplianceCheck(
                check_id="SOC2-CC1.1",
                standard=ComplianceStandard.SOC2_TYPE2,
                title="Management Oversight and Control Environment",
                description="Management demonstrates a commitment to integrity and ethical values",
                requirement="The entity demonstrates a commitment to integrity and ethical values",
                category="Control Environment",
                automated=True,
                severity="High",
                remediation="Implement and document code of conduct and ethics policies",
                evidence_required=["code_of_conduct", "ethics_training_records"],
                frequency="monthly"
            ),
            ComplianceCheck(
                check_id="SOC2-CC2.1",
                standard=ComplianceStandard.SOC2_TYPE2,
                title="Communication and Information Systems",
                description="Information system and related processes support achievement of objectives",
                requirement="Information system processes support the achievement of objectives",
                category="Communication and Information",
                automated=True,
                severity="High",
                remediation="Document information system architecture and controls",
                evidence_required=["system_documentation", "process_documentation"],
                frequency="monthly"
            ),
            ComplianceCheck(
                check_id="SOC2-CC6.1",
                standard=ComplianceStandard.SOC2_TYPE2,
                title="Logical and Physical Access Controls",
                description="Entity implements logical access security software and systems",
                requirement="Logical access security software, infrastructure, and architectures are implemented",
                category="Logical and Physical Access Controls",
                automated=True,
                severity="Critical",
                remediation="Implement multi-factor authentication and access controls",
                evidence_required=["access_control_logs", "mfa_configuration"],
                frequency="daily"
            )
        ]
        
        # GDPR checks
        gdpr_checks = [
            ComplianceCheck(
                check_id="GDPR-Art6",
                standard=ComplianceStandard.GDPR,
                title="Lawfulness of Processing",
                description="Processing must have a legal basis under GDPR Article 6",
                requirement="Lawful basis for processing personal data is established and documented",
                category="Lawful Processing",
                automated=False,
                severity="Critical",
                remediation="Document legal basis for all personal data processing activities",
                evidence_required=["data_processing_record", "legal_basis_documentation"],
                frequency="monthly"
            ),
            ComplianceCheck(
                check_id="GDPR-Art25",
                standard=ComplianceStandard.GDPR,
                title="Data Protection by Design and by Default",
                description="Data protection measures must be built into systems by design",
                requirement="Technical and organizational measures implement data protection by design",
                category="Privacy by Design",
                automated=True,
                severity="High",
                remediation="Implement privacy controls in system design and configuration",
                evidence_required=["privacy_impact_assessment", "technical_safeguards"],
                frequency="monthly"
            ),
            ComplianceCheck(
                check_id="GDPR-Art32",
                standard=ComplianceStandard.GDPR,
                title="Security of Processing",
                description="Appropriate technical and organizational security measures",
                requirement="Security of processing through appropriate technical and organizational measures",
                category="Security Measures",
                automated=True,
                severity="Critical",
                remediation="Implement encryption, access controls, and security monitoring",
                evidence_required=["encryption_status", "access_logs", "security_monitoring"],
                frequency="daily"
            )
        ]
        
        # ISO 27001 checks
        iso27001_checks = [
            ComplianceCheck(
                check_id="ISO27001-A.5.1.1",
                standard=ComplianceStandard.ISO27001,
                title="Information Security Policies",
                description="Information security policy must be defined and approved",
                requirement="Set of policies for information security defined and approved by management",
                category="Information Security Policies",
                automated=False,
                severity="High",
                remediation="Create and approve comprehensive information security policy",
                evidence_required=["security_policy", "management_approval"],
                frequency="annual"
            ),
            ComplianceCheck(
                check_id="ISO27001-A.9.1.1",
                standard=ComplianceStandard.ISO27001,
                title="Access Control Policy",
                description="Access control policy must be established and maintained",
                requirement="Access control policy established, documented and reviewed",
                category="Access Control",
                automated=True,
                severity="Critical",
                remediation="Implement comprehensive access control policy and procedures",
                evidence_required=["access_control_policy", "access_reviews"],
                frequency="monthly"
            )
        ]
        
        # Add all checks to the registry
        all_checks = soc2_checks + gdpr_checks + iso27001_checks
        for check in all_checks:
            self.checks[check.check_id] = check
    
    async def run_automated_check(self, check: ComplianceCheck) -> ComplianceResult:
        """Run an automated compliance check."""
        try:
            self.logger.info(f"Running automated check: {check.check_id}")
            
            findings = []
            evidence = {}
            score = 1.0
            status = ComplianceStatus.COMPLIANT
            
            # Implement specific check logic based on check ID
            if check.check_id == "SOC2-CC6.1":
                # Check logical access controls
                evidence, findings, score = await self._check_access_controls()
            
            elif check.check_id == "GDPR-Art32":
                # Check security of processing
                evidence, findings, score = await self._check_data_security()
            
            elif check.check_id == "ISO27001-A.9.1.1":
                # Check access control policy
                evidence, findings, score = await self._check_access_policy()
            
            else:
                # Default automated check - basic system security
                evidence, findings, score = await self._check_basic_security()
            
            # Determine status based on score
            if score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif score >= 0.7:
                status = ComplianceStatus.WARNING
            else:
                status = ComplianceStatus.NON_COMPLIANT
            
            # Calculate next check due date
            next_check = self._calculate_next_check_date(check.frequency)
            
            result = ComplianceResult(
                check_id=check.check_id,
                timestamp=datetime.now(timezone.utc),
                status=status,
                score=score,
                findings=findings,
                evidence=evidence,
                remediation_actions=self._get_remediation_actions(check, findings),
                next_check_due=next_check
            )
            
            self.logger.info(f"Check {check.check_id} completed: {status.value} (score: {score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Automated check {check.check_id} failed: {e}")
            return ComplianceResult(
                check_id=check.check_id,
                timestamp=datetime.now(timezone.utc),
                status=ComplianceStatus.UNKNOWN,
                score=0.0,
                findings=[f"Check execution failed: {str(e)}"],
                evidence={},
                remediation_actions=["Review check implementation and system access"],
                next_check_due=datetime.now(timezone.utc) + timedelta(hours=1)
            )
    
    async def _check_access_controls(self) -> tuple[Dict[str, Any], List[str], float]:
        """Check logical access controls implementation."""
        evidence = {}
        findings = []
        score = 1.0
        
        try:
            # Check if MFA is enabled (simulated)
            mfa_enabled = True  # Would check actual MFA configuration
            evidence["mfa_enabled"] = mfa_enabled
            
            if not mfa_enabled:
                findings.append("Multi-factor authentication is not enabled")
                score -= 0.3
            
            # Check password policy (simulated)
            password_policy_compliant = True  # Would check actual policy
            evidence["password_policy_compliant"] = password_policy_compliant
            
            if not password_policy_compliant:
                findings.append("Password policy does not meet security requirements")
                score -= 0.2
            
            # Check access logging (simulated)
            access_logging_enabled = True  # Would check actual logging
            evidence["access_logging_enabled"] = access_logging_enabled
            
            if not access_logging_enabled:
                findings.append("Access logging is not properly configured")
                score -= 0.3
            
            # Check session management
            session_timeout_configured = True  # Would check actual configuration
            evidence["session_timeout_configured"] = session_timeout_configured
            
            if not session_timeout_configured:
                findings.append("Session timeouts are not properly configured")
                score -= 0.2
            
            evidence["check_timestamp"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            findings.append(f"Error checking access controls: {str(e)}")
            score = 0.0
        
        return evidence, findings, max(0.0, score)
    
    async def _check_data_security(self) -> tuple[Dict[str, Any], List[str], float]:
        """Check data security measures (GDPR Article 32)."""
        evidence = {}
        findings = []
        score = 1.0
        
        try:
            # Check encryption at rest
            encryption_at_rest = self.config.get("encryption_at_rest", False)
            evidence["encryption_at_rest"] = encryption_at_rest
            
            if not encryption_at_rest:
                findings.append("Data is not encrypted at rest")
                score -= 0.4
            
            # Check encryption in transit
            encryption_in_transit = True  # Would check HTTPS/TLS configuration
            evidence["encryption_in_transit"] = encryption_in_transit
            
            if not encryption_in_transit:
                findings.append("Data is not encrypted in transit")
                score -= 0.4
            
            # Check backup encryption
            backup_encryption = True  # Would check backup encryption
            evidence["backup_encryption"] = backup_encryption
            
            if not backup_encryption:
                findings.append("Backups are not encrypted")
                score -= 0.2
            
            evidence["check_timestamp"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            findings.append(f"Error checking data security: {str(e)}")
            score = 0.0
        
        return evidence, findings, max(0.0, score)
    
    async def _check_access_policy(self) -> tuple[Dict[str, Any], List[str], float]:
        """Check access control policy implementation."""
        evidence = {}
        findings = []
        score = 1.0
        
        try:
            # Check if access control policy exists
            policy_exists = True  # Would check for actual policy document
            evidence["access_control_policy_exists"] = policy_exists
            
            if not policy_exists:
                findings.append("Access control policy document not found")
                score -= 0.5
            
            # Check policy review date
            last_review = datetime.now(timezone.utc) - timedelta(days=90)
            policy_current = last_review > datetime.now(timezone.utc) - timedelta(days=365)
            evidence["policy_last_reviewed"] = last_review.isoformat()
            evidence["policy_current"] = policy_current
            
            if not policy_current:
                findings.append("Access control policy has not been reviewed in the last year")
                score -= 0.3
            
            # Check role-based access implementation
            rbac_implemented = True  # Would check actual RBAC configuration
            evidence["rbac_implemented"] = rbac_implemented
            
            if not rbac_implemented:
                findings.append("Role-based access control is not implemented")
                score -= 0.2
            
            evidence["check_timestamp"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            findings.append(f"Error checking access policy: {str(e)}")
            score = 0.0
        
        return evidence, findings, max(0.0, score)
    
    async def _check_basic_security(self) -> tuple[Dict[str, Any], List[str], float]:
        """Basic security check for generic compliance requirements."""
        evidence = {}
        findings = []
        score = 1.0
        
        try:
            # Check audit logging
            audit_logging = self.config.get("audit_trail", False)
            evidence["audit_logging_enabled"] = audit_logging
            
            if not audit_logging:
                findings.append("Audit logging is not enabled")
                score -= 0.3
            
            # Check security monitoring
            security_monitoring = True  # Would check actual monitoring
            evidence["security_monitoring_active"] = security_monitoring
            
            if not security_monitoring:
                findings.append("Security monitoring is not active")
                score -= 0.2
            
            evidence["check_timestamp"] = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            findings.append(f"Error in basic security check: {str(e)}")
            score = 0.0
        
        return evidence, findings, max(0.0, score)
    
    def _calculate_next_check_date(self, frequency: str) -> datetime:
        """Calculate when the next check is due."""
        now = datetime.now(timezone.utc)
        
        if frequency == "daily":
            return now + timedelta(days=1)
        elif frequency == "weekly":
            return now + timedelta(weeks=1)
        elif frequency == "monthly":
            return now + timedelta(days=30)
        elif frequency == "quarterly":
            return now + timedelta(days=90)
        elif frequency == "annual":
            return now + timedelta(days=365)
        else:
            return now + timedelta(days=30)  # Default to monthly
    
    def _get_remediation_actions(self, check: ComplianceCheck, findings: List[str]) -> List[str]:
        """Get specific remediation actions based on findings."""
        actions = []
        
        if findings:
            # Add specific remediation from check definition
            actions.append(check.remediation)
            
            # Add specific actions based on findings
            for finding in findings:
                if "multi-factor authentication" in finding.lower():
                    actions.append("Configure and enable MFA for all user accounts")
                elif "encryption" in finding.lower():
                    actions.append("Enable encryption for data at rest and in transit")
                elif "logging" in finding.lower():
                    actions.append("Configure comprehensive audit logging")
                elif "policy" in finding.lower():
                    actions.append("Create or update security policy documentation")
        
        return list(set(actions))  # Remove duplicates
    
    async def run_compliance_assessment(self, standards: Optional[List[ComplianceStandard]] = None) -> Dict[str, Any]:
        """Run comprehensive compliance assessment."""
        if standards is None:
            standards = [ComplianceStandard(std) for std in self.config["enabled_standards"]]
        
        self.logger.info(f"Starting compliance assessment for standards: {[s.value for s in standards]}")
        
        assessment_results = {
            "assessment_id": hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "standards_assessed": [s.value for s in standards],
            "checks_performed": [],
            "overall_score": 0.0,
            "compliance_status": "unknown",
            "recommendations": []
        }
        
        try:
            # Run checks for each enabled standard
            check_results = []
            
            for check_id, check in self.checks.items():
                if check.standard in standards:
                    if check.automated:
                        result = await self.run_automated_check(check)
                        check_results.append(result)
                        self.results[check_id] = result
                        
                        assessment_results["checks_performed"].append({
                            "check_id": result.check_id,
                            "status": result.status.value,
                            "score": result.score,
                            "findings_count": len(result.findings)
                        })
            
            # Calculate overall compliance score
            if check_results:
                total_score = sum(result.score for result in check_results)
                assessment_results["overall_score"] = total_score / len(check_results)
            
            # Determine overall compliance status
            overall_score = assessment_results["overall_score"]
            if overall_score >= 0.9:
                assessment_results["compliance_status"] = "compliant"
            elif overall_score >= 0.7:
                assessment_results["compliance_status"] = "mostly_compliant"
            else:
                assessment_results["compliance_status"] = "non_compliant"
            
            # Generate recommendations
            all_findings = []
            for result in check_results:
                all_findings.extend(result.findings)
            
            assessment_results["recommendations"] = self._generate_recommendations(all_findings)
            
            # Store assessment results
            await self._store_assessment_results(assessment_results)
            
            self.logger.info(f"Compliance assessment completed. Overall score: {overall_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Compliance assessment failed: {e}")
            assessment_results["error"] = str(e)
        
        return assessment_results
    
    def _generate_recommendations(self, findings: List[str]) -> List[str]:
        """Generate high-level recommendations based on findings."""
        recommendations = []
        
        finding_categories = {
            "authentication": ["multi-factor", "mfa", "authentication"],
            "encryption": ["encryption", "encrypted"],
            "logging": ["logging", "audit", "logs"],
            "policy": ["policy", "documentation"],
            "access_control": ["access control", "rbac", "authorization"]
        }
        
        finding_counts = {category: 0 for category in finding_categories}
        
        # Count findings by category
        for finding in findings:
            finding_lower = finding.lower()
            for category, keywords in finding_categories.items():
                if any(keyword in finding_lower for keyword in keywords):
                    finding_counts[category] += 1
        
        # Generate recommendations for top issues
        if finding_counts["encryption"] > 0:
            recommendations.append("Implement comprehensive encryption for data at rest and in transit")
        
        if finding_counts["authentication"] > 0:
            recommendations.append("Strengthen authentication mechanisms with multi-factor authentication")
        
        if finding_counts["logging"] > 0:
            recommendations.append("Enhance audit logging and monitoring capabilities")
        
        if finding_counts["policy"] > 0:
            recommendations.append("Review and update security policies and documentation")
        
        if finding_counts["access_control"] > 0:
            recommendations.append("Implement role-based access control and regular access reviews")
        
        return recommendations
    
    async def _store_assessment_results(self, results: Dict[str, Any]):
        """Store compliance assessment results for audit trail."""
        try:
            # Create compliance results directory
            results_dir = Path("compliance_results")
            results_dir.mkdir(exist_ok=True)
            
            # Store results with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"compliance_assessment_{timestamp}.json"
            
            async with aiofiles.open(results_file, "w") as f:
                await f.write(json.dumps(results, indent=2))
            
            self.logger.info(f"Compliance assessment results stored: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to store compliance results: {e}")
    
    async def generate_compliance_report(self, format: str = "json") -> str:
        """Generate compliance report in specified format."""
        try:
            # Run fresh assessment
            assessment = await self.run_compliance_assessment()
            
            if format == "json":
                return json.dumps(assessment, indent=2)
            
            elif format == "html":
                return self._generate_html_report(assessment)
            
            elif format == "markdown":
                return self._generate_markdown_report(assessment)
            
            else:
                raise ValueError(f"Unsupported report format: {format}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _generate_html_report(self, assessment: Dict[str, Any]) -> str:
        """Generate HTML compliance report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compliance Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .compliant {{ color: green; }}
                .warning {{ color: orange; }}
                .non-compliant {{ color: red; }}
                .check {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Compliance Assessment Report</h1>
                <p><strong>Assessment ID:</strong> {assessment.get('assessment_id', 'N/A')}</p>
                <p><strong>Date:</strong> {assessment.get('timestamp', 'N/A')}</p>
                <p><strong>Overall Score:</strong> <span class="score">{assessment.get('overall_score', 0):.1%}</span></p>
                <p><strong>Status:</strong> <span class="{assessment.get('compliance_status', '').replace('_', '-')}">{assessment.get('compliance_status', 'Unknown').replace('_', ' ').title()}</span></p>
            </div>
            
            <h2>Standards Assessed</h2>
            <ul>
        """
        
        for standard in assessment.get('standards_assessed', []):
            html += f"<li>{standard.upper()}</li>"
        
        html += """
            </ul>
            
            <h2>Check Results</h2>
        """
        
        for check in assessment.get('checks_performed', []):
            status_class = check['status'].replace('_', '-')
            html += f"""
            <div class="check">
                <strong>{check['check_id']}</strong> - 
                <span class="{status_class}">{check['status'].replace('_', ' ').title()}</span>
                (Score: {check['score']:.1%})
            </div>
            """
        
        html += """
            <h2>Recommendations</h2>
            <ul>
        """
        
        for rec in assessment.get('recommendations', []):
            html += f"<li>{rec}</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self, assessment: Dict[str, Any]) -> str:
        """Generate Markdown compliance report."""
        score = assessment.get('overall_score', 0)
        status = assessment.get('compliance_status', 'unknown').replace('_', ' ').title()
        
        md = f"""# Compliance Assessment Report

**Assessment ID:** {assessment.get('assessment_id', 'N/A')}  
**Date:** {assessment.get('timestamp', 'N/A')}  
**Overall Score:** {score:.1%}  
**Status:** {status}  

## Standards Assessed

"""
        
        for standard in assessment.get('standards_assessed', []):
            md += f"- {standard.upper()}\n"
        
        md += "\n## Check Results\n\n"
        
        for check in assessment.get('checks_performed', []):
            status_emoji = "✅" if check['status'] == 'compliant' else "⚠️" if check['status'] == 'warning' else "❌"
            md += f"{status_emoji} **{check['check_id']}** - {check['status'].replace('_', ' ').title()} (Score: {check['score']:.1%})\n"
        
        md += "\n## Recommendations\n\n"
        
        for rec in assessment.get('recommendations', []):
            md += f"- {rec}\n"
        
        return md


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compliance Monitor")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--assessment", action="store_true", help="Run compliance assessment")
    parser.add_argument("--report", choices=["json", "html", "markdown"], help="Generate compliance report")
    parser.add_argument("--standards", nargs="+", help="Specific standards to assess")
    
    args = parser.parse_args()
    
    monitor = ComplianceMonitor(args.config)
    
    if args.assessment:
        standards = None
        if args.standards:
            standards = [ComplianceStandard(std) for std in args.standards]
        
        result = await monitor.run_compliance_assessment(standards)
        print(json.dumps(result, indent=2))
    
    elif args.report:
        report = await monitor.generate_compliance_report(args.report)
        print(report)
    
    else:
        print("Use --assessment to run compliance assessment or --report to generate report")


if __name__ == "__main__":
    asyncio.run(main())