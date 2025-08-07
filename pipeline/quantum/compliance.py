"""
Quantum Task Planner Compliance Framework

Comprehensive compliance management for global privacy and data protection regulations:
- GDPR (General Data Protection Regulation) - EU
- CCPA (California Consumer Privacy Act) - US/California  
- PDPA (Personal Data Protection Act) - Singapore/Thailand
- LGPD (Lei Geral de Proteção de Dados) - Brazil
- Data sovereignty and residency requirements
- Audit logging and compliance reporting
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ComplianceRegulation(str, Enum):
    """Supported privacy and data protection regulations."""

    GDPR = "GDPR"           # General Data Protection Regulation (EU)
    CCPA = "CCPA"           # California Consumer Privacy Act (US/CA)
    PDPA_SG = "PDPA_SG"     # Personal Data Protection Act (Singapore)
    PDPA_TH = "PDPA_TH"     # Personal Data Protection Act (Thailand)
    LGPD = "LGPD"           # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "PIPEDA"       # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "PRIVACY_ACT"  # Privacy Act (Australia)


class DataCategory(str, Enum):
    """Categories of data for compliance classification."""

    PERSONAL_IDENTIFIABLE = "personal_identifiable"    # PII data
    SENSITIVE_PERSONAL = "sensitive_personal"          # Special category data
    TECHNICAL_METADATA = "technical_metadata"          # System/technical data
    BEHAVIORAL_DATA = "behavioral_data"                # Usage patterns
    QUANTUM_STATE_DATA = "quantum_state_data"          # Quantum task states
    PERFORMANCE_METRICS = "performance_metrics"        # System performance data
    AUDIT_LOGS = "audit_logs"                         # Compliance audit logs


class ProcessingLawfulBasis(str, Enum):
    """Lawful bases for data processing under GDPR."""

    CONSENT = "consent"                    # Article 6(1)(a) - Consent
    CONTRACT = "contract"                  # Article 6(1)(b) - Contract performance
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c) - Legal obligation
    VITAL_INTERESTS = "vital_interests"    # Article 6(1)(d) - Vital interests
    PUBLIC_TASK = "public_task"           # Article 6(1)(e) - Public task
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Article 6(1)(f) - Legitimate interests


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    regulation: ComplianceRegulation = ComplianceRegulation.GDPR
    data_subject_id: str | None = None
    data_category: DataCategory = DataCategory.TECHNICAL_METADATA
    processing_purpose: str = ""
    lawful_basis: ProcessingLawfulBasis = ProcessingLawfulBasis.LEGITIMATE_INTERESTS
    data_controller: str = "Quantum Task Planner System"
    data_processor: str | None = None
    retention_period_days: int = 365
    geographic_location: str = "EU"
    anonymized: bool = False
    encrypted: bool = True
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "regulation": self.regulation.value,
            "data_subject_id": self.data_subject_id,
            "data_category": self.data_category.value,
            "processing_purpose": self.processing_purpose,
            "lawful_basis": self.lawful_basis.value,
            "data_controller": self.data_controller,
            "data_processor": self.data_processor,
            "retention_period_days": self.retention_period_days,
            "geographic_location": self.geographic_location,
            "anonymized": self.anonymized,
            "encrypted": self.encrypted,
            "additional_metadata": self.additional_metadata
        }


@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_subject_id: str = ""
    consent_timestamp: datetime = field(default_factory=datetime.utcnow)
    consent_version: str = "1.0"
    processing_purposes: list[str] = field(default_factory=list)
    data_categories: list[DataCategory] = field(default_factory=list)
    consent_method: str = "explicit"  # explicit, implicit, opt-in, opt-out
    consent_withdrawn: bool = False
    withdrawal_timestamp: datetime | None = None
    expiry_date: datetime | None = None
    geographic_location: str = ""
    regulation: ComplianceRegulation = ComplianceRegulation.GDPR

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.consent_withdrawn:
            return False

        if self.expiry_date and datetime.utcnow() > self.expiry_date:
            return False

        return True

    def withdraw_consent(self):
        """Withdraw consent."""
        self.consent_withdrawn = True
        self.withdrawal_timestamp = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "data_subject_id": self.data_subject_id,
            "consent_timestamp": self.consent_timestamp.isoformat(),
            "consent_version": self.consent_version,
            "processing_purposes": self.processing_purposes,
            "data_categories": [dc.value for dc in self.data_categories],
            "consent_method": self.consent_method,
            "consent_withdrawn": self.consent_withdrawn,
            "withdrawal_timestamp": self.withdrawal_timestamp.isoformat() if self.withdrawal_timestamp else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "geographic_location": self.geographic_location,
            "regulation": self.regulation.value
        }


class ComplianceRule(ABC):
    """Abstract base class for compliance rules."""

    @abstractmethod
    def check_compliance(self, processing_record: DataProcessingRecord) -> bool:
        """Check if processing record complies with this rule."""
        pass

    @abstractmethod
    def get_violation_message(self) -> str:
        """Get violation message if compliance check fails."""
        pass


class GDPRDataMinimizationRule(ComplianceRule):
    """GDPR Article 5(1)(c) - Data minimization principle."""

    def check_compliance(self, processing_record: DataProcessingRecord) -> bool:
        """Check data minimization compliance."""
        # Check if processing purpose is clearly defined
        if not processing_record.processing_purpose.strip():
            return False

        # Check if lawful basis is specified
        if not processing_record.lawful_basis:
            return False

        # Sensitive data requires additional protections
        if processing_record.data_category == DataCategory.SENSITIVE_PERSONAL:
            return processing_record.encrypted and processing_record.lawful_basis in [
                ProcessingLawfulBasis.CONSENT,
                ProcessingLawfulBasis.LEGAL_OBLIGATION
            ]

        return True

    def get_violation_message(self) -> str:
        return "GDPR Article 5(1)(c) violation: Data processing does not meet minimization requirements"


class GDPRRetentionLimitRule(ComplianceRule):
    """GDPR Article 5(1)(e) - Storage limitation principle."""

    def __init__(self, max_retention_days: int = 2555):  # ~7 years default
        self.max_retention_days = max_retention_days

    def check_compliance(self, processing_record: DataProcessingRecord) -> bool:
        """Check retention period compliance."""
        return processing_record.retention_period_days <= self.max_retention_days

    def get_violation_message(self) -> str:
        return f"GDPR Article 5(1)(e) violation: Retention period exceeds {self.max_retention_days} days"


class CCPADataSaleRestictionRule(ComplianceRule):
    """CCPA Section 1798.120 - Right to opt-out of sale."""

    def check_compliance(self, processing_record: DataProcessingRecord) -> bool:
        """Check CCPA data sale restrictions."""
        # For quantum task planner, we don't sell data, so this is always compliant
        return True

    def get_violation_message(self) -> str:
        return "CCPA Section 1798.120 violation: Data sale without opt-out mechanism"


class QuantumComplianceManager:
    """Main compliance management system for quantum task planner."""

    def __init__(self, default_regulation: ComplianceRegulation = ComplianceRegulation.GDPR):
        self.default_regulation = default_regulation
        self.processing_records: list[DataProcessingRecord] = []
        self.consent_records: dict[str, ConsentRecord] = {}
        self.compliance_rules: dict[ComplianceRegulation, list[ComplianceRule]] = {}

        # Initialize compliance rules
        self._initialize_compliance_rules()

        # Audit log
        self.audit_log: list[dict[str, Any]] = []

    def _initialize_compliance_rules(self):
        """Initialize compliance rules for different regulations."""

        # GDPR rules
        self.compliance_rules[ComplianceRegulation.GDPR] = [
            GDPRDataMinimizationRule(),
            GDPRRetentionLimitRule()
        ]

        # CCPA rules
        self.compliance_rules[ComplianceRegulation.CCPA] = [
            CCPADataSaleRestictionRule()
        ]

        # Other regulations would be added here
        for regulation in ComplianceRegulation:
            if regulation not in self.compliance_rules:
                self.compliance_rules[regulation] = []

    def record_data_processing(self,
                             data_subject_id: str | None,
                             data_category: DataCategory,
                             processing_purpose: str,
                             lawful_basis: ProcessingLawfulBasis,
                             geographic_location: str = "EU",
                             regulation: ComplianceRegulation | None = None,
                             **kwargs) -> str:
        """
        Record a data processing activity.
        
        Args:
            data_subject_id: ID of the data subject (if applicable)
            data_category: Category of data being processed
            processing_purpose: Purpose of processing
            lawful_basis: Legal basis for processing
            geographic_location: Geographic location of processing
            regulation: Applicable regulation
            **kwargs: Additional metadata
            
        Returns:
            Processing record ID
        """
        regulation = regulation or self.default_regulation

        record = DataProcessingRecord(
            regulation=regulation,
            data_subject_id=data_subject_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            lawful_basis=lawful_basis,
            geographic_location=geographic_location,
            additional_metadata=kwargs
        )

        self.processing_records.append(record)

        # Log the processing activity
        self._add_audit_log("data_processing_recorded", {
            "record_id": record.id,
            "data_category": data_category.value,
            "purpose": processing_purpose
        })

        # Check compliance
        self._check_processing_compliance(record)

        logger.info(f"Recorded data processing: {record.id}")
        return record.id

    def record_consent(self,
                      data_subject_id: str,
                      processing_purposes: list[str],
                      data_categories: list[DataCategory],
                      consent_method: str = "explicit",
                      regulation: ComplianceRegulation | None = None,
                      expiry_days: int | None = None) -> str:
        """
        Record user consent for data processing.
        
        Args:
            data_subject_id: ID of the data subject
            processing_purposes: List of processing purposes
            data_categories: List of data categories
            consent_method: Method of obtaining consent
            regulation: Applicable regulation
            expiry_days: Consent expiry in days
            
        Returns:
            Consent record ID
        """
        regulation = regulation or self.default_regulation

        expiry_date = None
        if expiry_days:
            expiry_date = datetime.utcnow() + timedelta(days=expiry_days)

        consent = ConsentRecord(
            data_subject_id=data_subject_id,
            processing_purposes=processing_purposes,
            data_categories=data_categories,
            consent_method=consent_method,
            regulation=regulation,
            expiry_date=expiry_date
        )

        self.consent_records[consent.id] = consent

        self._add_audit_log("consent_recorded", {
            "consent_id": consent.id,
            "data_subject_id": data_subject_id,
            "purposes": processing_purposes
        })

        logger.info(f"Recorded consent: {consent.id} for subject {data_subject_id}")
        return consent.id

    def withdraw_consent(self, consent_id: str) -> bool:
        """
        Withdraw consent for data processing.
        
        Args:
            consent_id: ID of consent to withdraw
            
        Returns:
            True if consent was withdrawn successfully
        """
        if consent_id in self.consent_records:
            consent = self.consent_records[consent_id]
            consent.withdraw_consent()

            self._add_audit_log("consent_withdrawn", {
                "consent_id": consent_id,
                "data_subject_id": consent.data_subject_id
            })

            logger.info(f"Consent withdrawn: {consent_id}")
            return True

        return False

    def check_consent_validity(self, data_subject_id: str,
                             processing_purpose: str,
                             data_category: DataCategory) -> bool:
        """
        Check if valid consent exists for processing.
        
        Args:
            data_subject_id: ID of the data subject
            processing_purpose: Purpose of processing
            data_category: Category of data
            
        Returns:
            True if valid consent exists
        """
        for consent in self.consent_records.values():
            if (consent.data_subject_id == data_subject_id and
                consent.is_valid() and
                processing_purpose in consent.processing_purposes and
                data_category in consent.data_categories):
                return True

        return False

    def _check_processing_compliance(self, record: DataProcessingRecord):
        """Check processing record against compliance rules."""
        applicable_rules = self.compliance_rules.get(record.regulation, [])

        for rule in applicable_rules:
            if not rule.check_compliance(record):
                violation_message = rule.get_violation_message()

                self._add_audit_log("compliance_violation", {
                    "record_id": record.id,
                    "regulation": record.regulation.value,
                    "violation": violation_message
                })

                logger.warning(f"Compliance violation: {violation_message}")

    def generate_data_subject_report(self, data_subject_id: str) -> dict[str, Any]:
        """
        Generate data subject report (GDPR Article 15 - Right of access).
        
        Args:
            data_subject_id: ID of the data subject
            
        Returns:
            Report containing all data and processing activities
        """
        # Find all processing records for this data subject
        subject_records = [
            record for record in self.processing_records
            if record.data_subject_id == data_subject_id
        ]

        # Find all consent records
        subject_consents = [
            consent for consent in self.consent_records.values()
            if consent.data_subject_id == data_subject_id
        ]

        report = {
            "data_subject_id": data_subject_id,
            "report_generated": datetime.utcnow().isoformat(),
            "processing_records": [record.to_dict() for record in subject_records],
            "consent_records": [consent.to_dict() for consent in subject_consents],
            "data_categories_processed": list(set(record.data_category.value for record in subject_records)),
            "processing_purposes": list(set(record.processing_purpose for record in subject_records)),
            "retention_periods": {
                record.id: record.retention_period_days for record in subject_records
            }
        }

        self._add_audit_log("data_subject_report_generated", {
            "data_subject_id": data_subject_id,
            "records_count": len(subject_records)
        })

        return report

    def initiate_data_deletion(self, data_subject_id: str,
                             reason: str = "Right to be forgotten") -> dict[str, Any]:
        """
        Initiate data deletion process (GDPR Article 17 - Right to erasure).
        
        Args:
            data_subject_id: ID of the data subject
            reason: Reason for deletion
            
        Returns:
            Deletion report
        """
        # Find all data associated with subject
        subject_records = [
            record for record in self.processing_records
            if record.data_subject_id == data_subject_id
        ]

        deletion_report = {
            "data_subject_id": data_subject_id,
            "deletion_initiated": datetime.utcnow().isoformat(),
            "reason": reason,
            "records_to_delete": len(subject_records),
            "deletion_status": "initiated",
            "records_details": [
                {
                    "record_id": record.id,
                    "data_category": record.data_category.value,
                    "processing_purpose": record.processing_purpose
                }
                for record in subject_records
            ]
        }

        # In a real implementation, this would trigger actual data deletion
        # For now, we just mark records for deletion
        for record in subject_records:
            record.additional_metadata["marked_for_deletion"] = True
            record.additional_metadata["deletion_reason"] = reason
            record.additional_metadata["deletion_requested"] = datetime.utcnow().isoformat()

        self._add_audit_log("data_deletion_initiated", {
            "data_subject_id": data_subject_id,
            "reason": reason,
            "records_count": len(subject_records)
        })

        logger.info(f"Data deletion initiated for subject: {data_subject_id}")
        return deletion_report

    def generate_compliance_report(self, regulation: ComplianceRegulation | None = None,
                                 start_date: datetime | None = None,
                                 end_date: datetime | None = None) -> dict[str, Any]:
        """
        Generate compliance report for audit purposes.
        
        Args:
            regulation: Specific regulation to report on
            start_date: Start date for report period
            end_date: End date for report period
            
        Returns:
            Comprehensive compliance report
        """
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=30))

        # Filter records by date range
        period_records = [
            record for record in self.processing_records
            if start_date <= record.timestamp <= end_date
        ]

        # Filter by regulation if specified
        if regulation:
            period_records = [
                record for record in period_records
                if record.regulation == regulation
            ]

        # Generate statistics
        data_categories = {}
        processing_purposes = {}
        lawful_bases = {}
        geographic_locations = {}

        for record in period_records:
            data_categories[record.data_category.value] = data_categories.get(record.data_category.value, 0) + 1
            processing_purposes[record.processing_purpose] = processing_purposes.get(record.processing_purpose, 0) + 1
            lawful_bases[record.lawful_basis.value] = lawful_bases.get(record.lawful_basis.value, 0) + 1
            geographic_locations[record.geographic_location] = geographic_locations.get(record.geographic_location, 0) + 1

        # Count violations
        violations = [
            log for log in self.audit_log
            if (log.get("action") == "compliance_violation" and
                start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date)
        ]

        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "regulation": regulation.value if regulation else "all",
            "summary": {
                "total_processing_records": len(period_records),
                "total_consent_records": len([c for c in self.consent_records.values()
                                            if start_date <= c.consent_timestamp <= end_date]),
                "total_violations": len(violations),
                "data_categories_processed": len(data_categories),
                "unique_processing_purposes": len(processing_purposes)
            },
            "statistics": {
                "data_categories": data_categories,
                "processing_purposes": processing_purposes,
                "lawful_bases": lawful_bases,
                "geographic_locations": geographic_locations
            },
            "violations": violations,
            "compliance_score": max(0, 100 - (len(violations) * 10)),  # Simple scoring
            "generated_timestamp": datetime.utcnow().isoformat()
        }

        self._add_audit_log("compliance_report_generated", {
            "regulation": regulation.value if regulation else "all",
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "records_count": len(period_records)
        })

        return report

    def get_privacy_policy_text(self, regulation: ComplianceRegulation) -> dict[str, str]:
        """
        Get privacy policy text for specific regulation.
        
        Args:
            regulation: Target regulation
            
        Returns:
            Privacy policy components
        """
        policies = {
            ComplianceRegulation.GDPR: {
                "title": "Privacy Policy - GDPR Compliance",
                "data_controller": "Quantum Task Planner System acts as data controller for processing activities.",
                "lawful_basis": "We process data based on legitimate interests for system operation and improvement.",
                "data_categories": "We process technical metadata, performance metrics, and quantum state data.",
                "retention": "Data is retained for up to 7 years unless deletion is requested.",
                "rights": "You have rights to access, rectify, erase, restrict, and port your data under GDPR.",
                "contact": "Contact our Data Protection Officer for privacy-related inquiries."
            },
            ComplianceRegulation.CCPA: {
                "title": "Privacy Policy - CCPA Compliance",
                "data_controller": "Quantum Task Planner System collects and processes personal information.",
                "categories": "We collect technical data, usage patterns, and system performance metrics.",
                "purposes": "Data is used for system operation, optimization, and service improvement.",
                "sharing": "We do not sell personal information to third parties.",
                "rights": "California residents have rights to know, delete, and opt-out under CCPA.",
                "contact": "Contact us to exercise your privacy rights under CCPA."
            }
        }

        return policies.get(regulation, policies[ComplianceRegulation.GDPR])

    def _add_audit_log(self, action: str, details: dict[str, Any]):
        """Add entry to audit log."""
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "details": details
        }

        self.audit_log.append(log_entry)

        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]  # Keep last 5000 entries

    def export_compliance_data(self, output_file: Path,
                              regulation: ComplianceRegulation | None = None):
        """
        Export compliance data for external audit.
        
        Args:
            output_file: File to save compliance data
            regulation: Specific regulation to export
        """
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "regulation": regulation.value if regulation else "all",
            "processing_records": [
                record.to_dict() for record in self.processing_records
                if not regulation or record.regulation == regulation
            ],
            "consent_records": [
                consent.to_dict() for consent in self.consent_records.values()
                if not regulation or consent.regulation == regulation
            ],
            "audit_log": self.audit_log,
            "compliance_rules": {
                reg.value: [rule.__class__.__name__ for rule in rules]
                for reg, rules in self.compliance_rules.items()
            }
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Compliance data exported to {output_file}")

            self._add_audit_log("compliance_data_exported", {
                "output_file": str(output_file),
                "regulation": regulation.value if regulation else "all"
            })

        except OSError as e:
            logger.error(f"Failed to export compliance data: {e}")


# Global compliance manager instance
_compliance_manager: QuantumComplianceManager | None = None


def get_compliance_manager(regulation: ComplianceRegulation = ComplianceRegulation.GDPR) -> QuantumComplianceManager:
    """Get the global compliance manager instance."""
    global _compliance_manager

    if _compliance_manager is None:
        _compliance_manager = QuantumComplianceManager(regulation)

    return _compliance_manager
