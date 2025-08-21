"""
Tests for Global Compliance Engine
Comprehensive testing for global-first compliance and localization features
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from pipeline.core.global_compliance_engine import (
    GlobalComplianceEngine,
    ComplianceRegion,
    SupportedLanguage,
    DataClassification,
    ComplianceFramework,
    ComplianceRule,
    LocalizationConfig,
    ComplianceAssessment,
    get_global_compliance_engine
)


class TestGlobalComplianceEngine:
    """Test suite for Global Compliance Engine"""

    @pytest.fixture
    def compliance_engine(self):
        """Create test instance of compliance engine"""
        return GlobalComplianceEngine()

    @pytest.fixture
    def sample_data_types(self):
        """Sample data types for testing"""
        return [
            DataClassification.PERSONAL,
            DataClassification.SENSITIVE_PERSONAL
        ]

    @pytest.fixture
    def sample_regions(self):
        """Sample regions for testing"""
        return [
            ComplianceRegion.EU,
            ComplianceRegion.US,
            ComplianceRegion.SINGAPORE
        ]

    def test_compliance_engine_initialization(self, compliance_engine):
        """Test proper initialization of compliance engine"""
        assert compliance_engine is not None
        assert len(compliance_engine.compliance_rules) > 0
        assert len(compliance_engine.localization_configs) > 0
        assert len(compliance_engine.supported_languages) >= 6
        assert ComplianceFramework.GDPR in compliance_engine.compliance_rules
        assert ComplianceFramework.CCPA in compliance_engine.compliance_rules
        assert ComplianceFramework.HIPAA in compliance_engine.compliance_rules
        assert SupportedLanguage.ENGLISH in compliance_engine.supported_languages

    def test_compliance_rules_initialization(self, compliance_engine):
        """Test compliance rules are properly initialized"""
        
        # Check GDPR rules
        gdpr_rules = compliance_engine.compliance_rules[ComplianceFramework.GDPR]
        assert len(gdpr_rules) >= 3
        
        # Verify rule structure
        for rule in gdpr_rules:
            assert isinstance(rule, ComplianceRule)
            assert rule.framework == ComplianceFramework.GDPR
            assert rule.region == ComplianceRegion.EU
            assert rule.rule_id.startswith("GDPR-")
            assert len(rule.title) > 0
            assert len(rule.description) > 0
            assert isinstance(rule.requires_consent, bool)
            assert isinstance(rule.requires_encryption, bool)
        
        # Check CCPA rules
        ccpa_rules = compliance_engine.compliance_rules[ComplianceFramework.CCPA]
        assert len(ccpa_rules) >= 2
        
        # Check HIPAA rules
        hipaa_rules = compliance_engine.compliance_rules[ComplianceFramework.HIPAA]
        assert len(hipaa_rules) >= 1
        
        # Verify HIPAA specific requirements
        hipaa_rule = hipaa_rules[0]
        assert hipaa_rule.requires_encryption is True
        assert hipaa_rule.audit_required is True
        assert hipaa_rule.retention_days == 2555  # 7 years

    def test_localization_initialization(self, compliance_engine):
        """Test localization configurations are properly initialized"""
        
        expected_languages = [
            SupportedLanguage.ENGLISH,
            SupportedLanguage.SPANISH,
            SupportedLanguage.FRENCH,
            SupportedLanguage.GERMAN,
            SupportedLanguage.JAPANESE,
            SupportedLanguage.CHINESE
        ]
        
        for language in expected_languages:
            assert language in compliance_engine.localization_configs
            
            config = compliance_engine.localization_configs[language]
            assert isinstance(config, LocalizationConfig)
            assert config.language == language
            assert len(config.currency) == 3  # Currency code should be 3 characters
            assert len(config.timezone) > 0
            assert config.text_direction in ["ltr", "rtl"]
        
        # Check specific configurations
        en_config = compliance_engine.localization_configs[SupportedLanguage.ENGLISH]
        assert en_config.currency == "USD"
        assert en_config.region == ComplianceRegion.US
        
        fr_config = compliance_engine.localization_configs[SupportedLanguage.FRENCH]
        assert fr_config.currency == "EUR"
        assert fr_config.region == ComplianceRegion.EU

    @pytest.mark.asyncio
    async def test_rule_compliance_check(self, compliance_engine):
        """Test individual rule compliance checking"""
        
        # Create test rule
        test_rule = ComplianceRule(
            framework=ComplianceFramework.GDPR,
            region=ComplianceRegion.EU,
            rule_id="TEST-001",
            title="Test Rule",
            description="Test compliance rule",
            data_types=[DataClassification.PERSONAL],
            retention_days=365,
            requires_consent=True,
            requires_encryption=True,
            cross_border_restrictions=True,
            audit_required=True
        )
        
        data_types = [DataClassification.PERSONAL]
        
        score = await compliance_engine._check_rule_compliance(test_rule, data_types)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Since we simulate all requirements as implemented, should be 1.0
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_framework_compliance_assessment(self, compliance_engine):
        """Test framework-specific compliance assessment"""
        
        data_types = [DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL]
        
        # Test GDPR compliance
        gdpr_assessment = await compliance_engine._assess_framework_compliance(
            ComplianceFramework.GDPR,
            ComplianceRegion.EU,
            data_types
        )
        
        assert isinstance(gdpr_assessment, ComplianceAssessment)
        assert gdpr_assessment.framework == ComplianceFramework.GDPR
        assert gdpr_assessment.region == ComplianceRegion.EU
        assert 0.0 <= gdpr_assessment.score <= 1.0
        assert isinstance(gdpr_assessment.compliant, bool)
        assert isinstance(gdpr_assessment.violations, list)
        assert isinstance(gdpr_assessment.recommendations, list)

    @pytest.mark.asyncio
    async def test_comprehensive_compliance_assessment(self, compliance_engine, sample_data_types, sample_regions):
        """Test comprehensive compliance assessment across multiple frameworks and regions"""
        
        frameworks = [ComplianceFramework.GDPR, ComplianceFramework.CCPA, ComplianceFramework.HIPAA]
        
        assessments = await compliance_engine.assess_compliance(
            data_types=sample_data_types,
            regions=sample_regions,
            frameworks=frameworks
        )
        
        # Should have assessments for each framework-region combination
        expected_count = len(frameworks) * len(sample_regions)
        assert len(assessments) <= expected_count  # Some combinations may not have rules
        
        # Check assessment quality
        for assessment in assessments:
            assert isinstance(assessment, ComplianceAssessment)
            assert assessment.framework in frameworks
            assert assessment.region in sample_regions
            assert 0.0 <= assessment.score <= 1.0
            assert isinstance(assessment.assessed_at, datetime)
        
        # Check that assessments are stored
        assert len(compliance_engine.compliance_assessments) >= len(assessments)

    @pytest.mark.asyncio
    async def test_content_localization(self, compliance_engine):
        """Test content localization for different languages"""
        
        test_content = {
            "welcome_message": "Welcome to our platform",
            "privacy_policy": "Your privacy is important to us",
            "terms_of_service": "Please read our terms carefully"
        }
        
        # Test localization for each supported language
        for language in compliance_engine.supported_languages:
            localized = await compliance_engine.localize_content(test_content, language)
            
            assert isinstance(localized, dict)
            assert len(localized) == len(test_content) + 1  # +1 for _localization metadata
            assert "_localization" in localized
            
            # Check localization metadata
            metadata = localized["_localization"]
            assert metadata["language"] == language.value
            assert "currency" in metadata
            assert "timezone" in metadata
            
            # Check content localization
            for key in test_content.keys():
                assert key in localized
                if language == SupportedLanguage.ENGLISH:
                    assert localized[key] == test_content[key]
                else:
                    # Should have language prefix for non-English
                    assert localized[key].startswith(f"[{language.value}]")

    @pytest.mark.asyncio
    async def test_unsupported_language_fallback(self, compliance_engine):
        """Test fallback to English for unsupported languages"""
        
        test_content = {"message": "Hello World"}
        
        # Use an invalid language enum (simulate by patching)
        with patch.object(compliance_engine, 'supported_languages', {SupportedLanguage.ENGLISH}):
            localized = await compliance_engine.localize_content(
                test_content, SupportedLanguage.SPANISH
            )
            
            # Should fallback to English
            assert localized["message"] == "Hello World"
            assert localized["_localization"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_data_residency_compliance(self, compliance_engine):
        """Test data residency compliance checking"""
        
        data_types = [DataClassification.PERSONAL]
        
        # Test EU to US transfer (should be restricted under GDPR)
        residency_results = await compliance_engine.ensure_data_residency(
            data_types=data_types,
            source_region=ComplianceRegion.EU,
            target_region=ComplianceRegion.US
        )
        
        assert isinstance(residency_results, dict)
        
        # Test US to EU transfer (generally more permissive)
        residency_results_us = await compliance_engine.ensure_data_residency(
            data_types=data_types,
            source_region=ComplianceRegion.US,
            target_region=ComplianceRegion.EU
        )
        
        assert isinstance(residency_results_us, dict)

    @pytest.mark.asyncio
    async def test_transfer_mechanism_evaluation(self, compliance_engine):
        """Test transfer mechanism evaluation for different region pairs"""
        
        test_rule = ComplianceRule(
            framework=ComplianceFramework.GDPR,
            region=ComplianceRegion.EU,
            rule_id="GDPR-TRANSFER",
            title="Cross-border Transfer",
            description="Test cross-border transfer rule",
            data_types=[DataClassification.PERSONAL],
            retention_days=None,
            requires_consent=True,
            requires_encryption=True,
            cross_border_restrictions=True,
            audit_required=True
        )
        
        # Test EU to adequate regions
        adequate_transfer = await compliance_engine._evaluate_transfer_mechanism(
            test_rule, ComplianceRegion.EU, ComplianceRegion.CANADA
        )
        assert adequate_transfer is True  # Canada is adequate
        
        # Test EU to non-adequate regions
        inadequate_transfer = await compliance_engine._evaluate_transfer_mechanism(
            test_rule, ComplianceRegion.EU, ComplianceRegion.BRAZIL
        )
        assert inadequate_transfer is False  # Brazil not in adequate list
        
        # Test US outbound transfers (more permissive)
        us_transfer = await compliance_engine._evaluate_transfer_mechanism(
            test_rule, ComplianceRegion.US, ComplianceRegion.EU
        )
        assert us_transfer is True  # US generally allows outbound

    def test_supported_regions_and_languages(self, compliance_engine):
        """Test supported regions and languages retrieval"""
        
        regions = compliance_engine.get_supported_regions()
        languages = compliance_engine.get_supported_languages()
        
        assert isinstance(regions, list)
        assert isinstance(languages, list)
        assert len(regions) >= 8  # Should support at least 8 regions
        assert len(languages) >= 6  # Should support at least 6 languages
        
        # Check that all major regions are supported
        expected_regions = [
            ComplianceRegion.EU,
            ComplianceRegion.US,
            ComplianceRegion.SINGAPORE,
            ComplianceRegion.CANADA,
            ComplianceRegion.JAPAN
        ]
        
        for region in expected_regions:
            assert region in regions

    def test_compliance_summary_no_assessments(self, compliance_engine):
        """Test compliance summary with no assessments"""
        
        # Ensure no assessments exist
        compliance_engine.compliance_assessments = []
        
        summary = compliance_engine.get_compliance_summary()
        
        assert summary["status"] == "no_assessments_completed"
        assert summary["supported_regions"] >= 8
        assert summary["supported_languages"] >= 6
        assert summary["compliance_frameworks"] >= 4

    @pytest.mark.asyncio
    async def test_compliance_summary_with_assessments(self, compliance_engine):
        """Test compliance summary with existing assessments"""
        
        # Create some test assessments
        test_assessments = [
            ComplianceAssessment(
                framework=ComplianceFramework.GDPR,
                region=ComplianceRegion.EU,
                compliant=True,
                score=0.95,
                violations=[],
                recommendations=[]
            ),
            ComplianceAssessment(
                framework=ComplianceFramework.CCPA,
                region=ComplianceRegion.US,
                compliant=True,
                score=0.92,
                violations=[],
                recommendations=[]
            ),
            ComplianceAssessment(
                framework=ComplianceFramework.HIPAA,
                region=ComplianceRegion.US,
                compliant=False,
                score=0.8,
                violations=["Missing audit trail"],
                recommendations=["Implement comprehensive auditing"]
            )
        ]
        
        compliance_engine.compliance_assessments = test_assessments
        
        summary = compliance_engine.get_compliance_summary()
        
        assert summary["compliance_status"] in ["compliant", "partial_compliance"]
        assert summary["total_assessments"] == 3
        assert summary["compliant_assessments"] == 2
        assert summary["compliance_rate"] == 2/3
        assert 0.0 <= summary["overall_score"] <= 1.0
        assert len(summary["frameworks_assessed"]) <= 3
        assert len(summary["regions_assessed"]) <= 2
        assert "last_assessment" in summary
        assert isinstance(summary["global_ready"], bool)

    def test_enum_completeness(self):
        """Test that all required enums are properly defined"""
        
        # Test ComplianceRegion enum
        regions = list(ComplianceRegion)
        assert len(regions) >= 8
        assert ComplianceRegion.EU in regions
        assert ComplianceRegion.US in regions
        assert ComplianceRegion.SINGAPORE in regions
        
        # Test SupportedLanguage enum
        languages = list(SupportedLanguage)
        assert len(languages) >= 6
        assert SupportedLanguage.ENGLISH in languages
        assert SupportedLanguage.SPANISH in languages
        assert SupportedLanguage.FRENCH in languages
        
        # Test DataClassification enum
        classifications = list(DataClassification)
        assert DataClassification.PERSONAL in classifications
        assert DataClassification.SENSITIVE_PERSONAL in classifications
        assert DataClassification.PUBLIC in classifications
        
        # Test ComplianceFramework enum
        frameworks = list(ComplianceFramework)
        assert ComplianceFramework.GDPR in frameworks
        assert ComplianceFramework.CCPA in frameworks
        assert ComplianceFramework.HIPAA in frameworks

    def test_singleton_pattern(self):
        """Test singleton pattern for global instance"""
        
        instance1 = get_global_compliance_engine()
        instance2 = get_global_compliance_engine()
        
        assert instance1 is instance2
        assert isinstance(instance1, GlobalComplianceEngine)


class TestComplianceRule:
    """Test ComplianceRule data structure"""

    def test_compliance_rule_creation(self):
        """Test ComplianceRule instantiation"""
        
        rule = ComplianceRule(
            framework=ComplianceFramework.GDPR,
            region=ComplianceRegion.EU,
            rule_id="TEST-001",
            title="Test Rule",
            description="Test compliance rule",
            data_types=[DataClassification.PERSONAL],
            retention_days=365,
            requires_consent=True,
            requires_encryption=True,
            cross_border_restrictions=True,
            audit_required=True,
            breach_notification_hours=72
        )
        
        assert rule.framework == ComplianceFramework.GDPR
        assert rule.region == ComplianceRegion.EU
        assert rule.rule_id == "TEST-001"
        assert rule.title == "Test Rule"
        assert rule.description == "Test compliance rule"
        assert DataClassification.PERSONAL in rule.data_types
        assert rule.retention_days == 365
        assert rule.requires_consent is True
        assert rule.requires_encryption is True
        assert rule.cross_border_restrictions is True
        assert rule.audit_required is True
        assert rule.breach_notification_hours == 72


class TestLocalizationConfig:
    """Test LocalizationConfig data structure"""

    def test_localization_config_creation(self):
        """Test LocalizationConfig instantiation"""
        
        config = LocalizationConfig(
            language=SupportedLanguage.GERMAN,
            region=ComplianceRegion.EU,
            currency="EUR",
            date_format="DD.MM.YYYY",
            time_format="24h",
            number_format="1.234,56",
            timezone="Europe/Berlin",
            text_direction="ltr"
        )
        
        assert config.language == SupportedLanguage.GERMAN
        assert config.region == ComplianceRegion.EU
        assert config.currency == "EUR"
        assert config.date_format == "DD.MM.YYYY"
        assert config.time_format == "24h"
        assert config.number_format == "1.234,56"
        assert config.timezone == "Europe/Berlin"
        assert config.text_direction == "ltr"


class TestComplianceAssessment:
    """Test ComplianceAssessment data structure"""

    def test_compliance_assessment_creation(self):
        """Test ComplianceAssessment instantiation"""
        
        assessment = ComplianceAssessment(
            framework=ComplianceFramework.CCPA,
            region=ComplianceRegion.US,
            compliant=True,
            score=0.95,
            violations=[],
            recommendations=["Maintain current standards"]
        )
        
        assert assessment.framework == ComplianceFramework.CCPA
        assert assessment.region == ComplianceRegion.US
        assert assessment.compliant is True
        assert assessment.score == 0.95
        assert assessment.violations == []
        assert assessment.recommendations == ["Maintain current standards"]
        assert isinstance(assessment.assessed_at, datetime)