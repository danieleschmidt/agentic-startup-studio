#!/usr/bin/env python3
"""
Global-First Features Validation Tests
Tests internationalization, multi-region support, and compliance features.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestGlobalFirstFeatures(unittest.TestCase):
    """Global-first implementation validation tests"""
    
    def test_internationalization_support(self):
        """Test i18n support for multiple languages"""
        from pipeline.quantum.i18n import (
            InternationalizationManager, 
            LocalizationConfig, 
            SupportedLanguage
        )
        
        # Test English (default)
        en_config = LocalizationConfig(language=SupportedLanguage.ENGLISH)
        en_manager = InternationalizationManager(en_config)
        
        self.assertEqual(en_manager.get_text("startup_idea"), "Startup Idea")
        self.assertEqual(en_manager.get_text("processing"), "Processing...")
        
        # Test Spanish
        es_config = LocalizationConfig(language=SupportedLanguage.SPANISH)
        es_manager = InternationalizationManager(es_config)
        
        self.assertEqual(es_manager.get_text("startup_idea"), "Idea de Startup")
        self.assertEqual(es_manager.get_text("processing"), "Procesando...")
        
        # Test French
        fr_config = LocalizationConfig(language=SupportedLanguage.FRENCH)
        fr_manager = InternationalizationManager(fr_config)
        
        self.assertEqual(fr_manager.get_text("startup_idea"), "Idée de Startup")
        self.assertEqual(fr_manager.get_text("processing"), "Traitement en cours...")
        
        # Test German
        de_config = LocalizationConfig(language=SupportedLanguage.GERMAN)
        de_manager = InternationalizationManager(de_config)
        
        self.assertEqual(de_manager.get_text("startup_idea"), "Startup-Idee")
        self.assertEqual(de_manager.get_text("processing"), "Verarbeitung...")
        
        # Test Japanese
        ja_config = LocalizationConfig(language=SupportedLanguage.JAPANESE)
        ja_manager = InternationalizationManager(ja_config)
        
        self.assertEqual(ja_manager.get_text("startup_idea"), "スタートアップのアイデア")
        self.assertEqual(ja_manager.get_text("processing"), "処理中...")
        
        # Test Chinese Simplified
        zh_config = LocalizationConfig(language=SupportedLanguage.CHINESE_SIMPLIFIED)
        zh_manager = InternationalizationManager(zh_config)
        
        self.assertEqual(zh_manager.get_text("startup_idea"), "创业想法")
        self.assertEqual(zh_manager.get_text("processing"), "处理中...")
    
    def test_parameterized_translations(self):
        """Test translations with parameters"""
        from pipeline.quantum.i18n import InternationalizationManager, LocalizationConfig, SupportedLanguage
        
        # Test English parameterized translation
        en_manager = InternationalizationManager(LocalizationConfig(language=SupportedLanguage.ENGLISH))
        result = en_manager.get_text("budget_warning", threshold=80)
        self.assertEqual(result, "Budget warning: 80% reached")
        
        result = en_manager.get_text("similar_ideas_found", count=5)
        self.assertEqual(result, "Found 5 similar ideas")
        
        # Test Spanish parameterized translation
        es_manager = InternationalizationManager(LocalizationConfig(language=SupportedLanguage.SPANISH))
        result = es_manager.get_text("budget_warning", threshold=75)
        self.assertEqual(result, "Advertencia de presupuesto: 75% alcanzado")
    
    def test_regional_currency_formatting(self):
        """Test currency formatting for different regions"""
        from pipeline.quantum.i18n import (
            InternationalizationManager, 
            LocalizationConfig, 
            Region
        )
        
        # Test North America (USD)
        na_config = LocalizationConfig(region=Region.NORTH_AMERICA)
        na_manager = InternationalizationManager(na_config)
        
        self.assertEqual(na_manager.format_currency(1234.56), "$1,234.56")
        
        # Test Europe (EUR)
        eu_config = LocalizationConfig(region=Region.EUROPE)
        eu_manager = InternationalizationManager(eu_config)
        
        self.assertEqual(eu_manager.format_currency(1234.56), "€1 234.56")
        
        # Test Asia Pacific (JPY)
        apac_config = LocalizationConfig(region=Region.ASIA_PACIFIC)
        apac_manager = InternationalizationManager(apac_config)
        
        self.assertEqual(apac_manager.format_currency(1234.56), "¥1,235")
    
    def test_regional_date_formatting(self):
        """Test date formatting for different regions"""
        from pipeline.quantum.i18n import (
            InternationalizationManager, 
            LocalizationConfig, 
            Region
        )
        
        test_date = datetime(2025, 8, 13)
        
        # Test North America (MM/DD/YYYY)
        na_config = LocalizationConfig(region=Region.NORTH_AMERICA)
        na_manager = InternationalizationManager(na_config)
        
        self.assertEqual(na_manager.format_date(test_date), "08/13/2025")
        
        # Test Europe (DD.MM.YYYY)
        eu_config = LocalizationConfig(region=Region.EUROPE)
        eu_manager = InternationalizationManager(eu_config)
        
        self.assertEqual(eu_manager.format_date(test_date), "13.08.2025")
        
        # Test Latin America (DD/MM/YYYY)
        latam_config = LocalizationConfig(region=Region.LATIN_AMERICA)
        latam_manager = InternationalizationManager(latam_config)
        
        self.assertEqual(latam_manager.format_date(test_date), "13/08/2025")
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance features"""
        from pipeline.quantum.i18n import ComplianceManager, Region
        
        # Test EU compliance manager
        eu_compliance = ComplianceManager(Region.EUROPE)
        
        # Test GDPR-specific requirements
        self.assertIn("GDPR", eu_compliance.compliance_rules["privacy_regulations"])
        self.assertTrue(eu_compliance.compliance_rules["requires_consent"])
        self.assertFalse(eu_compliance.compliance_rules["allows_profiling"])  # Requires explicit consent
        self.assertTrue(eu_compliance.compliance_rules["data_transfer_restrictions"])
        
        # Test data retention (GDPR default 3 years)
        self.assertEqual(eu_compliance.get_retention_period(), 1095)
        
        # Test required disclosures
        disclosures = eu_compliance.get_required_disclosures()
        self.assertIn("data_collection", disclosures)
        self.assertIn("legal_basis", disclosures)
        self.assertIn("data_subject_rights", disclosures)
        
        # Test profiling consent requirement
        self.assertTrue(eu_compliance.requires_explicit_consent_for_profiling())
    
    def test_ccpa_compliance(self):
        """Test CCPA compliance features"""
        from pipeline.quantum.i18n import ComplianceManager, Region
        
        # Test North America compliance manager
        na_compliance = ComplianceManager(Region.NORTH_AMERICA)
        
        # Test CCPA-specific requirements
        self.assertIn("CCPA", na_compliance.compliance_rules["privacy_regulations"])
        self.assertTrue(na_compliance.compliance_rules["requires_consent"])
        self.assertTrue(na_compliance.compliance_rules["allows_profiling"])
        self.assertFalse(na_compliance.compliance_rules["data_transfer_restrictions"])
        
        # Test data retention (7 years)
        self.assertEqual(na_compliance.get_retention_period(), 2555)
        
        # Test required disclosures
        disclosures = na_compliance.get_required_disclosures()
        self.assertIn("data_collection", disclosures)
        self.assertIn("third_party_sharing", disclosures)
    
    def test_pdpa_compliance(self):
        """Test PDPA compliance features"""
        from pipeline.quantum.i18n import ComplianceManager, Region
        
        # Test Asia Pacific compliance manager
        apac_compliance = ComplianceManager(Region.ASIA_PACIFIC)
        
        # Test PDPA-specific requirements
        self.assertIn("PDPA", apac_compliance.compliance_rules["privacy_regulations"])
        self.assertTrue(apac_compliance.compliance_rules["requires_consent"])
        self.assertTrue(apac_compliance.compliance_rules["data_transfer_restrictions"])
        
        # Test data retention (5 years)
        self.assertEqual(apac_compliance.get_retention_period(), 1825)
        
        # Test required disclosures
        disclosures = apac_compliance.get_required_disclosures()
        self.assertIn("data_collection", disclosures)
        self.assertIn("cross_border_transfer", disclosures)
    
    def test_cross_border_data_transfer(self):
        """Test cross-border data transfer validation"""
        from pipeline.quantum.i18n import ComplianceManager, Region
        
        # Test EU to US transfer (should be allowed with adequate protection)
        eu_compliance = ComplianceManager(Region.EUROPE)
        self.assertTrue(eu_compliance.validate_cross_border_transfer(Region.NORTH_AMERICA))
        
        # Test unrestricted region transfers
        na_compliance = ComplianceManager(Region.NORTH_AMERICA)
        self.assertTrue(na_compliance.validate_cross_border_transfer(Region.EUROPE))
        self.assertTrue(na_compliance.validate_cross_border_transfer(Region.ASIA_PACIFIC))
    
    def test_global_convenience_functions(self):
        """Test global convenience functions"""
        from pipeline.quantum.i18n import translate, format_currency, Region
        
        # Test translate function
        result = translate("startup_idea")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        
        # Test format_currency function
        formatted = format_currency(1000.50, Region.NORTH_AMERICA)
        self.assertEqual(formatted, "$1,000.50")
        
        formatted = format_currency(1000.50, Region.EUROPE)
        self.assertEqual(formatted, "€1 000.50")
    
    def test_language_switching(self):
        """Test dynamic language switching"""
        from pipeline.quantum.i18n import (
            InternationalizationManager, 
            LocalizationConfig, 
            SupportedLanguage
        )
        
        manager = InternationalizationManager()
        
        # Start with English
        self.assertEqual(manager.get_text("startup_idea"), "Startup Idea")
        
        # Switch to Spanish
        manager.set_language(SupportedLanguage.SPANISH)
        self.assertEqual(manager.get_text("startup_idea"), "Idea de Startup")
        
        # Switch to French
        manager.set_language(SupportedLanguage.FRENCH) 
        self.assertEqual(manager.get_text("startup_idea"), "Idée de Startup")
    
    def test_supported_languages_list(self):
        """Test getting list of supported languages"""
        from pipeline.quantum.i18n import InternationalizationManager
        
        manager = InternationalizationManager()
        supported = manager.get_supported_languages()
        
        expected_languages = ["en", "es", "fr", "de", "ja", "zh"]
        self.assertEqual(len(supported), 6)
        
        for lang in expected_languages:
            self.assertIn(lang, supported)

if __name__ == '__main__':
    print("🌍 Running Global-First Features Validation Tests")
    print("=" * 55)
    
    # Run tests
    unittest.main(verbosity=2)
