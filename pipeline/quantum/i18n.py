"""
Internationalization (i18n) Support for Global-First Implementation
Supports multi-language content and regional compliance.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class SupportedLanguage(str, Enum):
    """Supported languages for global deployment"""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"

class Region(str, Enum):
    """Supported regions for compliance and localization"""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    LATIN_AMERICA = "latam"

@dataclass
class LocalizationConfig:
    """Configuration for localization settings"""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: Region = Region.NORTH_AMERICA
    timezone: str = "UTC"
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    number_format: str = "en_US"

class InternationalizationManager:
    """Manages internationalization and localization"""
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        """Initialize i18n manager with configuration"""
        self.config = config or LocalizationConfig()
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
        
        logger.info(f"I18n initialized for {self.config.language.value} in {self.config.region.value}")
    
    def _load_translations(self):
        """Load translation files for supported languages"""
        # In production, these would be loaded from files
        self.translations = {
            SupportedLanguage.ENGLISH.value: {
                "startup_idea": "Startup Idea",
                "description": "Description", 
                "category": "Category",
                "status": "Status",
                "created": "Created",
                "validation_error": "Validation Error",
                "success_message": "Operation completed successfully",
                "processing": "Processing...",
                "budget_warning": "Budget warning: {threshold}% reached",
                "idea_validated": "Idea has been validated",
                "similar_ideas_found": "Found {count} similar ideas"
            },
            SupportedLanguage.SPANISH.value: {
                "startup_idea": "Idea de Startup",
                "description": "Descripción",
                "category": "Categoría", 
                "status": "Estado",
                "created": "Creado",
                "validation_error": "Error de Validación",
                "success_message": "Operación completada con éxito",
                "processing": "Procesando...",
                "budget_warning": "Advertencia de presupuesto: {threshold}% alcanzado",
                "idea_validated": "La idea ha sido validada",
                "similar_ideas_found": "Se encontraron {count} ideas similares"
            },
            SupportedLanguage.FRENCH.value: {
                "startup_idea": "Idée de Startup",
                "description": "Description",
                "category": "Catégorie",
                "status": "Statut", 
                "created": "Créé",
                "validation_error": "Erreur de Validation",
                "success_message": "Opération réussie",
                "processing": "Traitement en cours...",
                "budget_warning": "Avertissement budget: {threshold}% atteint",
                "idea_validated": "L'idée a été validée",
                "similar_ideas_found": "Trouvé {count} idées similaires"
            },
            SupportedLanguage.GERMAN.value: {
                "startup_idea": "Startup-Idee",
                "description": "Beschreibung",
                "category": "Kategorie",
                "status": "Status",
                "created": "Erstellt", 
                "validation_error": "Validierungsfehler",
                "success_message": "Operation erfolgreich abgeschlossen",
                "processing": "Verarbeitung...",
                "budget_warning": "Budget-Warnung: {threshold}% erreicht",
                "idea_validated": "Idee wurde validiert",
                "similar_ideas_found": "{count} ähnliche Ideen gefunden"
            },
            SupportedLanguage.JAPANESE.value: {
                "startup_idea": "スタートアップのアイデア",
                "description": "説明",
                "category": "カテゴリー",
                "status": "ステータス",
                "created": "作成日",
                "validation_error": "検証エラー", 
                "success_message": "操作が正常に完了しました",
                "processing": "処理中...",
                "budget_warning": "予算警告: {threshold}%に達しました",
                "idea_validated": "アイデアが検証されました",
                "similar_ideas_found": "{count}個の類似アイデアが見つかりました"
            },
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "startup_idea": "创业想法",
                "description": "描述",
                "category": "类别",
                "status": "状态",
                "created": "创建时间",
                "validation_error": "验证错误",
                "success_message": "操作成功完成", 
                "processing": "处理中...",
                "budget_warning": "预算警告：已达到{threshold}%",
                "idea_validated": "想法已通过验证",
                "similar_ideas_found": "找到{count}个类似想法"
            }
        }
    
    def get_text(self, key: str, **kwargs) -> str:
        """Get localized text for a given key"""
        language = self.config.language.value
        
        if language not in self.translations:
            language = SupportedLanguage.ENGLISH.value
            
        translation = self.translations[language].get(key, key)
        
        # Format with provided arguments
        try:
            return translation.format(**kwargs)
        except (KeyError, ValueError):
            return translation
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes"""
        return [lang.value for lang in SupportedLanguage]
    
    def set_language(self, language: SupportedLanguage):
        """Change the current language"""
        self.config.language = language
        logger.info(f"Language changed to {language.value}")
    
    def set_region(self, region: Region):
        """Change the current region"""
        self.config.region = region
        logger.info(f"Region changed to {region.value}")
    
    def format_currency(self, amount: float) -> str:
        """Format currency based on regional settings"""
        currency_formats = {
            Region.NORTH_AMERICA: lambda x: f"${x:,.2f}",
            Region.EUROPE: lambda x: f"€{x:,.2f}".replace(",", " "),
            Region.ASIA_PACIFIC: lambda x: f"¥{x:,.0f}",
            Region.LATIN_AMERICA: lambda x: f"${x:,.2f}"
        }
        
        formatter = currency_formats.get(self.config.region, currency_formats[Region.NORTH_AMERICA])
        return formatter(amount)
    
    def format_date(self, date_obj) -> str:
        """Format date based on regional preferences"""
        date_formats = {
            Region.NORTH_AMERICA: "%m/%d/%Y",
            Region.EUROPE: "%d.%m.%Y", 
            Region.ASIA_PACIFIC: "%Y年%m月%d日",
            Region.LATIN_AMERICA: "%d/%m/%Y"
        }
        
        date_format = date_formats.get(self.config.region, date_formats[Region.NORTH_AMERICA])
        return date_obj.strftime(date_format)

class ComplianceManager:
    """Manages regional compliance requirements"""
    
    def __init__(self, region: Region = Region.NORTH_AMERICA):
        """Initialize compliance manager for specific region"""
        self.region = region
        self.compliance_rules = self._load_compliance_rules()
        
        logger.info(f"Compliance manager initialized for {region.value}")
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules for the region"""
        rules = {
            Region.NORTH_AMERICA: {
                "data_retention_days": 2555,  # 7 years
                "requires_consent": True,
                "allows_profiling": True,
                "data_transfer_restrictions": False,
                "privacy_regulations": ["CCPA"],
                "required_disclosures": ["data_collection", "third_party_sharing"]
            },
            Region.EUROPE: {
                "data_retention_days": 1095,  # 3 years default
                "requires_consent": True,
                "allows_profiling": False,  # Requires explicit consent
                "data_transfer_restrictions": True,
                "privacy_regulations": ["GDPR"],
                "required_disclosures": ["data_collection", "legal_basis", "data_subject_rights"]
            },
            Region.ASIA_PACIFIC: {
                "data_retention_days": 1825,  # 5 years
                "requires_consent": True,
                "allows_profiling": True,
                "data_transfer_restrictions": True,
                "privacy_regulations": ["PDPA"],
                "required_disclosures": ["data_collection", "cross_border_transfer"]
            },
            Region.LATIN_AMERICA: {
                "data_retention_days": 1825,  # 5 years
                "requires_consent": True, 
                "allows_profiling": True,
                "data_transfer_restrictions": False,
                "privacy_regulations": ["LGPD"],
                "required_disclosures": ["data_collection", "retention_period"]
            }
        }
        
        return rules.get(self.region, rules[Region.NORTH_AMERICA])
    
    def validate_data_collection(self, data_types: list[str]) -> bool:
        """Validate that data collection complies with regional rules"""
        if not self.compliance_rules["requires_consent"]:
            return True
            
        # In production, this would check against consent records
        logger.info(f"Validating data collection for {len(data_types)} data types")
        return True
    
    def get_retention_period(self) -> int:
        """Get data retention period in days for the region"""
        return self.compliance_rules["data_retention_days"]
    
    def requires_explicit_consent_for_profiling(self) -> bool:
        """Check if profiling requires explicit consent"""
        return not self.compliance_rules["allows_profiling"]
    
    def get_required_disclosures(self) -> list[str]:
        """Get list of required privacy disclosures"""
        return self.compliance_rules["required_disclosures"]
    
    def validate_cross_border_transfer(self, target_region: Region) -> bool:
        """Validate if cross-border data transfer is allowed"""
        if not self.compliance_rules["data_transfer_restrictions"]:
            return True
            
        # Simplified validation - in production would check adequacy decisions
        allowed_transfers = {
            Region.EUROPE: [Region.NORTH_AMERICA],  # Assuming adequate protection
            Region.ASIA_PACIFIC: [Region.NORTH_AMERICA, Region.EUROPE]
        }
        
        return target_region in allowed_transfers.get(self.region, [])

# Global instances
_i18n_manager: Optional[InternationalizationManager] = None
_compliance_manager: Optional[ComplianceManager] = None

def get_i18n_manager(config: Optional[LocalizationConfig] = None) -> InternationalizationManager:
    """Get or create global i18n manager"""
    global _i18n_manager
    if _i18n_manager is None or config is not None:
        _i18n_manager = InternationalizationManager(config)
    return _i18n_manager

def get_compliance_manager(region: Optional[Region] = None) -> ComplianceManager:
    """Get or create global compliance manager"""
    global _compliance_manager
    if _compliance_manager is None or region is not None:
        _compliance_manager = ComplianceManager(region or Region.NORTH_AMERICA)
    return _compliance_manager

def translate(key: str, **kwargs) -> str:
    """Convenience function for translation"""
    return get_i18n_manager().get_text(key, **kwargs)

def format_currency(amount: float, region: Optional[Region] = None) -> str:
    """Convenience function for currency formatting"""
    if region:
        manager = InternationalizationManager(LocalizationConfig(region=region))
        return manager.format_currency(amount)
    return get_i18n_manager().format_currency(amount)