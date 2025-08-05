"""
Quantum Task Planner Internationalization (i18n)

Multi-language support for the quantum task planning system:
- Text translation for UI elements and messages
- Locale-aware formatting for dates, numbers, currencies
- Right-to-left (RTL) language support
- Cultural adaptation for quantum concepts
- GDPR/Privacy compliance messaging
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import locale
import gettext

logger = logging.getLogger(__name__)


class SupportedLocale(str, Enum):
    """Supported locales for the quantum task planner."""
    
    # Primary languages (Tier 1)
    ENGLISH_US = "en_US"
    SPANISH_ES = "es_ES"
    FRENCH_FR = "fr_FR"
    GERMAN_DE = "de_DE"
    JAPANESE_JA = "ja_JP"
    CHINESE_CN = "zh_CN"
    
    # Additional languages (Tier 2)
    PORTUGUESE_BR = "pt_BR"
    RUSSIAN_RU = "ru_RU"
    KOREAN_KR = "ko_KR"
    ITALIAN_IT = "it_IT"
    DUTCH_NL = "nl_NL"
    ARABIC_SA = "ar_SA"
    
    @classmethod
    def get_language_code(cls, locale_code: str) -> str:
        """Extract language code from locale (e.g., 'en' from 'en_US')."""
        return locale_code.split('_')[0]
    
    @classmethod 
    def is_rtl(cls, locale_code: str) -> bool:
        """Check if locale uses right-to-left text direction."""
        rtl_languages = {'ar', 'he', 'fa', 'ur'}
        return cls.get_language_code(locale_code) in rtl_languages


@dataclass
class LocaleInfo:
    """Information about a specific locale."""
    
    code: str
    name: str
    native_name: str
    language: str
    country: str
    is_rtl: bool = False
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    currency_symbol: str = "$"
    decimal_separator: str = "."
    thousands_separator: str = ","
    
    @classmethod
    def from_locale_code(cls, locale_code: str) -> 'LocaleInfo':
        """Create LocaleInfo from locale code."""
        locale_data = {
            "en_US": cls("en_US", "English (US)", "English", "en", "US", False, "%m/%d/%Y", "%I:%M %p", "$"),
            "es_ES": cls("es_ES", "Spanish (Spain)", "Español", "es", "ES", False, "%d/%m/%Y", "%H:%M", "€"),
            "fr_FR": cls("fr_FR", "French (France)", "Français", "fr", "FR", False, "%d/%m/%Y", "%H:%M", "€"),
            "de_DE": cls("de_DE", "German (Germany)", "Deutsch", "de", "DE", False, "%d.%m.%Y", "%H:%M", "€"),
            "ja_JP": cls("ja_JP", "Japanese (Japan)", "日本語", "ja", "JP", False, "%Y/%m/%d", "%H:%M", "¥"),
            "zh_CN": cls("zh_CN", "Chinese (China)", "中文", "zh", "CN", False, "%Y-%m-%d", "%H:%M", "¥"),
            "pt_BR": cls("pt_BR", "Portuguese (Brazil)", "Português", "pt", "BR", False, "%d/%m/%Y", "%H:%M", "R$"),
            "ru_RU": cls("ru_RU", "Russian (Russia)", "Русский", "ru", "RU", False, "%d.%m.%Y", "%H:%M", "₽"),
            "ko_KR": cls("ko_KR", "Korean (Korea)", "한국어", "ko", "KR", False, "%Y.%m.%d", "%H:%M", "₩"),
            "it_IT": cls("it_IT", "Italian (Italy)", "Italiano", "it", "IT", False, "%d/%m/%Y", "%H:%M", "€"),
            "nl_NL": cls("nl_NL", "Dutch (Netherlands)", "Nederlands", "nl", "NL", False, "%d-%m-%Y", "%H:%M", "€"),
            "ar_SA": cls("ar_SA", "Arabic (Saudi Arabia)", "العربية", "ar", "SA", True, "%d/%m/%Y", "%H:%M", "﷼"),
        }
        
        return locale_data.get(locale_code, locale_data["en_US"])


class QuantumI18nManager:
    """Main internationalization manager for quantum task planner."""
    
    def __init__(self, default_locale: str = "en_US", translations_dir: Optional[Path] = None):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations_dir = translations_dir or Path(__file__).parent / "translations"
        
        # Translation catalogs
        self.translations: Dict[str, Dict[str, str]] = {}
        self.locale_info: Dict[str, LocaleInfo] = {}
        
        # Quantum-specific terminology
        self.quantum_terms: Dict[str, Dict[str, str]] = {}
        
        # Initialize translations
        self._load_translations()
        self._load_quantum_terminology()
        self._initialize_locale_info()
    
    def _load_translations(self):
        """Load translation files for all supported locales."""
        try:
            if not self.translations_dir.exists():
                self.translations_dir.mkdir(parents=True)
                logger.warning(f"Created translations directory: {self.translations_dir}")
            
            for locale_code in SupportedLocale:
                translation_file = self.translations_dir / f"{locale_code.value}.json"
                
                if translation_file.exists():
                    try:
                        with open(translation_file, 'r', encoding='utf-8') as f:
                            self.translations[locale_code.value] = json.load(f)
                        logger.debug(f"Loaded translations for {locale_code.value}")
                    except (json.JSONDecodeError, IOError) as e:
                        logger.error(f"Failed to load translations for {locale_code.value}: {e}")
                        self.translations[locale_code.value] = {}
                else:
                    # Create empty translation file template
                    self.translations[locale_code.value] = {}
                    self._create_translation_template(locale_code.value)
                    
        except Exception as e:
            logger.error(f"Failed to initialize translations: {e}")
            self.translations[self.default_locale] = {}
    
    def _create_translation_template(self, locale_code: str):
        """Create a translation template file."""
        template = self._get_base_translations()
        
        template_file = self.translations_dir / f"{locale_code}.json"
        try:
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            logger.info(f"Created translation template: {template_file}")
        except IOError as e:
            logger.error(f"Failed to create translation template for {locale_code}: {e}")
    
    def _get_base_translations(self) -> Dict[str, str]:
        """Get base translation keys that need to be translated."""
        return {
            # Common UI elements
            "task": "Task",
            "tasks": "Tasks",
            "quantum_task": "Quantum Task",
            "quantum_tasks": "Quantum Tasks",
            "title": "Title",
            "description": "Description",
            "priority": "Priority",
            "status": "Status",
            "created": "Created",
            "updated": "Updated",
            "due_date": "Due Date",
            "estimated_duration": "Estimated Duration",
            "dependencies": "Dependencies",
            
            # Quantum states
            "quantum_state_superposition": "Superposition",
            "quantum_state_pending": "Pending",
            "quantum_state_executing": "Executing", 
            "quantum_state_completed": "Completed",
            "quantum_state_failed": "Failed",
            "quantum_state_cancelled": "Cancelled",
            "quantum_state_blocked": "Blocked",
            
            # Quantum priorities
            "priority_ground_state": "Ground State",
            "priority_excited_1": "Excited Level 1",
            "priority_excited_2": "Excited Level 2",
            "priority_excited_3": "Excited Level 3",
            "priority_ionized": "Ionized",
            
            # Actions
            "create": "Create",
            "edit": "Edit",
            "delete": "Delete",
            "save": "Save",
            "cancel": "Cancel",
            "execute": "Execute",
            "schedule": "Schedule",
            "measure": "Measure",
            "entangle": "Entangle",
            
            # Quantum operations
            "quantum_measurement": "Quantum Measurement",
            "quantum_evolution": "Quantum Evolution",
            "quantum_entanglement": "Quantum Entanglement",
            "quantum_superposition": "Quantum Superposition",
            "quantum_interference": "Quantum Interference",
            "quantum_tunneling": "Quantum Tunneling",
            "quantum_coherence": "Quantum Coherence",
            "quantum_decoherence": "Quantum Decoherence",
            
            # Messages
            "task_created": "Task created successfully",
            "task_updated": "Task updated successfully",
            "task_deleted": "Task deleted successfully",
            "task_executed": "Task executed successfully",
            "measurement_completed": "Quantum measurement completed",
            "entanglement_created": "Quantum entanglement created",
            
            # Errors
            "error_invalid_input": "Invalid input provided",
            "error_task_not_found": "Task not found",
            "error_measurement_failed": "Quantum measurement failed",
            "error_entanglement_failed": "Quantum entanglement failed",
            "error_validation_failed": "Validation failed",
            
            # Time units
            "seconds": "seconds",
            "minutes": "minutes", 
            "hours": "hours",
            "days": "days",
            "weeks": "weeks",
            "months": "months",
            
            # Privacy and compliance
            "privacy_notice": "Privacy Notice",
            "data_processing_notice": "We process your data in accordance with privacy regulations",
            "gdpr_compliance": "GDPR Compliant",
            "data_retention_notice": "Data is retained according to our retention policy",
            "consent_required": "Consent is required for data processing",
        }
    
    def _load_quantum_terminology(self):
        """Load quantum-specific terminology translations."""
        # Quantum physics concepts need careful translation to maintain scientific accuracy
        self.quantum_terms = {
            "en_US": {
                "superposition": "A quantum state where a task exists in multiple states simultaneously",
                "entanglement": "A quantum phenomenon where tasks become correlated",
                "measurement": "The process of observing a quantum task state",
                "coherence": "The degree of quantum correlation in the system",
                "interference": "Quantum effect that can enhance or reduce probabilities",
                "tunneling": "Quantum effect allowing tasks to overcome barriers"
            },
            "es_ES": {
                "superposition": "Un estado cuántico donde una tarea existe en múltiples estados simultáneamente",
                "entanglement": "Un fenómeno cuántico donde las tareas se correlacionan",
                "measurement": "El proceso de observar el estado de una tarea cuántica",
                "coherence": "El grado de correlación cuántica en el sistema",
                "interference": "Efecto cuántico que puede mejorar o reducir probabilidades",
                "tunneling": "Efecto cuántico que permite a las tareas superar barreras"
            },
            "fr_FR": {
                "superposition": "Un état quantique où une tâche existe dans plusieurs états simultanément",
                "entanglement": "Un phénomène quantique où les tâches deviennent corrélées",
                "measurement": "Le processus d'observation de l'état d'une tâche quantique",
                "coherence": "Le degré de corrélation quantique dans le système",
                "interference": "Effet quantique qui peut améliorer ou réduire les probabilités",
                "tunneling": "Effet quantique permettant aux tâches de surmonter les barrières"
            },
            "de_DE": {
                "superposition": "Ein Quantenzustand, in dem eine Aufgabe in mehreren Zuständen gleichzeitig existiert",
                "entanglement": "Ein Quantenphänomen, bei dem Aufgaben korreliert werden",
                "measurement": "Der Prozess der Beobachtung eines Quantenaufgabenzustands",
                "coherence": "Der Grad der Quantenkorrelation im System",
                "interference": "Quanteneffekt, der Wahrscheinlichkeiten verstärken oder reduzieren kann",
                "tunneling": "Quanteneffekt, der es Aufgaben ermöglicht, Barrieren zu überwinden"
            },
            "ja_JP": {
                "superposition": "タスクが複数の状態に同時に存在する量子状態",
                "entanglement": "タスクが相関する量子現象",
                "measurement": "量子タスク状態を観測するプロセス",
                "coherence": "システム内の量子相関の度合い",
                "interference": "確率を向上または減少させる量子効果",
                "tunneling": "タスクが障壁を克服することを可能にする量子効果"
            },
            "zh_CN": {
                "superposition": "任务同时存在于多个状态的量子态",
                "entanglement": "任务变得相关的量子现象",
                "measurement": "观察量子任务状态的过程",
                "coherence": "系统中量子关联的程度",
                "interference": "可以增强或减少概率的量子效应",
                "tunneling": "允许任务克服障碍的量子效应"
            }
        }
    
    def _initialize_locale_info(self):
        """Initialize locale information for all supported locales."""
        for locale_code in SupportedLocale:
            self.locale_info[locale_code.value] = LocaleInfo.from_locale_code(locale_code.value)
    
    def set_locale(self, locale_code: str) -> bool:
        """
        Set the current locale.
        
        Args:
            locale_code: Locale code (e.g., 'en_US', 'es_ES')
            
        Returns:
            True if locale was set successfully
        """
        if locale_code in self.translations:
            self.current_locale = locale_code
            logger.info(f"Locale set to {locale_code}")
            return True
        else:
            logger.warning(f"Locale {locale_code} not supported, keeping {self.current_locale}")
            return False
    
    def get_current_locale(self) -> str:
        """Get the current locale code."""
        return self.current_locale
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locale codes."""
        return list(self.translations.keys())
    
    def get_locale_info(self, locale_code: Optional[str] = None) -> LocaleInfo:
        """Get locale information."""
        locale_code = locale_code or self.current_locale
        return self.locale_info.get(locale_code, self.locale_info[self.default_locale])
    
    def translate(self, key: str, locale_code: Optional[str] = None, **kwargs) -> str:
        """
        Translate a text key to the specified or current locale.
        
        Args:
            key: Translation key
            locale_code: Target locale (uses current if not specified)
            **kwargs: Parameters for string formatting
            
        Returns:
            Translated text
        """
        locale_code = locale_code or self.current_locale
        
        # Get translation from current locale
        translations = self.translations.get(locale_code, {})
        translated_text = translations.get(key)
        
        # Fall back to default locale if not found
        if not translated_text and locale_code != self.default_locale:
            default_translations = self.translations.get(self.default_locale, {})
            translated_text = default_translations.get(key)
        
        # Fall back to key itself if no translation found
        if not translated_text:
            translated_text = key
            logger.debug(f"No translation found for key '{key}' in locale '{locale_code}'")
        
        # Apply string formatting if parameters provided
        if kwargs:
            try:
                translated_text = translated_text.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to format translation '{key}': {e}")
        
        return translated_text
    
    def get_quantum_term_explanation(self, term: str, locale_code: Optional[str] = None) -> str:
        """
        Get explanation of quantum terminology in specified locale.
        
        Args:
            term: Quantum term to explain
            locale_code: Target locale (uses current if not specified)
            
        Returns:
            Term explanation
        """
        locale_code = locale_code or self.current_locale
        
        # Get explanation from current locale
        terms = self.quantum_terms.get(locale_code, {})
        explanation = terms.get(term)
        
        # Fall back to English if not found
        if not explanation and locale_code != "en_US":
            english_terms = self.quantum_terms.get("en_US", {})
            explanation = english_terms.get(term)
        
        return explanation or f"Quantum term: {term}"
    
    def format_datetime(self, dt: datetime, locale_code: Optional[str] = None, 
                       include_time: bool = True) -> str:
        """
        Format datetime according to locale conventions.
        
        Args:
            dt: Datetime to format
            locale_code: Target locale (uses current if not specified)
            include_time: Whether to include time component
            
        Returns:
            Formatted datetime string
        """
        locale_info = self.get_locale_info(locale_code)
        
        try:
            if include_time:
                format_str = f"{locale_info.date_format} {locale_info.time_format}"
            else:
                format_str = locale_info.date_format
            
            return dt.strftime(format_str)
            
        except Exception as e:
            logger.warning(f"Failed to format datetime: {e}")
            return dt.isoformat()
    
    def format_duration(self, duration: timedelta, locale_code: Optional[str] = None) -> str:
        """
        Format duration according to locale conventions.
        
        Args:
            duration: Duration to format
            locale_code: Target locale (uses current if not specified)
            
        Returns:
            Formatted duration string
        """
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return self.translate("duration_seconds", locale_code, count=total_seconds)
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return self.translate("duration_minutes", locale_code, count=minutes)
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            return self.translate("duration_hours", locale_code, count=hours)
        else:
            days = total_seconds // 86400
            return self.translate("duration_days", locale_code, count=days)
    
    def format_number(self, number: Union[int, float], locale_code: Optional[str] = None) -> str:
        """
        Format number according to locale conventions.
        
        Args:
            number: Number to format
            locale_code: Target locale (uses current if not specified)
            
        Returns:
            Formatted number string
        """
        locale_info = self.get_locale_info(locale_code)
        
        try:
            # Simple formatting using locale conventions
            if isinstance(number, float):
                number_str = f"{number:.2f}"
            else:
                number_str = str(number)
            
            # Apply thousands separator
            if locale_info.thousands_separator != ",":
                number_str = number_str.replace(",", locale_info.thousands_separator)
            
            # Apply decimal separator
            if locale_info.decimal_separator != ".":
                number_str = number_str.replace(".", locale_info.decimal_separator)
            
            return number_str
            
        except Exception as e:
            logger.warning(f"Failed to format number: {e}")
            return str(number)
    
    def get_privacy_notice(self, locale_code: Optional[str] = None) -> Dict[str, str]:
        """
        Get privacy notice text for compliance (GDPR, etc.).
        
        Args:
            locale_code: Target locale (uses current if not specified)
            
        Returns:
            Dictionary with privacy notice components
        """
        return {
            "title": self.translate("privacy_notice", locale_code),
            "data_processing": self.translate("data_processing_notice", locale_code),
            "gdpr_compliance": self.translate("gdpr_compliance", locale_code),
            "data_retention": self.translate("data_retention_notice", locale_code),
            "consent_required": self.translate("consent_required", locale_code)
        }
    
    def get_rtl_support(self, locale_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Get right-to-left language support information.
        
        Args:
            locale_code: Target locale (uses current if not specified)
            
        Returns:
            RTL support configuration
        """
        locale_info = self.get_locale_info(locale_code)
        
        return {
            "is_rtl": locale_info.is_rtl,
            "text_direction": "rtl" if locale_info.is_rtl else "ltr",
            "css_direction": "direction: rtl;" if locale_info.is_rtl else "direction: ltr;",
            "text_align": "right" if locale_info.is_rtl else "left"
        }
    
    def export_translation_keys(self, output_file: Path):
        """
        Export all translation keys for translator reference.
        
        Args:
            output_file: File to save translation keys
        """
        base_translations = self._get_base_translations()
        quantum_terms = self.quantum_terms.get("en_US", {})
        
        export_data = {
            "translation_keys": base_translations,
            "quantum_terminology": quantum_terms,
            "supported_locales": self.get_supported_locales(),
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Translation keys exported to {output_file}")
        except IOError as e:
            logger.error(f"Failed to export translation keys: {e}")


# Global i18n manager instance
_i18n_manager: Optional[QuantumI18nManager] = None


def get_i18n_manager(default_locale: str = "en_US") -> QuantumI18nManager:
    """Get the global i18n manager instance."""
    global _i18n_manager
    
    if _i18n_manager is None:
        _i18n_manager = QuantumI18nManager(default_locale)
    
    return _i18n_manager


def t(key: str, locale_code: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function for translation.
    
    Args:
        key: Translation key
        locale_code: Target locale (uses current if not specified)
        **kwargs: Parameters for string formatting
        
    Returns:
        Translated text
    """
    return get_i18n_manager().translate(key, locale_code, **kwargs)


def set_locale(locale_code: str) -> bool:
    """
    Convenience function to set locale.
    
    Args:
        locale_code: Locale code to set
        
    Returns:
        True if locale was set successfully
    """
    return get_i18n_manager().set_locale(locale_code)