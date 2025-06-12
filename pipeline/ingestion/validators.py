"""
Comprehensive validation engine for idea ingestion.

This module provides input sanitization, business rule validation,
and security checks for startup idea data.
"""

import html
import re
import logging
from typing import List, Dict, Set, Optional, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup # Import BeautifulSoup

from pipeline.models.idea import IdeaDraft, ValidationResult, IdeaCategory
from pipeline.config.settings import get_validation_config, ValidationConfig

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Handles sanitization of user input data."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._html_tag_pattern = re.compile(r'<[^>]+>')
        self._script_pattern = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
        self._dangerous_protocols = {'javascript:', 'data:', 'vbscript:', 'file:'}
        
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input by removing dangerous content.
        
        Args:
            text: Raw text input
            
        Returns:
            Sanitized text safe for storage and display
        """
        if not text:
            return ""
        
        # Remove script tags first (most dangerous)
        sanitized = self._script_pattern.sub('', text)

        # Use BeautifulSoup to remove HTML tags if sanitization is enabled
        if self.config.enable_html_sanitization:
            soup = BeautifulSoup(sanitized, 'html.parser')
            sanitized = soup.get_text(separator=' ')
            logger.debug(f"HTML sanitization applied. Original: '{text[:50]}...', Sanitized: '{sanitized[:50]}...'")
        else:
            logger.debug(f"HTML sanitization skipped for: '{text[:50]}...'")
        
        # HTML escape remaining special characters to prevent XSS
        sanitized = html.escape(sanitized, quote=True)
        
        # Log the final sanitization step
        logger.debug(f"Final HTML escaping applied. Result: '{sanitized[:50]}...'")
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized.strip())
        
        # Check for dangerous protocols
        for protocol in self._dangerous_protocols:
            if protocol in sanitized.lower():
                logger.warning(
                    f"Dangerous protocol detected in text: {protocol}",
                    extra={"text_preview": sanitized[:100]}
                )
                sanitized = sanitized.replace(protocol, '')
        
        return sanitized
    
    def sanitize_url(self, url: str) -> Optional[str]:
        """
        Sanitize and validate URL.
        
        Args:
            url: Raw URL input
            
        Returns:
            Sanitized URL or None if invalid
        """
        if not url or not url.strip():
            return None
        
        url = url.strip()
        
        try:
            parsed = urlparse(url)
            
            # Only allow http/https protocols
            if parsed.scheme not in ('http', 'https'):
                logger.warning(f"Invalid URL scheme: {parsed.scheme}")
                return None
            
            # Basic hostname validation
            if not parsed.netloc:
                logger.warning(f"Invalid URL hostname: {url}")
                return None
            
            # Reconstruct clean URL
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"
            
            return clean_url
            
        except Exception as e:
            logger.warning(f"URL parsing failed: {e}")
            return None
    
    def sanitize_idea_draft(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize all fields in an idea draft.
        
        Args:
            draft: Raw idea draft data
            
        Returns:
            Sanitized idea draft data
        """
        sanitized = {}
        
        # Text fields that need sanitization
        text_fields = ['title', 'description', 'problem_statement', 
                      'solution_description', 'target_market']
        
        for field, value in draft.items():
            if field in text_fields and isinstance(value, str):
                sanitized[field] = self.sanitize_text(value)
            elif field == 'evidence_links' and isinstance(value, list):
                sanitized[field] = [
                    url for url in [self.sanitize_url(link) for link in value]
                    if url is not None
                ]
            elif field == 'category':
                # Validate category against enum
                try:
                    if value and value.lower() in [cat.value for cat in IdeaCategory]:
                        sanitized[field] = value.lower()
                    else:
                        sanitized[field] = IdeaCategory.UNCATEGORIZED.value
                except (AttributeError, ValueError):
                    sanitized[field] = IdeaCategory.UNCATEGORIZED.value
            else:
                sanitized[field] = value
        
        return sanitized


class ContentValidator:
    """Validates content quality and appropriateness."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._profanity_words = self._load_profanity_list()
        self._spam_patterns = self._load_spam_patterns()
    
    def _load_profanity_list(self) -> Set[str]:
        """Load profanity word list. In production, this would come from a file."""
        # Basic profanity list - in production this would be more comprehensive
        return {
            'spam', 'scam', 'fraud', 'fake', 'phishing',
            # Add more words as needed
        }
    
    def _load_spam_patterns(self) -> List[re.Pattern]:
        """Load spam detection patterns."""
        patterns = [
            re.compile(r'(.)\1{4,}'),  # Repeated characters
            re.compile(r'[A-Z]{10,}'),  # Too many caps
            re.compile(r'www\.[a-z]+\.[a-z]{2,3}', re.IGNORECASE),  # URLs in text
            re.compile(r'\b(buy now|click here|limited time)\b', re.IGNORECASE),
            re.compile(r'[$€£¥]{2,}'),  # Multiple currency symbols
        ]
        return patterns
    
    def check_profanity(self, text: str) -> ValidationResult:
        """
        Check text for inappropriate content.
        
        Args:
            text: Text to check
            
        Returns:
            ValidationResult with any profanity violations
        """
        result = ValidationResult(is_valid=True)
        
        if not self.config.enable_profanity_filter:
            return result
        
        text_lower = text.lower()
        found_words = []
        
        for word in self._profanity_words:
            if word in text_lower:
                found_words.append(word)
        
        if found_words:
            result.add_error(f"Inappropriate content detected: {', '.join(found_words)}")
            logger.warning(
                f"Profanity detected in text",
                extra={"words": found_words, "text_preview": text[:50]}
            )
        
        return result
    
    def check_spam_patterns(self, text: str) -> ValidationResult:
        """
        Check text for spam patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            ValidationResult with any spam violations
        """
        result = ValidationResult(is_valid=True)
        
        if not self.config.enable_spam_detection:
            return result
        
        spam_indicators = []
        
        for pattern in self._spam_patterns:
            if pattern.search(text):
                spam_indicators.append(pattern.pattern)
        
        # Check for excessive capitalization
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.6:
                spam_indicators.append("excessive_capitalization")
        
        # Check for repeated phrases
        words = text.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
            
            repeated_words = [word for word, count in word_counts.items() if count > 3]
            if repeated_words:
                spam_indicators.append("repeated_words")
        
        if spam_indicators:
            result.add_warning(f"Potential spam patterns detected: {', '.join(spam_indicators)}")
            logger.info(
                f"Spam patterns detected",
                extra={"patterns": spam_indicators, "text_preview": text[:50]}
            )
        
        return result
    
    def validate_content_quality(self, draft: IdeaDraft) -> ValidationResult:
        """
        Comprehensive content quality validation.
        
        Args:
            draft: Idea draft to validate
            
        Returns:
            ValidationResult with all content issues
        """
        result = ValidationResult(is_valid=True)
        
        # Check title
        profanity_result = self.check_profanity(draft.title)
        if profanity_result.has_errors():
            result.errors.extend(profanity_result.errors)
        
        spam_result = self.check_spam_patterns(draft.title)
        if spam_result.has_warnings():
            result.warnings.extend(spam_result.warnings)
        
        # Check description
        desc_profanity = self.check_profanity(draft.description)
        if desc_profanity.has_errors():
            result.errors.extend(desc_profanity.errors)
        
        desc_spam = self.check_spam_patterns(draft.description)
        if desc_spam.has_warnings():
            result.warnings.extend(desc_spam.warnings)
        
        # Check for minimum meaningful content
        if len(draft.description.split()) < 5:
            result.add_warning("Description may be too brief for meaningful analysis")
        
        # Validate evidence links if provided
        if draft.evidence_links:
            for i, link in enumerate(draft.evidence_links):
                if not self._is_valid_evidence_url(link):
                    result.add_warning(f"Evidence link {i+1} may not be accessible")
        
        return result
    
    def _is_valid_evidence_url(self, url: str) -> bool:
        """Basic validation of evidence URL format."""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ('http', 'https') and
                parsed.netloc and
                len(parsed.netloc) > 3
            )
        except Exception:
            return False


class BusinessRuleValidator:
    """Validates business rules and constraints."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._required_fields = ['title', 'description']
        self._optional_fields = [
            'problem_statement', 'solution_description', 
            'target_market', 'evidence_links'
        ]
    
    def validate_field_lengths(self, draft: IdeaDraft) -> ValidationResult:
        """
        Validate field length constraints.
        
        Args:
            draft: Idea draft to validate
            
        Returns:
            ValidationResult with length violations
        """
        result = ValidationResult(is_valid=True)
        
        # Title length validation
        if len(draft.title) < self.config.min_title_length:
            result.add_error(
                f"Title must be at least {self.config.min_title_length} characters"
            )
        elif len(draft.title) > self.config.max_title_length:
            result.add_error(
                f"Title must be no more than {self.config.max_title_length} characters"
            )
        
        # Description length validation
        if len(draft.description) < self.config.min_description_length:
            result.add_error(
                f"Description must be at least {self.config.min_description_length} characters"
            )
        elif len(draft.description) > self.config.max_description_length:
            result.add_error(
                f"Description must be no more than {self.config.max_description_length} characters"
            )
        
        # Optional field length validation
        max_optional_length = 1000
        
        for field_name in ['problem_statement', 'solution_description']:
            field_value = getattr(draft, field_name, None)
            if field_value and len(field_value) > max_optional_length:
                result.add_error(
                    f"{field_name} must be no more than {max_optional_length} characters"
                )
        
        if draft.target_market and len(draft.target_market) > 500:
            result.add_error("Target market description must be no more than 500 characters")
        
        return result
    
    def validate_required_fields(self, draft: IdeaDraft) -> ValidationResult:
        """
        Validate that required fields are present and meaningful.
        
        Args:
            draft: Idea draft to validate
            
        Returns:
            ValidationResult with missing field errors
        """
        result = ValidationResult(is_valid=True)
        
        # Check title
        if not draft.title or not draft.title.strip():
            result.add_error("Title is required")
        
        # Check description
        if not draft.description or not draft.description.strip():
            result.add_error("Description is required")
        
        # Validate category
        if draft.category and draft.category not in [cat.value for cat in IdeaCategory]:
            result.add_error(f"Invalid category: {draft.category}")
        
        return result
    
    def validate_business_logic(self, draft: IdeaDraft) -> ValidationResult:
        """
        Validate business-specific rules and constraints.
        
        Args:
            draft: Idea draft to validate
            
        Returns:
            ValidationResult with business rule violations
        """
        result = ValidationResult(is_valid=True)
        
        # Check for meaningful title-description relationship
        title_words = set(draft.title.lower().split())
        desc_words = set(draft.description.lower().split())
        
        common_words = title_words.intersection(desc_words)
        # Remove common stop words for better analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        meaningful_common = common_words - stop_words
        
        if len(meaningful_common) == 0 and len(title_words) > 2:
            result.add_warning("Title and description seem unrelated")
        
        # Validate evidence links count
        if draft.evidence_links and len(draft.evidence_links) > 10:
            result.add_warning("Too many evidence links may indicate spam")
        
        # Check for duplicate evidence links
        if draft.evidence_links:
            unique_links = set(draft.evidence_links)
            if len(unique_links) != len(draft.evidence_links):
                result.add_warning("Duplicate evidence links detected")
        
        return result


class IdeaValidator:
    """Main validator orchestrating all validation checks."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or get_validation_config()
        self.sanitizer = InputSanitizer(self.config)
        self.content_validator = ContentValidator(self.config)
        self.business_validator = BusinessRuleValidator(self.config)
    
    def validate_and_sanitize_draft(self, raw_data: Dict[str, Any]) -> tuple[IdeaDraft, ValidationResult]:
        """
        Complete validation and sanitization of idea draft.
        
        Args:
            raw_data: Raw input data dictionary
            
        Returns:
            Tuple of (sanitized_draft, validation_result)
        """
        overall_result = ValidationResult(is_valid=True)
        
        try:
            # Step 1: Sanitize input data
            sanitized_data = self.sanitizer.sanitize_idea_draft(raw_data)
            
            # Step 2: Create draft object (this performs Pydantic validation)
            try:
                draft = IdeaDraft(**sanitized_data)
            except Exception as e:
                overall_result.add_error(f"Invalid data format: {str(e)}")
                # Return minimal valid draft if creation fails
                fallback_draft = IdeaDraft(
                    title="INVALID_DATA_FALLBACK",
                    description="Data validation failed during processing"
                )
                return fallback_draft, overall_result
            
            # Step 3: Business rule validation
            business_result = self.business_validator.validate_required_fields(draft)
            overall_result.errors.extend(business_result.errors)
            overall_result.warnings.extend(business_result.warnings)
            
            length_result = self.business_validator.validate_field_lengths(draft)
            overall_result.errors.extend(length_result.errors)
            overall_result.warnings.extend(length_result.warnings)
            
            logic_result = self.business_validator.validate_business_logic(draft)
            overall_result.errors.extend(logic_result.errors)
            overall_result.warnings.extend(logic_result.warnings)
            
            # Step 4: Content quality validation
            content_result = self.content_validator.validate_content_quality(draft)
            overall_result.errors.extend(content_result.errors)
            overall_result.warnings.extend(content_result.warnings)
            
            # Step 5: Set overall validity
            overall_result.is_valid = len(overall_result.errors) == 0
            
            logger.info(
                f"Idea validation completed",
                extra={
                    "is_valid": overall_result.is_valid,
                    "error_count": len(overall_result.errors),
                    "warning_count": len(overall_result.warnings),
                    "title": draft.title[:50]
                }
            )
            
            return draft, overall_result
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            overall_result.add_error(f"Validation system error: {str(e)}")
            fallback_draft = IdeaDraft(
                title="VALIDATION_ERROR_FALLBACK",
                description="System validation error occurred during processing"
            )
            return fallback_draft, overall_result
    
    def validate_partial_update(self, existing_draft: IdeaDraft, updates: Dict[str, Any]) -> ValidationResult:
        """
        Validate partial updates to existing idea.
        
        Args:
            existing_draft: Current idea draft
            updates: Fields to update
            
        Returns:
            ValidationResult for the updates
        """
        result = ValidationResult(is_valid=True)
        
        # Sanitize update data
        sanitized_updates = self.sanitizer.sanitize_idea_draft(updates)
        
        # Validate individual fields being updated
        for field, value in sanitized_updates.items():
            if field == 'title' and value:
                if not self.config.min_title_length <= len(value) <= self.config.max_title_length:
                    result.add_error(f"Title length must be between {self.config.min_title_length} and {self.config.max_title_length} characters")
            
            elif field == 'description' and value:
                if not self.config.min_description_length <= len(value) <= self.config.max_description_length:
                    result.add_error(f"Description length must be between {self.config.min_description_length} and {self.config.max_description_length} characters")
            
            elif field == 'category' and value:
                if value not in [cat.value for cat in IdeaCategory]:
                    result.add_error(f"Invalid category: {value}")
        
        return result


# Factory function for easy instantiation
def create_validator(config: Optional[ValidationConfig] = None) -> IdeaValidator:
    """
    Create a new IdeaValidator instance.
    
    Args:
        config: Optional validation configuration
        
    Returns:
        Configured IdeaValidator instance
    """
    return IdeaValidator(config)