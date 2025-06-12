"""
Comprehensive test suite for validation engine components.

Tests input sanitization (SQL injection, XSS prevention), business rule
validation, content quality checks, and security validation with edge cases.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from pipeline.ingestion.validators import (
    InputSanitizer, ContentValidator, BusinessRuleValidator, 
    IdeaValidator, create_validator
)
from pipeline.models.idea import (
    IdeaDraft, ValidationResult, IdeaCategory
)
from pipeline.config.settings import ValidationConfig


class TestInputSanitizer:
    """Test input sanitization for security and data integrity."""
    
    @pytest.fixture
    def config(self) -> ValidationConfig:
        """Provide test configuration."""
        return ValidationConfig()
    
    @pytest.fixture
    def sanitizer(self, config) -> InputSanitizer:
        """Provide InputSanitizer instance."""
        return InputSanitizer(config)
    
    def test_when_clean_text_then_returns_unchanged(self, sanitizer):
        """Given clean text input, when sanitizing, then returns text unchanged."""
        clean_text = "This is a clean text input with no dangerous content."
        
        result = sanitizer.sanitize_text(clean_text)
        
        assert result == clean_text
    
    def test_when_script_tag_then_removes_script(self, sanitizer):
        """Given text with script tag, when sanitizing, then removes script content."""
        dangerous_text = "Hello <script>alert('xss')</script> world"
        
        result = sanitizer.sanitize_text(dangerous_text)
        
        assert "<script" not in result
        assert "alert" not in result
        assert "Hello" in result
        assert "world" in result
    
    def test_when_javascript_protocol_then_removes_protocol(self, sanitizer):
        """Given text with javascript protocol, when sanitizing, then removes dangerous protocol."""
        dangerous_text = "Click javascript:alert('xss') here"
        
        result = sanitizer.sanitize_text(dangerous_text)
        
        assert "javascript:" not in result
        assert "Click" in result
        assert "here" in result
    
    def test_when_html_sanitization_enabled_then_escapes_html(self, config):
        """Given HTML content with sanitization enabled, when sanitizing, then escapes HTML."""
        config.enable_html_sanitization = True
        sanitizer = InputSanitizer(config)
        html_text = "Hello <b>bold</b> & <i>italic</i> text"
        
        result = sanitizer.sanitize_text(html_text)
        
        assert "&lt;b&gt;" in result
        assert "&lt;/b&gt;" in result
        assert "&amp;" in result
    
    def test_when_html_sanitization_disabled_then_strips_tags(self, config):
        """Given HTML content with sanitization disabled, when sanitizing, then strips tags."""
        config.enable_html_sanitization = False
        sanitizer = InputSanitizer(config)
        html_text = "Hello <b>bold</b> text"
        
        result = sanitizer.sanitize_text(html_text)
        
        assert "<b>" not in result
        assert "</b>" not in result
        assert "Hello bold text" == result
    
    def test_when_excessive_whitespace_then_normalizes(self, sanitizer):
        """Given text with excessive whitespace, when sanitizing, then normalizes whitespace."""
        messy_text = "  Hello    world\n\n\ttest  "
        
        result = sanitizer.sanitize_text(messy_text)
        
        assert result == "Hello world test"
    
    def test_when_empty_text_then_returns_empty(self, sanitizer):
        """Given empty text, when sanitizing, then returns empty string."""
        result = sanitizer.sanitize_text("")
        assert result == ""
        
        result = sanitizer.sanitize_text(None)
        assert result == ""
    
    def test_when_all_dangerous_protocols_then_removes_all(self, sanitizer):
        """Given text with all dangerous protocols, when sanitizing, then removes all protocols."""
        dangerous_text = "javascript:alert() data:text/html vbscript:msgbox() file:///etc/passwd"
        
        result = sanitizer.sanitize_text(dangerous_text)
        
        assert "javascript:" not in result
        assert "data:" not in result
        assert "vbscript:" not in result
        assert "file:" not in result
    
    def test_when_valid_http_url_then_returns_url(self, sanitizer):
        """Given valid HTTP URL, when sanitizing URL, then returns sanitized URL."""
        valid_url = "https://example.com/path?param=value"
        
        result = sanitizer.sanitize_url(valid_url)
        
        assert result == "https://example.com/path?param=value"
    
    def test_when_javascript_url_then_returns_none(self, sanitizer):
        """Given javascript URL, when sanitizing URL, then returns None."""
        dangerous_url = "javascript:alert('xss')"
        
        result = sanitizer.sanitize_url(dangerous_url)
        
        assert result is None
    
    def test_when_invalid_scheme_then_returns_none(self, sanitizer):
        """Given URL with invalid scheme, when sanitizing URL, then returns None."""
        invalid_url = "ftp://example.com/file"
        
        result = sanitizer.sanitize_url(invalid_url)
        
        assert result is None
    
    def test_when_malformed_url_then_returns_none(self, sanitizer):
        """Given malformed URL, when sanitizing URL, then returns None."""
        malformed_url = "not-a-url"
        
        result = sanitizer.sanitize_url(malformed_url)
        
        assert result is None
    
    def test_when_empty_url_then_returns_none(self, sanitizer):
        """Given empty URL, when sanitizing URL, then returns None."""
        result = sanitizer.sanitize_url("")
        assert result is None
        
        result = sanitizer.sanitize_url("   ")
        assert result is None
    
    def test_when_idea_draft_data_then_sanitizes_all_fields(self, sanitizer):
        """Given idea draft data, when sanitizing, then sanitizes all text fields."""
        raw_data = {
            "title": "  <script>alert('xss')</script>Great Idea  ",
            "description": "Amazing <b>solution</b> with javascript:alert() link",
            "category": "AI_ML",
            "problem_statement": "Problem with <i>formatting</i>",
            "solution_description": "Solution with data:text/html content",
            "target_market": "Users who like <script>alert()</script>",
            "evidence_links": [
                "https://valid.com",
                "javascript:alert()",
                "not-a-url",
                "https://another-valid.com"
            ]
        }
        
        result = sanitizer.sanitize_idea_draft(raw_data)
        
        # Check text sanitization
        assert "<script" not in result["title"]
        assert "Great Idea" in result["title"]
        
        assert "<b>" not in result["description"]
        assert "javascript:" not in result["description"]
        assert "Amazing solution" in result["description"]
        
        # Check category validation
        assert result["category"] == "ai_ml"
        
        # Check URL filtering
        assert len(result["evidence_links"]) == 2
        assert "https://valid.com" in result["evidence_links"]
        assert "https://another-valid.com" in result["evidence_links"]
    
    def test_when_invalid_category_then_defaults_to_uncategorized(self, sanitizer):
        """Given invalid category, when sanitizing draft, then defaults to uncategorized."""
        raw_data = {
            "title": "Valid title here",
            "description": "Valid description here",
            "category": "invalid_category"
        }
        
        result = sanitizer.sanitize_idea_draft(raw_data)
        
        assert result["category"] == IdeaCategory.UNCATEGORIZED.value


class TestContentValidator:
    """Test content quality and appropriateness validation."""
    
    @pytest.fixture
    def config(self) -> ValidationConfig:
        """Provide test configuration."""
        return ValidationConfig()
    
    @pytest.fixture
    def validator(self, config) -> ContentValidator:
        """Provide ContentValidator instance."""
        return ContentValidator(config)
    
    @pytest.fixture
    def sample_draft(self) -> IdeaDraft:
        """Provide sample idea draft for testing."""
        return IdeaDraft(
            title="AI-powered productivity tool",
            description="Revolutionary solution that uses artificial intelligence to boost productivity"
        )
    
    def test_when_clean_content_then_no_profanity_errors(self, validator, sample_draft):
        """Given clean content, when checking profanity, then returns valid result."""
        result = validator.check_profanity(sample_draft.title)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_when_profanity_filter_disabled_then_skips_check(self, config):
        """Given profanity filter disabled, when checking profanity, then skips validation."""
        config.enable_profanity_filter = False
        validator = ContentValidator(config)
        
        result = validator.check_profanity("spam content here")
        
        assert result.is_valid is True
    
    def test_when_profanity_words_then_adds_errors(self, validator):
        """Given text with profanity words, when checking profanity, then adds errors."""
        profane_text = "This is spam and fraud content"
        
        result = validator.check_profanity(profane_text)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "spam" in result.errors[0] or "fraud" in result.errors[0]
    
    def test_when_no_spam_patterns_then_no_warnings(self, validator):
        """Given text without spam patterns, when checking spam, then returns valid result."""
        clean_text = "This is a normal business description"
        
        result = validator.check_spam_patterns(clean_text)
        
        assert result.is_valid is True
        assert len(result.warnings) == 0
    
    def test_when_spam_detection_disabled_then_skips_check(self, config):
        """Given spam detection disabled, when checking spam, then skips validation."""
        config.enable_spam_detection = False
        validator = ContentValidator(config)
        
        result = validator.check_spam_patterns("CLICK HERE NOW!!! BUY BUY BUY")
        
        assert result.is_valid is True
    
    def test_when_excessive_caps_then_adds_warning(self, validator):
        """Given text with excessive capitalization, when checking spam, then adds warning."""
        caps_text = "THIS IS ALL CAPS TEXT FOR TESTING"
        
        result = validator.check_spam_patterns(caps_text)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "excessive_capitalization" in result.warnings[0]
    
    def test_when_repeated_words_then_adds_warning(self, validator):
        """Given text with repeated words, when checking spam, then adds warning."""
        repeated_text = "buy now buy now buy now buy now"
        
        result = validator.check_spam_patterns(repeated_text)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "repeated_words" in result.warnings[0]
    
    def test_when_spam_keywords_then_adds_warning(self, validator):
        """Given text with spam keywords, when checking spam, then adds warning."""
        spam_text = "Click here for limited time offer! Buy now!"
        
        result = validator.check_spam_patterns(spam_text)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
    
    def test_when_repeated_characters_then_adds_warning(self, validator):
        """Given text with repeated characters, when checking spam, then adds warning."""
        repeated_chars = "Greeeeeeat opportunity"
        
        result = validator.check_spam_patterns(repeated_chars)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
    
    def test_when_comprehensive_validation_then_checks_all_aspects(self, validator, sample_draft):
        """Given idea draft, when validating content quality, then checks all aspects."""
        with patch.object(validator, 'check_profanity') as mock_profanity, \
             patch.object(validator, 'check_spam_patterns') as mock_spam:
            
            mock_profanity.return_value = ValidationResult(is_valid=True)
            mock_spam.return_value = ValidationResult(is_valid=True)
            
            result = validator.validate_content_quality(sample_draft)
            
            # Should check both title and description for profanity
            assert mock_profanity.call_count == 2
            # Should check both title and description for spam
            assert mock_spam.call_count == 2
            
            assert result.is_valid is True
    
    def test_when_brief_description_then_adds_warning(self, validator):
        """Given very brief description, when validating quality, then adds warning."""
        brief_draft = IdeaDraft(
            title="Valid title here",
            description="Brief desc"  # Less than 5 words
        )
        
        result = validator.validate_content_quality(brief_draft)
        
        assert "too brief" in " ".join(result.warnings)
    
    def test_when_invalid_evidence_urls_then_adds_warnings(self, validator):
        """Given invalid evidence URLs, when validating quality, then adds warnings."""
        draft_with_urls = IdeaDraft(
            title="Valid title here",
            description="Valid description here",
            evidence_links=["invalid-url", "https://valid.com"]
        )
        
        result = validator.validate_content_quality(draft_with_urls)
        
        assert any("not be accessible" in warning for warning in result.warnings)
    
    def test_when_validation_errors_propagate_then_accumulates_all(self, validator, sample_draft):
        """Given validation errors from sub-checks, when validating quality, then accumulates all errors."""
        with patch.object(validator, 'check_profanity') as mock_profanity, \
             patch.object(validator, 'check_spam_patterns') as mock_spam:
            
            # Mock profanity error
            profanity_result = ValidationResult(is_valid=False)
            profanity_result.add_error("Profanity detected")
            mock_profanity.return_value = profanity_result
            
            # Mock spam warning
            spam_result = ValidationResult(is_valid=True)
            spam_result.add_warning("Spam pattern detected")
            mock_spam.return_value = spam_result
            
            result = validator.validate_content_quality(sample_draft)
            
            assert len(result.errors) >= 2  # Both title and description profanity errors
            assert len(result.warnings) >= 2  # Both title and description spam warnings


class TestBusinessRuleValidator:
    """Test business rule validation and constraints."""
    
    @pytest.fixture
    def config(self) -> ValidationConfig:
        """Provide test configuration."""
        return ValidationConfig()
    
    @pytest.fixture
    def validator(self, config) -> BusinessRuleValidator:
        """Provide BusinessRuleValidator instance."""
        return BusinessRuleValidator(config)
    
    @pytest.fixture
    def valid_draft(self) -> IdeaDraft:
        """Provide valid idea draft for testing."""
        return IdeaDraft(
            title="AI-powered productivity tool for teams",
            description="Revolutionary solution that uses artificial intelligence to boost team productivity by automating routine tasks"
        )
    
    def test_when_valid_lengths_then_passes_validation(self, validator, valid_draft):
        """Given draft with valid field lengths, when validating lengths, then passes."""
        result = validator.validate_field_lengths(valid_draft)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_when_title_too_short_then_adds_error(self, validator, config):
        """Given title shorter than minimum, when validating lengths, then adds error."""
        short_draft = IdeaDraft(
            title="Short",  # Less than min_title_length (10)
            description="Valid description here"
        )
        
        result = validator.validate_field_lengths(short_draft)
        
        assert result.is_valid is False
        assert any(f"at least {config.min_title_length}" in error for error in result.errors)
    
    def test_when_title_too_long_then_adds_error(self, validator, config):
        """Given title longer than maximum, when validating lengths, then adds error."""
        long_title = "x" * (config.max_title_length + 1)
        long_draft = IdeaDraft(
            title=long_title,
            description="Valid description here"
        )
        
        result = validator.validate_field_lengths(long_draft)
        
        assert result.is_valid is False
        assert any(f"no more than {config.max_title_length}" in error for error in result.errors)
    
    def test_when_description_too_short_then_adds_error(self, validator, config):
        """Given description shorter than minimum, when validating lengths, then adds error."""
        short_desc_draft = IdeaDraft(
            title="Valid title here",
            description="Short"  # Less than min_description_length (10)
        )
        
        result = validator.validate_field_lengths(short_desc_draft)
        
        assert result.is_valid is False
        assert any(f"at least {config.min_description_length}" in error for error in result.errors)
    
    def test_when_optional_fields_too_long_then_adds_errors(self, validator):
        """Given optional fields too long, when validating lengths, then adds errors."""
        long_optional_draft = IdeaDraft(
            title="Valid title here",
            description="Valid description here",
            problem_statement="x" * 1001,  # Exceeds 1000 char limit
            solution_description="y" * 1001,  # Exceeds 1000 char limit
            target_market="z" * 501  # Exceeds 500 char limit
        )
        
        result = validator.validate_field_lengths(long_optional_draft)
        
        assert result.is_valid is False
        assert len(result.errors) == 3  # All three optional fields exceed limits
    
    def test_when_required_fields_present_then_passes(self, validator, valid_draft):
        """Given draft with required fields, when validating required fields, then passes."""
        result = validator.validate_required_fields(valid_draft)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_when_title_missing_then_adds_error(self, validator):
        """Given draft without title, when validating required fields, then adds error."""
        no_title_draft = IdeaDraft(
            title="",  # Empty title
            description="Valid description here"
        )
        
        result = validator.validate_required_fields(no_title_draft)
        
        assert result.is_valid is False
        assert any("Title is required" in error for error in result.errors)
    
    def test_when_description_missing_then_adds_error(self, validator):
        """Given draft without description, when validating required fields, then adds error."""
        no_desc_draft = IdeaDraft(
            title="Valid title here",
            description=""  # Empty description
        )
        
        result = validator.validate_required_fields(no_desc_draft)
        
        assert result.is_valid is False
        assert any("Description is required" in error for error in result.errors)
    
    def test_when_invalid_category_then_adds_error(self, validator):
        """Given draft with invalid category, when validating required fields, then adds error."""
        # This test simulates a scenario where category validation might fail
        # In practice, this would be caught earlier by Pydantic validation
        draft_data = {
            "title": "Valid title here",
            "description": "Valid description here",
            "category": "invalid_category"
        }
        
        # Create a draft with valid category first, then modify
        valid_draft = IdeaDraft(title="Valid title here", description="Valid description here")
        # Simulate invalid category by direct assignment (bypassing Pydantic validation)
        valid_draft.category = "invalid_category"
        
        result = validator.validate_required_fields(valid_draft)
        
        assert result.is_valid is False
        assert any("Invalid category" in error for error in result.errors)
    
    def test_when_related_title_description_then_passes(self, validator):
        """Given title and description with common meaningful words, when validating business logic, then passes."""
        related_draft = IdeaDraft(
            title="AI productivity tool",
            description="This tool uses AI to improve productivity in workplaces"
        )
        
        result = validator.validate_business_logic(related_draft)
        
        # Should not add the "unrelated" warning
        assert not any("unrelated" in warning for warning in result.warnings)
    
    def test_when_unrelated_title_description_then_adds_warning(self, validator):
        """Given title and description with no common words, when validating business logic, then adds warning."""
        unrelated_draft = IdeaDraft(
            title="Mobile banking application",
            description="Educational platform for learning programming skills"
        )
        
        result = validator.validate_business_logic(unrelated_draft)
        
        assert any("unrelated" in warning for warning in result.warnings)
    
    def test_when_too_many_evidence_links_then_adds_warning(self, validator):
        """Given draft with too many evidence links, when validating business logic, then adds warning."""
        many_links = [f"https://example{i}.com" for i in range(15)]  # More than 10
        many_links_draft = IdeaDraft(
            title="Valid title here",
            description="Valid description here",
            evidence_links=many_links
        )
        
        result = validator.validate_business_logic(many_links_draft)
        
        assert any("Too many evidence links" in warning for warning in result.warnings)
    
    def test_when_duplicate_evidence_links_then_adds_warning(self, validator):
        """Given draft with duplicate evidence links, when validating business logic, then adds warning."""
        duplicate_links = [
            "https://example.com",
            "https://test.com", 
            "https://example.com"  # Duplicate
        ]
        duplicate_draft = IdeaDraft(
            title="Valid title here",
            description="Valid description here",
            evidence_links=duplicate_links
        )
        
        result = validator.validate_business_logic(duplicate_draft)
        
        assert any("Duplicate evidence links" in warning for warning in result.warnings)


class TestIdeaValidator:
    """Test main validator orchestrating all validation checks."""
    
    @pytest.fixture
    def config(self) -> ValidationConfig:
        """Provide test configuration."""
        return ValidationConfig()
    
    @pytest.fixture
    def mock_sanitizer(self):
        """Provide mock InputSanitizer."""
        return Mock(spec=InputSanitizer)
    
    @pytest.fixture  
    def mock_content_validator(self):
        """Provide mock ContentValidator."""
        return Mock(spec=ContentValidator)
    
    @pytest.fixture
    def mock_business_validator(self):
        """Provide mock BusinessRuleValidator."""
        return Mock(spec=BusinessRuleValidator)
    
    @pytest.fixture
    def validator(self, config, mock_sanitizer, mock_content_validator, mock_business_validator):
        """Provide IdeaValidator with mocked dependencies."""
        validator = IdeaValidator(config)
        validator.sanitizer = mock_sanitizer
        validator.content_validator = mock_content_validator
        validator.business_validator = mock_business_validator
        return validator
    
    @pytest.fixture
    def sample_raw_data(self) -> Dict[str, Any]:
        """Provide sample raw data for testing."""
        return {
            "title": "AI-powered productivity tool",
            "description": "Revolutionary solution using AI to boost productivity",
            "category": "ai_ml"
        }
    
    def test_when_successful_validation_then_returns_draft_and_result(
        self, validator, mock_sanitizer, mock_content_validator, 
        mock_business_validator, sample_raw_data
    ):
        """Given valid data, when validating and sanitizing, then returns draft and valid result."""
        # Setup mocks
        sanitized_data = sample_raw_data.copy()
        mock_sanitizer.sanitize_idea_draft.return_value = sanitized_data
        
        valid_result = ValidationResult(is_valid=True)
        mock_business_validator.validate_required_fields.return_value = valid_result
        mock_business_validator.validate_field_lengths.return_value = valid_result
        mock_business_validator.validate_business_logic.return_value = valid_result
        mock_content_validator.validate_content_quality.return_value = valid_result
        
        # Execute
        draft, result = validator.validate_and_sanitize_draft(sample_raw_data)
        
        # Verify
        assert isinstance(draft, IdeaDraft)
        assert result.is_valid is True
        assert draft.title == sample_raw_data["title"]
        
        # Verify all validation steps were called
        mock_sanitizer.sanitize_idea_draft.assert_called_once_with(sample_raw_data)
        mock_business_validator.validate_required_fields.assert_called_once()
        mock_business_validator.validate_field_lengths.assert_called_once()
        mock_business_validator.validate_business_logic.assert_called_once()
        mock_content_validator.validate_content_quality.assert_called_once()
    
    def test_when_pydantic_validation_fails_then_returns_error(
        self, validator, mock_sanitizer, sample_raw_data
    ):
        """Given invalid data for Pydantic, when validating, then returns error and fallback draft."""
        # Setup mock to return invalid data
        invalid_data = {"title": "", "description": ""}  # Will fail Pydantic validation
        mock_sanitizer.sanitize_idea_draft.return_value = invalid_data
        
        # Execute
        draft, result = validator.validate_and_sanitize_draft(sample_raw_data)
        
        # Verify
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "Invalid data format" in result.errors[0]
        assert draft.title == "INVALID_DATA_FALLBACK"  # Valid fallback draft returned
    
    def test_when_business_validation_fails_then_accumulates_errors(
        self, validator, mock_sanitizer, mock_content_validator, 
        mock_business_validator, sample_raw_data
    ):
        """Given business validation failures, when validating, then accumulates all errors."""
        # Setup mocks
        sanitized_data = sample_raw_data.copy()
        mock_sanitizer.sanitize_idea_draft.return_value = sanitized_data
        
        # Business validation with errors
        error_result = ValidationResult(is_valid=False)
        error_result.add_error("Business rule violation")
        mock_business_validator.validate_required_fields.return_value = error_result
        mock_business_validator.validate_field_lengths.return_value = error_result
        mock_business_validator.validate_business_logic.return_value = error_result
        
        # Content validation passes
        valid_result = ValidationResult(is_valid=True)
        mock_content_validator.validate_content_quality.return_value = valid_result
        
        # Execute
        draft, result = validator.validate_and_sanitize_draft(sample_raw_data)
        
        # Verify
        assert result.is_valid is False
        assert len(result.errors) == 3  # Three business validation errors
        assert all("Business rule violation" in error for error in result.errors)
    
    def test_when_content_validation_fails_then_accumulates_errors(
        self, validator, mock_sanitizer, mock_content_validator, 
        mock_business_validator, sample_raw_data
    ):
        """Given content validation failures, when validating, then accumulates all errors."""
        # Setup mocks
        sanitized_data = sample_raw_data.copy()
        mock_sanitizer.sanitize_idea_draft.sanitized_data
        
        # Business validation passes
        valid_result = ValidationResult(is_valid=True)
        mock_business_validator.validate_required_fields.return_value = valid_result
        mock_business_validator.validate_field_lengths.return_value = valid_result
        mock_business_validator.validate_business_logic.return_value = valid_result
        
        # Content validation with errors
        content_error_result = ValidationResult(is_valid=False)
        content_error_result.add_error("Content violation")
        content_error_result.add_warning("Content warning")
        mock_content_validator.validate_content_quality.return_value = content_error_result
        
        # Execute
        draft, result = validator.validate_and_sanitize_draft(sample_raw_data)
        
        # Verify
        assert result.is_valid is False
        assert "Content violation" in result.errors
        assert "Content warning" in result.warnings
    
    def test_when_exception_occurs_then_handles_gracefully(
        self, validator, mock_sanitizer, sample_raw_data
    ):
        """Given exception during validation, when validating, then handles gracefully."""
        # Setup mock to raise exception
        mock_sanitizer.sanitize_idea_draft.side_effect = Exception("Sanitization error")
        
        # Execute
        draft, result = validator.validate_and_sanitize_draft(sample_raw_data)
        
        # Verify
        assert result.is_valid is False
        assert "Validation system error" in result.errors[0]
        assert draft.title == ""  # Empty draft returned


class TestCreateValidator:
    """Test validator factory function."""
    
    def test_when_no_config_then_creates_with_default_config(self):
        """Given no config, when creating validator, then uses default config."""
        with patch('pipeline.ingestion.validators.get_validation_config') as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            validator = create_validator()
            
            assert isinstance(validator, IdeaValidator)
            mock_get_config.assert_called_once()
    
    def test_when_config_provided_then_uses_provided_config(self):
        """Given config, when creating validator, then uses provided config."""
        config = ValidationConfig()
        
        validator = create_validator(config)
        
        assert isinstance(validator, IdeaValidator)
        assert validator.config is config