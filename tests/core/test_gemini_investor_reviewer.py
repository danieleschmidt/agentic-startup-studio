"""
Test suite for Gemini Pro-based investor review functionality.

Tests the integration of Google Gemini Pro for qualitative investor feedback
that complements the existing rubric-based scoring system.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from core.gemini_investor_reviewer import (
    GeminiInvestorReviewer,
    InvestorFeedback,
    ReviewCriteria,
    ReviewConfiguration
)


class TestGeminiInvestorReviewer:
    """Test cases for the Gemini Pro investor review integration."""
    
    @pytest.fixture
    def reviewer_config(self):
        """Create a test configuration for the reviewer."""
        return ReviewConfiguration(
            model_name="gemini-1.5-pro",
            temperature=0.3,
            max_tokens=1000,
            enable_safety_settings=True,
            use_streaming=False
        )
    
    @pytest.fixture
    def sample_pitch_deck(self):
        """Sample pitch deck content for testing."""
        return """
        # AI-Powered Startup Validation Platform
        
        ## Problem
        90% of startups fail due to lack of market validation
        
        ## Solution
        Automated startup idea validation using AI agents
        
        ## Market
        $50B startup ecosystem market
        
        ## Business Model
        SaaS subscription with per-validation pricing
        
        ## Team
        Experienced AI engineers and startup veterans
        
        ## Financials
        $1M ARR projected by year 2
        """
    
    @pytest.fixture
    def sample_investor_profile(self):
        """Sample investor profile for testing."""
        return {
            "persona": {
                "role": "Seed Stage VC Partner",
                "investment_focus": ["AI", "B2B SaaS", "Early Stage"],
                "typical_check_size": "$250K - $1M",
                "portfolio_companies": ["TechCorp", "AIStart", "DataFlow"]
            },
            "scoring_rubric": {
                "team": {"weight": 0.3, "criteria": ["Technical expertise", "Domain experience"]},
                "market": {"weight": 0.25, "criteria": ["Market size", "Growth potential"]},
                "product": {"weight": 0.25, "criteria": ["Product-market fit", "Innovation"]},
                "business_model": {"weight": 0.2, "criteria": ["Revenue model", "Scalability"]}
            }
        }
    
    @pytest.fixture
    def gemini_reviewer(self, reviewer_config):
        """Create a GeminiInvestorReviewer instance."""
        return GeminiInvestorReviewer(reviewer_config)
    
    @pytest.mark.asyncio
    async def test_reviewer_initialization(self, reviewer_config):
        """Test that the reviewer initializes correctly."""
        reviewer = GeminiInvestorReviewer(reviewer_config)
        
        assert reviewer.config == reviewer_config
        assert reviewer.model_name == "gemini-1.5-pro"
        assert reviewer.temperature == 0.3
        assert reviewer.max_tokens == 1000
    
    @pytest.mark.asyncio
    async def test_generate_qualitative_feedback(
        self, 
        gemini_reviewer, 
        sample_pitch_deck, 
        sample_investor_profile
    ):
        """Test generating qualitative feedback for a pitch deck."""
        # Mock the Gemini API response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "overall_impression": "Strong technical team with clear market opportunity",
            "strengths": [
                "Experienced team with relevant AI expertise",
                "Large addressable market with proven pain point",
                "Clear revenue model and monetization strategy"
            ],
            "concerns": [
                "Competitive landscape needs more analysis",
                "Customer acquisition strategy could be more detailed"
            ],
            "questions": [
                "What is your defensible moat against larger tech companies?",
                "How will you achieve the projected $1M ARR timeline?"
            ],
            "recommendation": "Proceed to due diligence",
            "confidence_score": 0.75
        })
        
        with patch.object(gemini_reviewer, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            feedback = await gemini_reviewer.generate_qualitative_feedback(
                pitch_deck_content=sample_pitch_deck,
                investor_profile=sample_investor_profile,
                rubric_score=0.73,
                rubric_feedback=["Strong technical execution", "Market opportunity validated"]
            )
            
            # Verify the feedback structure
            assert isinstance(feedback, InvestorFeedback)
            assert feedback.overall_impression == "Strong technical team with clear market opportunity"
            assert len(feedback.strengths) == 3
            assert len(feedback.concerns) == 2
            assert len(feedback.questions) == 2
            assert feedback.recommendation == "Proceed to due diligence"
            assert feedback.confidence_score == 0.75
            
            # Verify API was called with correct parameters
            mock_api.assert_called_once()
            call_args = mock_api.call_args[1]
            assert "pitch_deck_content" in call_args["context"]
            assert "investor_profile" in call_args["context"]
    
    @pytest.mark.asyncio
    async def test_prompt_generation(self, gemini_reviewer, sample_investor_profile):
        """Test that the system prompt is generated correctly."""
        prompt = gemini_reviewer._generate_system_prompt(sample_investor_profile)
        
        # Verify key elements are in the prompt
        assert "Seed Stage VC Partner" in prompt
        assert "AI" in prompt
        assert "B2B SaaS" in prompt
        assert "$250K - $1M" in prompt
        assert "Technical expertise" in prompt
        assert "Market size" in prompt
        
        # Verify prompt structure
        assert "You are an experienced" in prompt
        assert "Investment Focus" in prompt
        assert "Evaluation Criteria" in prompt
        assert "Response Format" in prompt
    
    @pytest.mark.asyncio
    async def test_context_preparation(
        self, 
        gemini_reviewer, 
        sample_pitch_deck, 
        sample_investor_profile
    ):
        """Test that context is prepared correctly for the API call."""
        context = gemini_reviewer._prepare_context(
            pitch_deck_content=sample_pitch_deck,
            investor_profile=sample_investor_profile,
            rubric_score=0.73,
            rubric_feedback=["Strong technical execution"]
        )
        
        assert "pitch_deck_content" in context
        assert "investor_persona" in context
        assert "rubric_score" in context
        assert "rubric_feedback" in context
        assert context["rubric_score"] == 0.73
        assert "Strong technical execution" in context["rubric_feedback"]
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, gemini_reviewer, sample_pitch_deck, sample_investor_profile):
        """Test that API errors are handled gracefully."""
        with patch.object(gemini_reviewer, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = Exception("API Error")
            
            with pytest.raises(Exception) as exc_info:
                await gemini_reviewer.generate_qualitative_feedback(
                    pitch_deck_content=sample_pitch_deck,
                    investor_profile=sample_investor_profile,
                    rubric_score=0.73,
                    rubric_feedback=[]
                )
            
            assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(
        self, 
        gemini_reviewer, 
        sample_pitch_deck, 
        sample_investor_profile
    ):
        """Test handling of invalid JSON responses from Gemini."""
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"
        
        with patch.object(gemini_reviewer, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            with pytest.raises(ValueError) as exc_info:
                await gemini_reviewer.generate_qualitative_feedback(
                    pitch_deck_content=sample_pitch_deck,
                    investor_profile=sample_investor_profile,
                    rubric_score=0.73,
                    rubric_feedback=[]
                )
            
            assert "Invalid JSON" in str(exc_info.value) or "parse" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_safety_settings_configuration(self, reviewer_config):
        """Test that safety settings are configured correctly."""
        reviewer_config.enable_safety_settings = True
        reviewer = GeminiInvestorReviewer(reviewer_config)
        
        safety_settings = reviewer._get_safety_settings()
        
        # Verify safety settings structure
        assert isinstance(safety_settings, list)
        assert len(safety_settings) > 0
        
        # Check that each setting has required fields
        for setting in safety_settings:
            assert "category" in setting
            assert "threshold" in setting
    
    @pytest.mark.asyncio
    async def test_token_usage_tracking(
        self, 
        gemini_reviewer, 
        sample_pitch_deck, 
        sample_investor_profile
    ):
        """Test that token usage is tracked correctly."""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "overall_impression": "Test impression",
            "strengths": ["Test strength"],
            "concerns": ["Test concern"],
            "questions": ["Test question"],
            "recommendation": "Test recommendation",
            "confidence_score": 0.8
        })
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 500
        mock_response.usage_metadata.candidates_token_count = 200
        mock_response.usage_metadata.total_token_count = 700
        
        with patch.object(gemini_reviewer, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            feedback = await gemini_reviewer.generate_qualitative_feedback(
                pitch_deck_content=sample_pitch_deck,
                investor_profile=sample_investor_profile,
                rubric_score=0.73,
                rubric_feedback=[]
            )
            
            # Verify token usage is tracked
            assert feedback.token_usage is not None
            assert feedback.token_usage["prompt_tokens"] == 500
            assert feedback.token_usage["completion_tokens"] == 200
            assert feedback.token_usage["total_tokens"] == 700
    
    @pytest.mark.asyncio
    async def test_multiple_investor_profiles(self, gemini_reviewer, sample_pitch_deck):
        """Test generating feedback for different investor profiles."""
        # Angel investor profile
        angel_profile = {
            "persona": {
                "role": "Angel Investor",
                "investment_focus": ["Pre-seed", "Consumer"],
                "typical_check_size": "$25K - $100K"
            },
            "scoring_rubric": {
                "founder": {"weight": 0.4, "criteria": ["Passion", "Execution ability"]},
                "product": {"weight": 0.35, "criteria": ["User traction", "Product quality"]},
                "market": {"weight": 0.25, "criteria": ["Market timing", "Competition"]}
            }
        }
        
        # Series A VC profile
        series_a_profile = {
            "persona": {
                "role": "Series A VC Partner",
                "investment_focus": ["B2B", "Enterprise", "Growth"],
                "typical_check_size": "$5M - $15M"
            },
            "scoring_rubric": {
                "metrics": {"weight": 0.3, "criteria": ["Revenue growth", "Unit economics"]},
                "scalability": {"weight": 0.3, "criteria": ["Market expansion", "Team scaling"]},
                "competitive_advantage": {"weight": 0.4, "criteria": ["Moat", "Defensibility"]}
            }
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "overall_impression": "Profile-specific impression",
            "strengths": ["Strength 1"],
            "concerns": ["Concern 1"],
            "questions": ["Question 1"],
            "recommendation": "Profile-specific recommendation",
            "confidence_score": 0.7
        })
        
        with patch.object(gemini_reviewer, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            # Test angel investor feedback
            angel_feedback = await gemini_reviewer.generate_qualitative_feedback(
                pitch_deck_content=sample_pitch_deck,
                investor_profile=angel_profile,
                rubric_score=0.7,
                rubric_feedback=[]
            )
            
            # Test Series A VC feedback
            series_a_feedback = await gemini_reviewer.generate_qualitative_feedback(
                pitch_deck_content=sample_pitch_deck,
                investor_profile=series_a_profile,
                rubric_score=0.8,
                rubric_feedback=[]
            )
            
            # Verify both calls succeeded
            assert angel_feedback.overall_impression is not None
            assert series_a_feedback.overall_impression is not None
            assert mock_api.call_count == 2
    
    def test_review_criteria_validation(self):
        """Test that review criteria are validated correctly."""
        # Valid criteria
        valid_criteria = ReviewCriteria(
            focus_areas=["team", "market", "product"],
            evaluation_depth="detailed",
            include_comparables=True,
            risk_tolerance="medium"
        )
        
        assert valid_criteria.focus_areas == ["team", "market", "product"]
        assert valid_criteria.evaluation_depth == "detailed"
        assert valid_criteria.include_comparables is True
        assert valid_criteria.risk_tolerance == "medium"
        
        # Test default values
        default_criteria = ReviewCriteria()
        assert len(default_criteria.focus_areas) > 0
        assert default_criteria.evaluation_depth is not None
    
    def test_configuration_validation(self):
        """Test that configuration parameters are validated."""
        # Valid configuration
        config = ReviewConfiguration(
            model_name="gemini-1.5-pro",
            temperature=0.3,
            max_tokens=1000
        )
        
        assert config.model_name == "gemini-1.5-pro"
        assert 0 <= config.temperature <= 1
        assert config.max_tokens > 0
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            ReviewConfiguration(temperature=1.5)
        
        # Test invalid max_tokens
        with pytest.raises(ValueError):
            ReviewConfiguration(max_tokens=-1)


class TestInvestorFeedbackIntegration:
    """Integration tests for the investor feedback system."""
    
    @pytest.mark.asyncio
    async def test_feedback_integration_with_existing_scorer(self):
        """Test that Gemini feedback integrates well with existing rubric scorer."""
        from core.investor_scorer import score_pitch_with_rubric
        
        # Mock data
        deck_content = "# Test Pitch Deck\n## Problem\nTest problem description"
        idea_details = {"idea_name": "Test Idea", "idea_description": "Test description"}
        rubric = {
            "team": {"weight": 0.4, "criteria": ["Experience"]},
            "market": {"weight": 0.6, "criteria": ["Size", "Growth"]}
        }
        
        # Get rubric-based score
        rubric_score, rubric_feedback = score_pitch_with_rubric(
            deck_content, idea_details, rubric
        )
        
        # Verify integration points
        assert isinstance(rubric_score, float)
        assert 0 <= rubric_score <= 1
        assert isinstance(rubric_feedback, list)
        assert len(rubric_feedback) > 0
        
        # These would be passed to the Gemini reviewer
        assert rubric_score is not None
        assert rubric_feedback is not None
    
    def test_feedback_serialization(self):
        """Test that feedback can be serialized and stored."""
        feedback = InvestorFeedback(
            overall_impression="Test impression",
            strengths=["Strength 1", "Strength 2"],
            concerns=["Concern 1"],
            questions=["Question 1", "Question 2"],
            recommendation="Proceed",
            confidence_score=0.8,
            token_usage={"total_tokens": 500}
        )
        
        # Test serialization
        feedback_dict = feedback.model_dump()
        assert feedback_dict["overall_impression"] == "Test impression"
        assert len(feedback_dict["strengths"]) == 2
        assert feedback_dict["confidence_score"] == 0.8
        
        # Test JSON serialization
        import json
        json_str = json.dumps(feedback_dict)
        assert "Test impression" in json_str
        
        # Test deserialization
        restored_dict = json.loads(json_str)
        restored_feedback = InvestorFeedback(**restored_dict)
        assert restored_feedback.overall_impression == feedback.overall_impression