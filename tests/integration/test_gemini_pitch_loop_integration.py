"""
Integration tests for Gemini Pro investor review in the pitch loop.

Tests the end-to-end integration of Gemini Pro qualitative feedback
within the existing pitch loop workflow.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Mock the dependencies before importing the pitch loop
import sys
from unittest.mock import MagicMock

# Mock all the complex dependencies
sys.modules['core.alert_manager'] = MagicMock()
sys.modules['core.evidence_collector'] = MagicMock() 
sys.modules['core.deck_generator'] = MagicMock()
sys.modules['core.token_budget_sentinel'] = MagicMock()
sys.modules['core.evidence_summarizer'] = MagicMock()
sys.modules['core.bias_monitor'] = MagicMock()

class TestGeminiPitchLoopIntegration:
    """Integration tests for Gemini Pro in the pitch loop."""
    
    @pytest.fixture
    def mock_token_sentinel(self):
        """Mock token sentinel for budget checking."""
        sentinel = Mock()
        sentinel.check_usage.return_value = True  # Allow operations by default
        return sentinel
    
    @pytest.fixture
    def sample_vc_profile(self):
        """Sample VC profile for testing."""
        return {
            "persona": {
                "role": "Seed Stage VC Partner",
                "investment_focus": ["AI", "B2B SaaS"],
                "typical_check_size": "$500K - $2M"
            },
            "scoring_rubric": {
                "team": {"weight": 0.3, "criteria": ["Technical expertise"]},
                "market": {"weight": 0.4, "criteria": ["Market size", "TAM"]},
                "product": {"weight": 0.3, "criteria": ["Product-market fit"]}
            }
        }
    
    @pytest.fixture
    def sample_pitch_deck(self):
        """Sample pitch deck content."""
        return """
        # AI-Powered Startup Platform
        
        ## Problem
        90% of startups fail due to poor validation
        
        ## Solution  
        AI agents automate startup validation process
        
        ## Market
        $50B startup ecosystem, growing 15% annually
        
        ## Business Model
        SaaS subscription with usage-based pricing
        
        ## Team
        Experienced AI engineers from top tech companies
        
        ## Financials
        Projecting $1M ARR by end of year 2
        """
    
    @pytest.fixture
    def mock_state(self, sample_pitch_deck):
        """Mock GraphState for testing."""
        return {
            "deck_content": sample_pitch_deck,
            "idea_name": "AI Startup Platform",
            "idea_description": "AI-powered platform for startup validation",
            "total_tokens_consumed": 100,
            "token_budget_exceeded": False,
            "current_phase": "InvestorReview"
        }
    
    @pytest.mark.asyncio
    async def test_gemini_integration_successful(
        self, 
        mock_token_sentinel, 
        sample_vc_profile, 
        mock_state
    ):
        """Test successful Gemini Pro integration in investor review."""
        
        # Mock the scoring function
        with patch('core.investor_scorer.score_pitch_with_rubric') as mock_scorer:
            mock_scorer.return_value = (0.75, [
                "Strong technical team identified",
                "Large addressable market validated", 
                "Clear revenue model presented"
            ])
            
            # Mock the Gemini Pro function
            with patch('core.gemini_investor_reviewer.call_gemini_pro_investor_agent') as mock_gemini:
                mock_gemini.return_value = [
                    "AI Assessment: Impressive technical depth with strong market opportunity",
                    "Key Strengths: Experienced team; Validated market need; Scalable technology",
                    "Areas of Concern: Competitive landscape; Customer acquisition costs", 
                    "Investment Recommendation: strong interest",
                    "AI Confidence: 0.78",
                    "Key Due Diligence Questions: What is your defensible moat?; How will you scale customer acquisition?"
                ]
                
                # Mock token sentinel and other dependencies
                with patch('core.token_budget_sentinel.TokenBudgetSentinel') as mock_sentinel_class:
                    mock_sentinel_class.return_value = mock_token_sentinel
                    
                    with patch('core.investor_scorer.load_investor_profile') as mock_load_profile:
                        mock_load_profile.return_value = sample_vc_profile
                        
                        # Import and test the investor review function
                        from configs.langgraph.pitch_loop import investor_review_node
                        
                        result = investor_review_node(mock_state)
                        
                        # Verify the result structure
                        assert "investor_feedback" in result
                        assert "funding_score" in result
                        assert "current_phase" in result
                        assert "total_tokens_consumed" in result
                        
                        # Verify Gemini feedback was included
                        feedback_items = result["investor_feedback"]
                        assert len(feedback_items) > 3  # Original rubric + Gemini feedback
                        
                        # Check for Gemini-specific feedback
                        gemini_feedback_found = any(
                            "AI Assessment" in item or "AI Confidence" in item 
                            for item in feedback_items
                        )
                        assert gemini_feedback_found, "Gemini feedback not found in results"
                        
                        # Verify token usage was updated correctly
                        assert result["total_tokens_consumed"] > mock_state["total_tokens_consumed"]
                        
                        # Verify Gemini was called with correct parameters
                        mock_gemini.assert_called_once()
                        call_args = mock_gemini.call_args[0]
                        assert sample_vc_profile in call_args
                        assert 0.75 in call_args  # rubric score
    
    @pytest.mark.asyncio
    async def test_gemini_integration_budget_exceeded(
        self, 
        sample_vc_profile, 
        mock_state
    ):
        """Test Gemini integration when budget is exceeded."""
        
        # Mock token sentinel to reject Gemini usage
        mock_token_sentinel = Mock()
        mock_token_sentinel.check_usage.side_effect = lambda tokens, desc: tokens < 500
        
        with patch('core.investor_scorer.score_pitch_with_rubric') as mock_scorer:
            mock_scorer.return_value = (0.65, ["Basic feedback item"])
            
            with patch('core.token_budget_sentinel.TokenBudgetSentinel') as mock_sentinel_class:
                mock_sentinel_class.return_value = mock_token_sentinel
                
                with patch('core.investor_scorer.load_investor_profile') as mock_load_profile:
                    mock_load_profile.return_value = sample_vc_profile
                    
                    # Import and test with high token usage
                    high_token_state = {**mock_state, "total_tokens_consumed": 1000}
                    
                    from configs.langgraph.pitch_loop import investor_review_node
                    result = investor_review_node(high_token_state)
                    
                    # Verify budget limit message was added
                    feedback_items = result["investor_feedback"]
                    budget_message_found = any(
                        "budget limit" in item.lower() or "skipped due to budget" in item.lower()
                        for item in feedback_items
                    )
                    assert budget_message_found, "Budget limit message not found"
    
    @pytest.mark.asyncio
    async def test_gemini_integration_api_failure(
        self, 
        mock_token_sentinel, 
        sample_vc_profile, 
        mock_state
    ):
        """Test Gemini integration when API call fails."""
        
        with patch('core.investor_scorer.score_pitch_with_rubric') as mock_scorer:
            mock_scorer.return_value = (0.60, ["Rubric feedback"])
            
            # Mock Gemini to raise an exception
            with patch('core.gemini_investor_reviewer.call_gemini_pro_investor_agent') as mock_gemini:
                mock_gemini.side_effect = Exception("API Error")
                
                with patch('core.token_budget_sentinel.TokenBudgetSentinel') as mock_sentinel_class:
                    mock_sentinel_class.return_value = mock_token_sentinel
                    
                    with patch('core.investor_scorer.load_investor_profile') as mock_load_profile:
                        mock_load_profile.return_value = sample_vc_profile
                        
                        from configs.langgraph.pitch_loop import investor_review_node
                        result = investor_review_node(mock_state)
                        
                        # Verify error was handled gracefully
                        feedback_items = result["investor_feedback"]
                        error_message_found = any(
                            "unavailable" in item.lower() or "failed" in item.lower()
                            for item in feedback_items
                        )
                        assert error_message_found, "Error handling message not found"
                        
                        # Verify the function still returned a valid result
                        assert "funding_score" in result
                        assert result["funding_score"] == 0.60  # Original rubric score maintained
    
    def test_gemini_feedback_format_validation(self):
        """Test that Gemini feedback format is compatible with existing system."""
        
        # Mock a typical Gemini response
        sample_gemini_feedback = [
            "AI Assessment: Strong opportunity with clear market validation",
            "Key Strengths: Technical team; Market opportunity; Business model",
            "Areas of Concern: Competition; Customer acquisition",
            "Investment Recommendation: proceed",
            "AI Confidence: 0.73"
        ]
        
        # Verify each item is a string (required by existing system)
        for item in sample_gemini_feedback:
            assert isinstance(item, str)
            assert len(item) > 0
        
        # Verify expected patterns exist
        assert any("AI Assessment" in item for item in sample_gemini_feedback)
        assert any("Recommendation" in item for item in sample_gemini_feedback)
        assert any("Confidence" in item for item in sample_gemini_feedback)
    
    @pytest.mark.asyncio
    async def test_token_cost_calculation_accuracy(
        self, 
        mock_token_sentinel, 
        sample_vc_profile, 
        mock_state
    ):
        """Test that token costs are calculated accurately."""
        
        with patch('core.investor_scorer.score_pitch_with_rubric') as mock_scorer:
            mock_scorer.return_value = (0.70, ["Test feedback"])
            
            with patch('core.gemini_investor_reviewer.call_gemini_pro_investor_agent') as mock_gemini:
                mock_gemini.return_value = ["Gemini feedback item"]
                
                with patch('core.token_budget_sentinel.TokenBudgetSentinel') as mock_sentinel_class:
                    mock_sentinel_class.return_value = mock_token_sentinel
                    
                    with patch('core.investor_scorer.load_investor_profile') as mock_load_profile:
                        mock_load_profile.return_value = sample_vc_profile
                        
                        from configs.langgraph.pitch_loop import investor_review_node
                        result = investor_review_node(mock_state)
                        
                        # Verify token usage calculation
                        initial_tokens = mock_state["total_tokens_consumed"]
                        final_tokens = result["total_tokens_consumed"]
                        
                        # Should include both base investor review cost (250) and Gemini cost (300)
                        expected_increase = 250 + 300  # INVESTOR_REVIEW_COST + GEMINI_REVIEW_COST
                        actual_increase = final_tokens - initial_tokens
                        
                        assert actual_increase == expected_increase, (
                            f"Expected token increase of {expected_increase}, got {actual_increase}"
                        )


class TestGeminiErrorHandling:
    """Test error handling in Gemini integration."""
    
    def test_missing_google_ai_api_key(self):
        """Test behavior when Google AI API key is missing."""
        
        with patch.dict('os.environ', {}, clear=True):
            # Remove any existing API key
            from core.gemini_investor_reviewer import GeminiInvestorReviewer, ReviewConfiguration
            
            config = ReviewConfiguration()
            
            # Should initialize without error but log warning
            with patch('core.gemini_investor_reviewer.logging') as mock_logging:
                reviewer = GeminiInvestorReviewer(config)
                
                # Verify warning was logged
                assert mock_logging.getLogger.return_value.warning.called
    
    @pytest.mark.asyncio
    async def test_malformed_gemini_response(self):
        """Test handling of malformed responses from Gemini."""
        
        from core.gemini_investor_reviewer import GeminiInvestorReviewer, ReviewConfiguration
        
        config = ReviewConfiguration()
        reviewer = GeminiInvestorReviewer(config)
        
        # Mock a malformed response
        mock_response = Mock()
        mock_response.text = "Not valid JSON at all"
        
        with patch.object(reviewer, '_call_gemini_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response
            
            with pytest.raises(ValueError, match="Invalid JSON"):
                await reviewer.generate_qualitative_feedback(
                    pitch_deck_content="Test content",
                    investor_profile={"persona": {}, "scoring_rubric": {}},
                    rubric_score=0.5,
                    rubric_feedback=[]
                )
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of API timeouts."""
        
        from core.gemini_investor_reviewer import call_gemini_pro_investor_agent
        
        # Mock a timeout scenario
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.side_effect = asyncio.TimeoutError("Request timed out")
            
            with pytest.raises(asyncio.TimeoutError):
                await call_gemini_pro_investor_agent(
                    pitch_deck_content="Test content",
                    investor_profile={"persona": {}, "scoring_rubric": {}},
                    rubric_score=0.5,
                    rubric_feedback=[]
                )