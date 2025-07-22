"""
Gemini Pro-based Investor Review System

This module provides qualitative investor feedback using Google's Gemini Pro model
to complement the existing rubric-based scoring system. It generates nuanced,
persona-specific feedback that mimics real investor thought processes.

Key features:
- Investor persona-based evaluation
- Structured qualitative feedback
- Integration with existing rubric scores
- Token usage tracking
- Safety settings for appropriate content
- Async operation for performance
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import google.generativeai as genai
from pydantic import BaseModel, Field, validator

from core.token_budget_sentinel import TokenBudgetSentinel


class EvaluationDepth(str, Enum):
    """Evaluation depth levels for investor review."""
    QUICK = "quick"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class RiskTolerance(str, Enum):
    """Risk tolerance levels for investor profiles."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ReviewConfiguration(BaseModel):
    """Configuration for the Gemini Pro investor reviewer."""
    
    model_name: str = "gemini-1.5-pro"
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1500, gt=0, le=8192)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=40, gt=0, le=100)
    
    # Safety and content settings
    enable_safety_settings: bool = True
    use_streaming: bool = False
    
    # Performance settings
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Temperature must be between 0.0 and 1.0')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError('max_tokens must be positive')
        return v


class ReviewCriteria(BaseModel):
    """Criteria for conducting investor review."""
    
    focus_areas: List[str] = Field(
        default=["team", "market", "product", "business_model", "financials", "competition"]
    )
    evaluation_depth: EvaluationDepth = EvaluationDepth.STANDARD
    include_comparables: bool = True
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    custom_questions: List[str] = Field(default_factory=list)


class InvestorFeedback(BaseModel):
    """Structured feedback from Gemini Pro investor review."""
    
    overall_impression: str = Field(description="High-level assessment of the opportunity")
    
    strengths: List[str] = Field(
        description="Key strengths identified in the pitch",
        default_factory=list
    )
    
    concerns: List[str] = Field(
        description="Areas of concern or weakness",
        default_factory=list
    )
    
    questions: List[str] = Field(
        description="Questions the investor would ask in due diligence",
        default_factory=list
    )
    
    recommendation: str = Field(
        description="Investment recommendation (pass, maybe, proceed, strong interest)"
    )
    
    confidence_score: float = Field(
        description="Confidence in the assessment (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Technical metadata
    token_usage: Optional[Dict[str, int]] = None
    generation_time_ms: Optional[int] = None
    model_version: Optional[str] = None
    
    # Additional insights
    comparable_companies: List[str] = Field(default_factory=list)
    market_insights: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class GeminiInvestorReviewer:
    """
    Gemini Pro-based investor reviewer for startup pitch decks.
    
    This class provides sophisticated, persona-driven investment analysis
    that complements traditional rubric-based scoring systems.
    """
    
    def __init__(self, config: ReviewConfiguration):
        """
        Initialize the Gemini investor reviewer.
        
        Args:
            config: Configuration for the reviewer
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract common config values
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        
        # Initialize token budget tracking
        self.token_sentinel = TokenBudgetSentinel()
        
        # Configure Gemini client
        self._configure_gemini()
        
        # Performance tracking
        self.stats = {
            'reviews_generated': 0,
            'total_tokens_used': 0,
            'average_generation_time': 0,
            'api_errors': 0
        }
    
    def _configure_gemini(self):
        """Configure the Gemini API client with appropriate settings."""
        try:
            # Configure Gemini with API key from environment or secrets
            # In production, this would use proper secrets management
            import os
            api_key = os.getenv('GOOGLE_AI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.logger.info("Gemini API configured successfully")
            else:
                self.logger.warning("GOOGLE_AI_API_KEY not found, using default configuration")
                
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API: {e}")
            raise
    
    def _get_safety_settings(self) -> List[Dict[str, Any]]:
        """Get safety settings for Gemini API calls."""
        if not self.config.enable_safety_settings:
            return []
        
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    
    def _generate_system_prompt(self, investor_profile: Dict[str, Any]) -> str:
        """
        Generate a system prompt based on the investor profile.
        
        Args:
            investor_profile: Dictionary containing investor persona and rubric
            
        Returns:
            Formatted system prompt for Gemini
        """
        persona = investor_profile.get('persona', {})
        role = persona.get('role', 'Investor')
        investment_focus = persona.get('investment_focus', [])
        check_size = persona.get('typical_check_size', 'Not specified')
        
        # Extract scoring criteria for context
        rubric = investor_profile.get('scoring_rubric', {})
        criteria_text = []
        for category, details in rubric.items():
            weight = details.get('weight', 0) * 100
            criteria = details.get('criteria', [])
            criteria_text.append(f"- {category.replace('_', ' ').title()} ({weight:.0f}%): {', '.join(criteria)}")
        
        prompt = f"""You are an experienced {role} conducting a thorough evaluation of a startup pitch deck.

Your Investment Profile:
- Role: {role}
- Investment Focus: {', '.join(investment_focus) if investment_focus else 'General'}
- Typical Check Size: {check_size}
- Portfolio Experience: {', '.join(persona.get('portfolio_companies', [])) if persona.get('portfolio_companies') else 'Diverse portfolio'}

Evaluation Criteria:
{chr(10).join(criteria_text) if criteria_text else "Standard investment criteria apply"}

Your task is to provide a comprehensive qualitative assessment that complements the quantitative rubric scoring. Focus on:

1. **Strategic Insights**: What makes this opportunity unique or concerning?
2. **Market Dynamics**: How does this fit in the current market landscape?
3. **Execution Risk**: What are the key risks and mitigation strategies?
4. **Due Diligence**: What questions would you ask in the next meeting?
5. **Investment Thesis**: Does this align with your investment strategy?

Approach this as you would a real pitch meeting - be thorough, insightful, and honest. Consider both the upside potential and downside risks.

Response Format:
Provide your response as a valid JSON object with the following structure:
{{
    "overall_impression": "Your high-level assessment in 1-2 sentences",
    "strengths": ["List of 3-5 key strengths you identify"],
    "concerns": ["List of 2-4 main concerns or weaknesses"],
    "questions": ["List of 3-5 due diligence questions you would ask"],
    "recommendation": "One of: 'pass', 'weak interest', 'interested', 'strong interest'",
    "confidence_score": 0.0-1.0,
    "comparable_companies": ["List of 2-3 comparable companies if applicable"],
    "market_insights": ["List of 1-3 relevant market observations"],
    "next_steps": ["List of 2-3 specific next steps if proceeding"]
}}

Ensure your response is a valid JSON object that can be parsed programmatically."""
        
        return prompt
    
    def _prepare_context(
        self,
        pitch_deck_content: str,
        investor_profile: Dict[str, Any],
        rubric_score: float,
        rubric_feedback: List[str],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare context for the Gemini API call.
        
        Args:
            pitch_deck_content: The pitch deck markdown content
            investor_profile: Investor persona and scoring rubric
            rubric_score: Score from the rubric-based evaluation
            rubric_feedback: Feedback items from rubric evaluation
            additional_context: Any additional context to include
            
        Returns:
            Structured context for the API call
        """
        context = {
            "pitch_deck_content": pitch_deck_content,
            "investor_persona": investor_profile.get('persona', {}),
            "scoring_rubric": investor_profile.get('scoring_rubric', {}),
            "rubric_score": rubric_score,
            "rubric_feedback": rubric_feedback
        }
        
        if additional_context:
            context.update(additional_context)
        
        return context
    
    def _format_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format the user prompt with the pitch deck and context.
        
        Args:
            context: Context dictionary with pitch deck and scoring info
            
        Returns:
            Formatted prompt for the user message
        """
        pitch_content = context.get('pitch_deck_content', '')
        rubric_score = context.get('rubric_score', 0)
        rubric_feedback = context.get('rubric_feedback', [])
        
        # Truncate pitch content if too long
        max_content_length = 3000
        if len(pitch_content) > max_content_length:
            pitch_content = pitch_content[:max_content_length] + "\n\n[Content truncated...]"
        
        prompt = f"""Please evaluate the following startup pitch deck:

PITCH DECK CONTENT:
---
{pitch_content}
---

QUANTITATIVE ANALYSIS CONTEXT:
- Rubric-based Score: {rubric_score:.2f}/1.00
- Key Rubric Feedback: {'; '.join(rubric_feedback[:5]) if rubric_feedback else 'No specific feedback'}

Please provide your qualitative investor assessment following the JSON format specified in the system prompt. Consider both the quantitative score and the pitch content in your evaluation."""
        
        return prompt
    
    async def _call_gemini_api(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        Make an async call to the Gemini API.
        
        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt with the pitch content
            context: Additional context for the call
            
        Returns:
            API response object
        """
        import time
        start_time = time.time()
        
        try:
            # Create the model with configuration
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "max_output_tokens": self.max_tokens,
                },
                safety_settings=self._get_safety_settings()
            )
            
            # Prepare the full prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate content
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt
            )
            
            # Track generation time
            generation_time = int((time.time() - start_time) * 1000)
            
            # Log the API call
            self.logger.info(
                f"Gemini API call completed in {generation_time}ms, "
                f"response length: {len(response.text) if response.text else 0} chars"
            )
            
            return response
            
        except Exception as e:
            self.stats['api_errors'] += 1
            self.logger.error(f"Gemini API call failed: {e}")
            raise
    
    async def generate_qualitative_feedback(
        self,
        pitch_deck_content: str,
        investor_profile: Dict[str, Any],
        rubric_score: float,
        rubric_feedback: List[str],
        criteria: Optional[ReviewCriteria] = None
    ) -> InvestorFeedback:
        """
        Generate qualitative investor feedback using Gemini Pro.
        
        Args:
            pitch_deck_content: The pitch deck markdown content
            investor_profile: Investor persona and scoring rubric
            rubric_score: Score from rubric-based evaluation (0.0-1.0)
            rubric_feedback: List of feedback from rubric evaluation
            criteria: Optional review criteria for customization
            
        Returns:
            Structured investor feedback
        """
        import time
        start_time = time.time()
        
        try:
            # Check token budget before proceeding
            estimated_tokens = len(pitch_deck_content) // 4 + 500  # Rough estimate
            if not self.token_sentinel.check_usage(estimated_tokens, "Gemini Investor Review"):
                raise Exception("Token budget exceeded for Gemini review")
            
            # Prepare context and prompts
            context = self._prepare_context(
                pitch_deck_content=pitch_deck_content,
                investor_profile=investor_profile,
                rubric_score=rubric_score,
                rubric_feedback=rubric_feedback
            )
            
            system_prompt = self._generate_system_prompt(investor_profile)
            user_prompt = self._format_user_prompt(context)
            
            # Call Gemini API
            response = await self._call_gemini_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context=context
            )
            
            # Parse the response
            feedback_data = self._parse_response(response)
            
            # Extract token usage if available
            token_usage = None
            if hasattr(response, 'usage_metadata'):
                token_usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
                self.stats['total_tokens_used'] += token_usage.get('total_tokens', 0)
            
            # Calculate generation time
            generation_time = int((time.time() - start_time) * 1000)
            
            # Create feedback object
            feedback = InvestorFeedback(
                **feedback_data,
                token_usage=token_usage,
                generation_time_ms=generation_time,
                model_version=self.model_name
            )
            
            # Update stats
            self.stats['reviews_generated'] += 1
            self.stats['average_generation_time'] = (
                (self.stats['average_generation_time'] * (self.stats['reviews_generated'] - 1) + generation_time)
                / self.stats['reviews_generated']
            )
            
            self.logger.info(
                f"Generated investor feedback in {generation_time}ms, "
                f"confidence: {feedback.confidence_score:.2f}"
            )
            
            return feedback
            
        except Exception as e:
            self.logger.error(f"Failed to generate qualitative feedback: {e}")
            raise
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """
        Parse the Gemini API response into structured feedback.
        
        Args:
            response: Raw response from Gemini API
            
        Returns:
            Parsed feedback dictionary
        """
        try:
            # Extract text from response
            if not response.text:
                raise ValueError("Empty response from Gemini API")
            
            response_text = response.text.strip()
            
            # Clean up response text (remove markdown code blocks if present)
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            feedback_data = json.loads(response_text)
            
            # Validate required fields and provide defaults
            required_defaults = {
                'overall_impression': 'Assessment pending',
                'strengths': [],
                'concerns': [],
                'questions': [],
                'recommendation': 'review needed',
                'confidence_score': 0.5,
                'comparable_companies': [],
                'market_insights': [],
                'next_steps': []
            }
            
            for key, default_value in required_defaults.items():
                if key not in feedback_data:
                    feedback_data[key] = default_value
            
            # Ensure confidence_score is valid
            if not 0 <= feedback_data.get('confidence_score', 0) <= 1:
                feedback_data['confidence_score'] = 0.5
            
            return feedback_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw response: {response.text[:500]}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the reviewer."""
        return {
            **self.stats,
            'config': {
                'model_name': self.model_name,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }


# Factory function for easy initialization
def create_gemini_reviewer(
    model_name: str = "gemini-1.5-pro",
    temperature: float = 0.3,
    max_tokens: int = 1500,
    **kwargs
) -> GeminiInvestorReviewer:
    """
    Create a Gemini investor reviewer with common defaults.
    
    Args:
        model_name: Gemini model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GeminiInvestorReviewer instance
    """
    config = ReviewConfiguration(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return GeminiInvestorReviewer(config)


# Integration function for existing pitch loop
async def call_gemini_pro_investor_agent(
    pitch_deck_content: str,
    investor_profile: Dict[str, Any],
    rubric_score: float,
    rubric_feedback: List[str],
    config: Optional[ReviewConfiguration] = None
) -> List[str]:
    """
    Integration function for the existing pitch loop.
    
    This function provides a bridge between the new Gemini Pro reviewer
    and the existing pitch loop code, maintaining backward compatibility.
    
    Args:
        pitch_deck_content: The pitch deck markdown content
        investor_profile: Investor persona and scoring rubric
        rubric_score: Score from rubric-based evaluation
        rubric_feedback: Feedback from rubric evaluation
        config: Optional configuration for the reviewer
        
    Returns:
        List of feedback strings for integration with existing code
    """
    try:
        # Create reviewer with default config if none provided
        if config is None:
            config = ReviewConfiguration()
        
        reviewer = GeminiInvestorReviewer(config)
        
        # Generate feedback
        feedback = await reviewer.generate_qualitative_feedback(
            pitch_deck_content=pitch_deck_content,
            investor_profile=investor_profile,
            rubric_score=rubric_score,
            rubric_feedback=rubric_feedback
        )
        
        # Convert to list format for existing code
        feedback_items = []
        
        # Add overall impression
        feedback_items.append(f"AI Assessment: {feedback.overall_impression}")
        
        # Add strengths
        if feedback.strengths:
            feedback_items.append("Key Strengths: " + "; ".join(feedback.strengths))
        
        # Add concerns
        if feedback.concerns:
            feedback_items.append("Areas of Concern: " + "; ".join(feedback.concerns))
        
        # Add recommendation
        feedback_items.append(f"Investment Recommendation: {feedback.recommendation}")
        
        # Add confidence
        feedback_items.append(f"AI Confidence: {feedback.confidence_score:.2f}")
        
        # Add key questions (first 2)
        if feedback.questions:
            key_questions = feedback.questions[:2]
            feedback_items.append("Key Due Diligence Questions: " + "; ".join(key_questions))
        
        return feedback_items
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Gemini investor review failed: {e}")
        # Return fallback feedback to maintain system stability
        return [
            "AI Assessment: Unable to generate detailed review at this time",
            f"Fallback Note: Relying on rubric score of {rubric_score:.2f}",
            "Recommendation: Manual review recommended"
        ]