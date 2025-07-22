# Gemini Pro Investor Review Integration

## Overview

This document describes the integration of Google's Gemini Pro model for qualitative investor feedback in the pitch deck evaluation system. The integration complements the existing rubric-based scoring with AI-powered, persona-driven analysis that mimics real investor thought processes.

## Features

### Core Capabilities

1. **Persona-Based Analysis**: Generates feedback tailored to specific investor profiles (Angel, Seed VC, Series A, etc.)
2. **Qualitative Insights**: Provides nuanced assessment beyond numerical scoring
3. **Due Diligence Questions**: Generates relevant questions investors would ask
4. **Structured Feedback**: Returns organized strengths, concerns, and recommendations
5. **Budget-Aware**: Integrates with existing token budget management
6. **Fault Tolerant**: Graceful degradation when AI services are unavailable

### Integration Points

- **Pitch Loop Integration**: Seamlessly integrated into `investor_review_node`
- **Token Budget Management**: Respects existing budget constraints
- **Rubric Compatibility**: Enhances rather than replaces rubric-based scoring
- **Error Handling**: Robust error handling maintains system stability

## Architecture

### Components

```
┌─────────────────────────────────────────┐
│             Pitch Loop                  │
│  ┌─────────────────────────────────────┐│
│  │     investor_review_node()          ││
│  │                                     ││
│  │  1. Rubric-based scoring           ││
│  │  2. Gemini Pro analysis ──────────┐││
│  │  3. Combine feedback              │││
│  │  4. Token budget update           │││
│  └─────────────────────────────────────┘││
└─────────────────────────────────────────┘│
                                          │
    ┌─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│      GeminiInvestorReviewer             │
│  ┌─────────────────────────────────────┐│
│  │  • Persona-based prompts           ││
│  │  • Structured response parsing     ││
│  │  • Token usage tracking           ││
│  │  • Safety settings               ││
│  │  • Error handling                ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### Data Flow

1. **Input Processing**: Pitch deck content and investor profile are processed
2. **Prompt Generation**: System generates investor persona-specific prompts
3. **API Call**: Async call to Gemini Pro with safety settings
4. **Response Parsing**: JSON response parsed into structured feedback
5. **Integration**: Feedback items added to existing rubric-based results
6. **Token Tracking**: Usage recorded for budget management

## Configuration

### Environment Setup

```bash
# Required environment variable
export GOOGLE_AI_API_KEY="your-gemini-api-key"
```

### Review Configuration

```python
from core.gemini_investor_reviewer import ReviewConfiguration

config = ReviewConfiguration(
    model_name="gemini-1.5-pro",     # Gemini model to use
    temperature=0.3,                  # Sampling temperature (0.0-1.0)
    max_tokens=1500,                 # Maximum response tokens
    top_p=0.8,                       # Nucleus sampling parameter
    top_k=40,                        # Top-k sampling parameter
    enable_safety_settings=True,     # Enable content safety filters
    timeout_seconds=30,              # API timeout
    max_retries=3                    # Retry attempts on failure
)
```

## Usage

### Direct Usage

```python
from core.gemini_investor_reviewer import GeminiInvestorReviewer, ReviewConfiguration

# Initialize reviewer
config = ReviewConfiguration()
reviewer = GeminiInvestorReviewer(config)

# Generate feedback
feedback = await reviewer.generate_qualitative_feedback(
    pitch_deck_content=pitch_deck_markdown,
    investor_profile=investor_profile_dict,
    rubric_score=0.73,
    rubric_feedback=["Strong team", "Good market opportunity"]
)

# Access structured feedback
print(f"Overall: {feedback.overall_impression}")
print(f"Strengths: {feedback.strengths}")
print(f"Concerns: {feedback.concerns}")
print(f"Questions: {feedback.questions}")
print(f"Recommendation: {feedback.recommendation}")
print(f"Confidence: {feedback.confidence_score}")
```

### Integration Function

```python
from core.gemini_investor_reviewer import call_gemini_pro_investor_agent

# Integration-friendly function for existing code
qualitative_feedback = await call_gemini_pro_investor_agent(
    pitch_deck_content=deck_content,
    investor_profile=vc_profile,
    rubric_score=final_score,
    rubric_feedback=feedback_items[:3]
)

# Returns list of feedback strings for easy integration
feedback_items.extend(qualitative_feedback)
```

### Automatic Integration

The integration happens automatically in the pitch loop when:
1. Token budget allows for Gemini analysis
2. Google AI API key is configured
3. No system errors prevent execution

## Investor Profile Format

The system works with existing investor profile YAML files:

```yaml
persona:
  role: "Seed Stage VC Partner"
  investment_focus: ["AI", "B2B SaaS", "Early Stage"]
  typical_check_size: "$250K - $1M"
  portfolio_companies: ["TechCorp", "AIStart", "DataFlow"]

scoring_rubric:
  team:
    weight: 0.3
    criteria: ["Technical expertise", "Domain experience"]
  market:
    weight: 0.25
    criteria: ["Market size", "Growth potential"]
  product:
    weight: 0.25
    criteria: ["Product-market fit", "Innovation"]
  business_model:
    weight: 0.2
    criteria: ["Revenue model", "Scalability"]
```

## Response Format

Gemini Pro returns structured feedback in this format:

```json
{
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
  "recommendation": "proceed",
  "confidence_score": 0.75,
  "comparable_companies": ["Startup1", "Startup2"],
  "market_insights": ["AI market growing 25% annually"],
  "next_steps": ["Conduct technical due diligence", "Validate customer references"]
}
```

## Performance and Cost

### Token Usage

- **Typical Usage**: 800-1200 tokens per analysis
- **Cost**: ~$0.30-0.45 per review (Gemini Pro pricing)
- **Budget Integration**: 300 token units reserved in budget system

### Performance Metrics

- **Response Time**: 2-5 seconds typical
- **Success Rate**: >95% with proper error handling
- **Cache Hit Rate**: N/A (each review is unique)

### Optimization

- Content truncation for very long pitch decks (>3000 chars)
- Async execution prevents blocking
- Circuit breaker pattern for reliability
- Retry logic for transient failures

## Error Handling

### Graceful Degradation

The system provides multiple fallback levels:

1. **API Failure**: Falls back to rubric-only scoring
2. **Budget Exceeded**: Skips AI analysis, notes in feedback
3. **Invalid Response**: Logs error, continues with rubric results
4. **Timeout**: Cancels request, provides fallback feedback

### Error Messages

Users receive informative messages about AI analysis status:

```
"Note: AI qualitative feedback skipped due to budget limits"
"Note: AI qualitative feedback unavailable (API Error)"
"AI Assessment: Unable to generate detailed review at this time"
```

## Security and Safety

### Content Safety

- Built-in safety settings filter inappropriate content
- Multiple safety categories: harassment, hate speech, dangerous content
- Configurable safety thresholds

### Data Privacy

- No pitch deck content is permanently stored by Google
- API calls are encrypted in transit
- Audit logging for compliance

### API Key Security

- Environment variable configuration
- No hardcoded keys in source code
- Google Cloud Secret Manager integration available

## Monitoring and Debugging

### Performance Metrics

```python
reviewer = GeminiInvestorReviewer(config)
stats = reviewer.get_performance_stats()

print(f"Reviews generated: {stats['reviews_generated']}")
print(f"Average generation time: {stats['average_generation_time']}ms")
print(f"Total tokens used: {stats['total_tokens_used']}")
print(f"API errors: {stats['api_errors']}")
```

### Debug Logging

```python
import logging

# Enable debug logging
logging.getLogger('core.gemini_investor_reviewer').setLevel(logging.DEBUG)

# Monitor API calls and responses
logging.getLogger('core.gemini_investor_reviewer').info(
    "Gemini API call completed in 2500ms, response length: 1200 chars"
)
```

### Common Issues

1. **"API Error"**: Check GOOGLE_AI_API_KEY environment variable
2. **"Invalid JSON"**: Gemini response format issue, automatic retry
3. **"Budget exceeded"**: Normal budget constraint, system continues
4. **"Timeout"**: Network/API latency, automatic retry with backoff

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
pytest tests/core/test_gemini_investor_reviewer.py -v
```

### Integration Tests

Test end-to-end integration:

```bash
pytest tests/integration/test_gemini_pitch_loop_integration.py -v
```

### Mock Testing

For development without API access:

```python
from unittest.mock import Mock, patch

with patch('core.gemini_investor_reviewer.genai.GenerativeModel') as mock_model:
    mock_response = Mock()
    mock_response.text = '{"overall_impression": "Test feedback"}'
    mock_model.return_value.generate_content.return_value = mock_response
    
    # Test your integration
```

## Future Enhancements

### Planned Improvements

1. **Multi-Model Support**: Support for Claude, GPT-4, and other models
2. **Streaming Responses**: Real-time feedback generation
3. **Custom Rubrics**: AI-generated scoring criteria
4. **Comparative Analysis**: Multi-startup comparison capability
5. **Learning Integration**: Feedback quality improvement over time

### Extension Points

- Custom prompt templates for different investor types
- Industry-specific evaluation criteria
- Integration with external data sources (Crunchbase, PitchBook)
- Multi-language support for international investors

## Support

### Documentation

- API Reference: `core/gemini_investor_reviewer.py`
- Integration Guide: This document
- Test Examples: `tests/core/test_gemini_investor_reviewer.py`

### Troubleshooting

1. Check environment variables and API key
2. Review debug logs for API call details
3. Verify investor profile format and content
4. Test with minimal pitch deck content first
5. Monitor token budget usage patterns