# Sample Data Package

This directory contains comprehensive sample data for testing and demonstrating the Agentic Startup Studio data pipeline. The package includes diverse startup ideas, expected outputs, benchmark data, and demo scripts for various scenarios.

## Contents Overview

### Core Sample Data
- [`startup_ideas.json`](startup_ideas.json) - 10 diverse startup ideas across categories
- [`expected_outputs/`](expected_outputs/) - Pipeline outputs for each sample idea
- [`quality_scores.json`](quality_scores.json) - Quality scoring examples and thresholds

### Demo Scripts
- [`demo_basic_workflow.py`](demo_basic_workflow.py) - Basic pipeline demonstration
- [`demo_batch_processing.py`](demo_batch_processing.py) - Batch idea processing
- [`demo_similarity_detection.py`](demo_similarity_detection.py) - Duplicate detection showcase
- [`demo_quality_gates.py`](demo_quality_gates.py) - Quality gate validation examples

### Performance Data
- [`benchmarks/`](benchmarks/) - Performance benchmark results
- [`load_test_results.json`](load_test_results.json) - Load testing data
- [`cost_analysis.json`](cost_analysis.json) - Budget tracking examples

## Quick Start

### Running Basic Demo

```bash
# Set up environment
export DB_HOST=localhost
export DB_NAME=startup_studio_test
export EMBEDDING_API_KEY=your_test_key

# Run basic workflow demo
python docs/sample-data/demo_basic_workflow.py

# Run batch processing demo
python docs/sample-data/demo_batch_processing.py --ideas 5
```

### CLI Testing with Sample Data

```bash
# Create ideas from sample data
python -m pipeline.cli.ingestion_cli create \
    --title "AI-Powered Fitness Coach" \
    --description "Personalized workout and nutrition AI assistant..." \
    --category healthtech \
    --problem "Generic fitness apps lack personalization" \
    --solution "AI analyzes user data for custom recommendations" \
    --market "Health-conscious individuals and fitness enthusiasts"

# Test similarity detection
python -m pipeline.cli.ingestion_cli similar <idea_id> --limit 3

# Test quality gates
python -m pipeline.cli.ingestion_cli advance <idea_id> research
```

## Sample Data Categories

### Technology Sectors
- **AI/ML**: Intelligent automation and prediction systems
- **FinTech**: Financial technology innovations
- **HealthTech**: Healthcare and wellness solutions
- **SaaS**: Software-as-a-Service platforms
- **Consumer**: Direct-to-consumer applications

### Validation Scenarios
- **High Quality**: Well-defined problems with clear solutions
- **Edge Cases**: Boundary conditions and error scenarios  
- **Duplicates**: Similar ideas for testing deduplication
- **Low Quality**: Ideas requiring improvement or rejection

## Expected Performance Metrics

### Processing Times (Target)
- Idea validation: <2 seconds
- Similarity detection: <5 seconds
- Pipeline stage advancement: <10 seconds
- Batch processing (10 ideas): <30 seconds

### Quality Thresholds
- Research quality gate: ≥3 evidence sources, ≥0.7 quality score
- Deck generation: 10 slides, ≥90% accessibility score
- Investor evaluation: ≥0.8 investor score, ≥0.8 consensus

### Budget Constraints
- Per-idea processing: ≤$2.00
- Research phase: ≤$0.50
- Deck generation: ≤$0.30
- Full pipeline: ≤$1.20

## Testing Scenarios

### Positive Test Cases
1. **Complete Pipeline**: Idea progresses through all stages successfully
2. **High-Quality Idea**: Passes all quality gates on first attempt
3. **Batch Processing**: Multiple ideas processed efficiently
4. **Similarity Detection**: Accurately identifies duplicate content

### Negative Test Cases
1. **Validation Failures**: Invalid input data rejection
2. **Quality Gate Failures**: Ideas rejected at quality checkpoints
3. **Budget Limits**: Processing halted when costs exceed thresholds
4. **System Errors**: Graceful handling of database/API failures

### Edge Cases
1. **Minimum Content**: Ideas with minimal valid content
2. **Maximum Content**: Ideas at character limits
3. **Special Characters**: Unicode, emojis, and formatting
4. **Borderline Quality**: Ideas near quality thresholds

## Data Generation Notes

All sample data was created to:
- Represent realistic startup concepts across diverse industries
- Include both high-quality and low-quality examples
- Test edge cases and boundary conditions
- Maintain consistent formatting and structure
- Avoid any real proprietary information or sensitive data

## Usage Guidelines

### For Development
- Use sample ideas for unit and integration testing
- Benchmark performance against provided metrics
- Test error handling with edge case scenarios
- Validate quality gates with known good/bad examples

### For Demonstrations
- Start with high-quality ideas for successful showcases
- Use similarity detection to show duplicate prevention
- Demonstrate budget tracking with cost examples
- Show quality gates with both passing and failing cases

### For Load Testing
- Process multiple sample ideas concurrently
- Monitor system performance under various loads
- Test database connection pooling and resource management
- Validate budget enforcement under high usage

## Security Considerations

- No real API keys, passwords, or sensitive information included
- All URLs point to example domains or documentation
- Sample data suitable for public demonstrations
- Test environment isolation recommended

---

**Note**: This sample data package is designed for testing and demonstration purposes only. For production use, ensure proper environment configuration, security measures, and budget monitoring are in place.