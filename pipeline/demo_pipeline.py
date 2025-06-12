"""
Pipeline Demo - Example usage and demonstration of the complete pipeline.

Shows how to use the main pipeline with sample data and complete execution examples.
"""

import logging
import asyncio
from pipeline.main_pipeline import get_main_pipeline
from pipeline.services.pitch_deck_generator import InvestorType


async def demo_pipeline_execution():
    """Demonstrate complete pipeline execution with sample data."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    pipeline = get_main_pipeline()
    
    # Sample startup idea
    startup_idea = """
    A mobile app that uses AI to help people reduce food waste by tracking expiration dates,
    suggesting recipes based on ingredients about to expire, and connecting users with local
    food banks for donation opportunities. The app would include a barcode scanner for easy
    inventory management and personalized meal planning based on dietary preferences.
    """
    
    try:
        # Execute complete pipeline
        result = await pipeline.execute_full_pipeline(
            startup_idea=startup_idea,
            target_investor=InvestorType.SEED,
            generate_mvp=True,
            max_total_budget=55.0
        )
        
        # Generate comprehensive report
        report = await pipeline.generate_pipeline_report(result)
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*80)
        print(f"Execution ID: {report['execution_summary']['execution_id']}")
        print(f"Phases Completed: {report['execution_summary']['phases_completed']}/4")
        print(f"Overall Success: {report['execution_summary']['overall_success']}")
        print(f"Quality Score: {report['quality_metrics']['overall_quality_score']:.2f}")
        print(f"Budget Utilization: {report['budget_tracking']['budget_utilization']:.1%}")
        print(f"Execution Time: {report['execution_summary']['execution_time_seconds']:.1f}s")
        
        if report['errors']:
            print(f"\nErrors: {len(report['errors'])}")
            for error in report['errors']:
                print(f"  - {error}")
        
        if report['recommendations']:
            print(f"\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "="*80)
        
        return result
        
    except Exception as e:
        print(f"Demo execution failed: {e}")
        raise


async def demo_quick_validation():
    """Quick demo of just the validation phase."""
    logging.basicConfig(level=logging.INFO)
    
    pipeline = get_main_pipeline()
    
    # Quick validation test
    quick_idea = "AI-powered fitness app for seniors"
    
    try:
        # Just run through Phase 1
        await pipeline._initialize_async_dependencies()
        
        from pipeline.main_pipeline import PipelineResult
        result = PipelineResult(startup_idea=quick_idea)
        
        await pipeline._execute_phase_1(quick_idea, result)
        
        print(f"\nQuick Validation Results:")
        print(f"Idea: {quick_idea}")
        print(f"Validation Score: {result.validation_result.get('overall_score', 0.0):.2f}")
        print(f"Valid: {result.validation_result.get('is_valid', False)}")
        
        return result
        
    except Exception as e:
        print(f"Quick validation failed: {e}")
        raise


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_pipeline_execution())