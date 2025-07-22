#!/usr/bin/env python3
"""
Demonstration of Claude Code Max Plan compute capabilities.

This script shows how to use the enhanced compute features available
with Claude Code Max Plan including extended resources and long-running
operations.
"""

import asyncio
import logging
from datetime import datetime

from pipeline.core.service_factory import create_service_container
from pipeline.services.claude_code_service import ComputeRequest


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_compute_features():
    """Demonstrate Claude Code Max Plan compute capabilities."""
    
    async with create_service_container() as container:
        # Get the Claude Code service
        claude_service = await container.claude_code_service()
        
        if not claude_service:
            logger.error("Claude Code service is not enabled. Please configure CLAUDE_CODE_ENABLED=true")
            return
        
        # Check if compute features are available
        if not claude_service.is_compute_available():
            logger.error("Claude Code compute features require Max Plan. Please set CLAUDE_CODE_PLAN=max")
            return
        
        # Display compute limits
        limits = claude_service.get_compute_limits()
        logger.info("Claude Code Max Plan Compute Limits:")
        logger.info(f"  - Max CPU cores: {limits['max_cpu_cores']}")
        logger.info(f"  - Max memory: {limits['max_memory_mb']} MB")
        logger.info(f"  - Timeout: {limits['timeout_seconds']} seconds")
        logger.info(f"  - Model: {limits['model']}")
        logger.info(f"  - Max tokens: {limits['max_tokens']}")
        
        # Example 1: Code Analysis
        logger.info("\n--- Example 1: Code Analysis ---")
        analysis_request = ComputeRequest(
            operation="code_analysis",
            parameters={
                "repository": "/root/repo",
                "languages": ["python"],
                "analysis_type": "security"
            },
            timeout=60,
            memory_limit=2048,
            cpu_limit=2
        )
        
        request_id = await claude_service.submit_compute_request(analysis_request)
        logger.info(f"Submitted code analysis request: {request_id}")
        
        # Wait for result
        result = await claude_service.wait_for_compute_result(request_id)
        logger.info(f"Analysis completed with status: {result.status}")
        logger.info(f"Duration: {result.duration_seconds:.2f} seconds")
        if result.result:
            logger.info(f"Result: {result.result}")
        
        # Example 2: Code Optimization
        logger.info("\n--- Example 2: Code Optimization ---")
        optimization_request = ComputeRequest(
            operation="optimization",
            parameters={
                "target_file": "pipeline/services/budget_sentinel.py",
                "optimization_goals": ["performance", "memory"],
                "preserve_functionality": True
            },
            timeout=120,
            memory_limit=4096,
            cpu_limit=4
        )
        
        request_id = await claude_service.submit_compute_request(optimization_request)
        logger.info(f"Submitted optimization request: {request_id}")
        
        # Monitor progress
        while True:
            result = await claude_service.get_compute_result(request_id)
            if result.status.value in ["completed", "failed", "timeout"]:
                break
            logger.info(f"Status: {result.status.value}")
            await asyncio.sleep(1)
        
        logger.info(f"Optimization completed with status: {result.status}")
        if result.result:
            logger.info(f"Performance gain: {result.result.get('performance_gain')}")
            logger.info(f"Memory saved: {result.result.get('memory_saved')}")
        
        # Example 3: Parallel Compute Operations
        logger.info("\n--- Example 3: Parallel Compute Operations ---")
        
        # Submit multiple compute requests in parallel
        requests = []
        for i in range(3):
            req = ComputeRequest(
                operation=f"parallel_task_{i}",
                parameters={"task_id": i, "complexity": "medium"},
                timeout=30,
                memory_limit=1024,
                cpu_limit=1
            )
            req_id = await claude_service.submit_compute_request(req)
            requests.append(req_id)
            logger.info(f"Submitted parallel task {i}: {req_id}")
        
        # Wait for all results
        results = await asyncio.gather(*[
            claude_service.wait_for_compute_result(req_id) 
            for req_id in requests
        ])
        
        logger.info("\nParallel tasks completed:")
        for i, result in enumerate(results):
            logger.info(f"  Task {i}: {result.status.value} in {result.duration_seconds:.2f}s")
        
        # Display resource usage summary
        logger.info("\n--- Resource Usage Summary ---")
        total_cpu_seconds = sum(
            r.resources_used.get('cpu_seconds', 0) 
            for r in results 
            if r.resources_used
        )
        total_memory_peak = max(
            r.resources_used.get('memory_peak_mb', 0) 
            for r in results 
            if r.resources_used
        )
        logger.info(f"Total CPU seconds used: {total_cpu_seconds:.2f}")
        logger.info(f"Peak memory usage: {total_memory_peak} MB")


async def main():
    """Main entry point."""
    logger.info("Claude Code Max Plan Compute Demo")
    logger.info("=================================")
    
    try:
        await demonstrate_compute_features()
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())