# Claude Code Max Plan Setup Guide

This guide explains how to enable and configure Claude Code Max Plan compute features in the Agentic Startup Studio system.

## Overview

Claude Code Max Plan provides enhanced compute capabilities including:
- Extended compute resources (CPU and memory allocation)
- Long-running compute operations
- Advanced code execution features
- Parallel processing capabilities

## Configuration

### 1. Environment Variables

Add the following environment variables to your `.env` file:

```bash
# Enable Claude Code integration
CLAUDE_CODE_ENABLED=true

# Set plan to 'max' for compute features
CLAUDE_CODE_PLAN=max

# Your Claude Code API key
CLAUDE_CODE_API_KEY=your_claude_code_api_key_here

# Model configuration
CLAUDE_CODE_MODEL=claude-opus-4-20250514
CLAUDE_CODE_MAX_TOKENS=4096

# Enable compute features (Max Plan only)
CLAUDE_CODE_COMPUTE_ENABLED=true
CLAUDE_CODE_COMPUTE_TIMEOUT=300        # Max execution time in seconds
CLAUDE_CODE_COMPUTE_MAX_MEMORY=8192    # Max memory in MB
CLAUDE_CODE_COMPUTE_MAX_CPU=4          # Max CPU cores
```

### 2. Available Plans

- **standard**: Basic Claude Code features
- **pro**: Enhanced features with higher limits
- **max**: Full compute capabilities with maximum resources

### 3. Resource Limits

The following resource limits can be configured for Max Plan:

| Resource | Min | Max | Default |
|----------|-----|-----|---------|
| CPU Cores | 1 | 16 | 4 |
| Memory (MB) | 512 | 32768 | 8192 |
| Timeout (seconds) | 10 | 3600 | 300 |

## Usage

### 1. Basic Setup

```python
from pipeline.config.settings import get_claude_code_config

# Get configuration
config = get_claude_code_config()

# Check if Max Plan is enabled
if config.is_max_plan():
    print("Claude Code Max Plan is active!")
    print(f"CPU cores: {config.compute_max_cpu}")
    print(f"Memory: {config.compute_max_memory} MB")
```

### 2. Using the Claude Code Service

```python
from pipeline.core.service_factory import create_service_container
from pipeline.services.claude_code_service import ComputeRequest

async def use_claude_compute():
    async with create_service_container() as container:
        claude_service = await container.claude_code_service()
        
        if claude_service and claude_service.is_compute_available():
            # Submit a compute request
            request = ComputeRequest(
                operation="code_analysis",
                parameters={"target": "my_code.py"},
                timeout=60,
                memory_limit=2048,
                cpu_limit=2
            )
            
            request_id = await claude_service.submit_compute_request(request)
            result = await claude_service.wait_for_compute_result(request_id)
            
            print(f"Status: {result.status}")
            print(f"Duration: {result.duration_seconds}s")
```

### 3. Parallel Compute Operations

```python
# Submit multiple compute tasks in parallel
requests = []
for i in range(5):
    req = ComputeRequest(
        operation=f"task_{i}",
        parameters={"task_id": i},
        timeout=30,
        memory_limit=1024,
        cpu_limit=1
    )
    req_id = await claude_service.submit_compute_request(req)
    requests.append(req_id)

# Wait for all results
results = await asyncio.gather(*[
    claude_service.wait_for_compute_result(req_id) 
    for req_id in requests
])
```

## Integration with Existing Services

The Claude Code service is automatically registered in the service factory when enabled. It integrates with:

- **Budget Sentinel**: Tracks compute costs as part of the overall budget
- **Cache Manager**: Caches compute results for efficiency
- **Service Registry**: Manages lifecycle and health checks

## Monitoring and Health Checks

The Claude Code service provides health monitoring:

```python
# Get service info
info = claude_service.get_service_info()
print(f"Service status: {info['status']}")
print(f"Plan: {info['plan']}")
print(f"Compute enabled: {info['compute_enabled']}")

# Check compute limits
limits = claude_service.get_compute_limits()
print(f"Available: {limits['available']}")
print(f"Max CPU: {limits['max_cpu_cores']} cores")
print(f"Max Memory: {limits['max_memory_mb']} MB")
```

## Example: Running the Demo

A complete demonstration is available:

```bash
python examples/claude_code_compute_demo.py
```

This demo shows:
1. Code analysis with compute resources
2. Code optimization tasks
3. Parallel compute operations
4. Resource usage tracking

## Troubleshooting

### Service Not Available

If the Claude Code service is not available:
1. Check `CLAUDE_CODE_ENABLED=true` is set
2. Verify `CLAUDE_CODE_API_KEY` is configured
3. Ensure `CLAUDE_CODE_PLAN=max` for compute features

### Compute Features Not Working

If compute features are not available:
1. Verify `CLAUDE_CODE_COMPUTE_ENABLED=true`
2. Check that plan is set to `max`
3. Ensure resource limits are within allowed ranges

### Resource Limit Errors

If you get resource limit errors:
1. Check requested resources don't exceed configured limits
2. Verify limits are within plan boundaries
3. Consider adjusting timeout or memory settings

## Security Considerations

1. **API Key Security**: Store API keys securely, never commit to version control
2. **Resource Limits**: Set appropriate limits to prevent resource exhaustion
3. **Timeout Configuration**: Use reasonable timeouts to prevent hanging operations
4. **Access Control**: Limit access to Claude Code features based on user roles

## Cost Management

Claude Code Max Plan usage is tracked by the Budget Sentinel service. Monitor costs through:
- Real-time cost tracking
- Budget alerts at configured thresholds
- Automatic shutdown on budget exhaustion

## Future Enhancements

Planned improvements include:
- GPU compute support
- Distributed compute operations
- Advanced caching strategies
- Real-time progress monitoring
- Custom compute environments