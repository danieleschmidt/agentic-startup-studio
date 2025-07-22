"""
Claude Code Service - Integration with Claude Code Max Plan for compute capabilities.

Provides enhanced compute features for Claude Code Max Plan users including:
- Extended compute resources (CPU, memory)
- Long-running compute operations
- Advanced code execution capabilities
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from pipeline.config.settings import get_claude_code_config
from pipeline.core.service_registry import ServiceInterface


class ComputeStatus(Enum):
    """Status of compute operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ComputeRequest:
    """Request for compute operation."""
    operation: str
    parameters: Dict[str, Any]
    timeout: Optional[int] = None
    memory_limit: Optional[int] = None
    cpu_limit: Optional[int] = None


@dataclass
class ComputeResult:
    """Result of compute operation."""
    request_id: str
    status: ComputeStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    resources_used: Optional[Dict[str, Any]] = None


class ClaudeCodeService(ServiceInterface):
    """
    Service for integrating Claude Code Max Plan compute capabilities.
    
    This service provides access to enhanced compute features available
    in Claude Code Max Plan including extended resources and long-running
    operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_claude_code_config()
        self._client = None
        self._compute_queue: asyncio.Queue = asyncio.Queue()
        self._active_operations: Dict[str, ComputeResult] = {}
        
    async def initialize(self) -> None:
        """Initialize the Claude Code service."""
        if not self.config.enabled:
            self.logger.info("Claude Code service is disabled")
            return
            
        if not self.config.api_key:
            raise ValueError("Claude Code API key is required when service is enabled")
            
        # Initialize Claude Code client here
        # This would connect to the actual Claude Code API
        self.logger.info(f"Initialized Claude Code service with plan: {self.config.plan}")
        
        if self.config.is_max_plan():
            self.logger.info(
                f"Max Plan compute features enabled - "
                f"CPU: {self.config.compute_max_cpu} cores, "
                f"Memory: {self.config.compute_max_memory} MB, "
                f"Timeout: {self.config.compute_timeout}s"
            )
    
    async def shutdown(self) -> None:
        """Shutdown the Claude Code service gracefully."""
        # Cancel any pending operations
        while not self._compute_queue.empty():
            try:
                self._compute_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        # Wait for active operations to complete
        if self._active_operations:
            self.logger.info(f"Waiting for {len(self._active_operations)} active operations to complete")
            await asyncio.sleep(1)  # Give operations a chance to complete
            
        self.logger.info("Claude Code service shutdown complete")
    
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        if not self.config.enabled:
            return True  # Service is healthy when disabled
            
        # Check API connectivity
        # In a real implementation, this would ping the Claude Code API
        return True
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            'name': 'ClaudeCodeService',
            'version': '1.0.0',
            'status': 'ready' if self.config.enabled else 'disabled',
            'plan': self.config.plan,
            'compute_enabled': self.config.is_max_plan(),
            'limits': {
                'max_tokens': self.config.max_tokens,
                'compute_timeout': self.config.compute_timeout,
                'compute_memory': self.config.compute_max_memory,
                'compute_cpu': self.config.compute_max_cpu
            } if self.config.is_max_plan() else None
        }
    
    async def submit_compute_request(self, request: ComputeRequest) -> str:
        """
        Submit a compute request for processing.
        
        Args:
            request: The compute request to process
            
        Returns:
            Request ID for tracking the operation
            
        Raises:
            ValueError: If Max Plan features are not enabled
        """
        if not self.config.is_max_plan():
            raise ValueError("Compute features require Claude Code Max Plan")
            
        # Generate request ID
        request_id = f"compute_{datetime.utcnow().timestamp()}"
        
        # Apply limits from configuration
        if request.timeout is None:
            request.timeout = self.config.compute_timeout
        if request.memory_limit is None:
            request.memory_limit = self.config.compute_max_memory
        if request.cpu_limit is None:
            request.cpu_limit = self.config.compute_max_cpu
            
        # Validate limits don't exceed plan limits
        if request.timeout > self.config.compute_timeout:
            raise ValueError(f"Timeout {request.timeout}s exceeds plan limit {self.config.compute_timeout}s")
        if request.memory_limit > self.config.compute_max_memory:
            raise ValueError(f"Memory {request.memory_limit}MB exceeds plan limit {self.config.compute_max_memory}MB")
        if request.cpu_limit > self.config.compute_max_cpu:
            raise ValueError(f"CPU {request.cpu_limit} cores exceeds plan limit {self.config.compute_max_cpu} cores")
            
        # Create result tracker
        result = ComputeResult(
            request_id=request_id,
            status=ComputeStatus.PENDING
        )
        self._active_operations[request_id] = result
        
        # Queue the request for processing
        await self._compute_queue.put((request_id, request))
        
        # Start processing in background
        asyncio.create_task(self._process_compute_request(request_id, request))
        
        self.logger.info(f"Submitted compute request {request_id}: {request.operation}")
        return request_id
    
    async def get_compute_result(self, request_id: str) -> ComputeResult:
        """
        Get the result of a compute operation.
        
        Args:
            request_id: The ID of the compute request
            
        Returns:
            The compute result
            
        Raises:
            ValueError: If request ID is not found
        """
        if request_id not in self._active_operations:
            raise ValueError(f"Compute request {request_id} not found")
            
        return self._active_operations[request_id]
    
    async def wait_for_compute_result(
        self, 
        request_id: str, 
        poll_interval: float = 1.0
    ) -> ComputeResult:
        """
        Wait for a compute operation to complete.
        
        Args:
            request_id: The ID of the compute request
            poll_interval: How often to check for completion (seconds)
            
        Returns:
            The completed compute result
        """
        while True:
            result = await self.get_compute_result(request_id)
            if result.status in [ComputeStatus.COMPLETED, ComputeStatus.FAILED, ComputeStatus.TIMEOUT]:
                return result
            await asyncio.sleep(poll_interval)
    
    async def _process_compute_request(self, request_id: str, request: ComputeRequest) -> None:
        """Process a compute request asynchronously."""
        result = self._active_operations[request_id]
        start_time = datetime.utcnow()
        
        try:
            # Update status to running
            result.status = ComputeStatus.RUNNING
            
            # In a real implementation, this would:
            # 1. Send the request to Claude Code API
            # 2. Monitor resource usage
            # 3. Handle timeouts and resource limits
            # 4. Return the actual compute results
            
            # Simulate compute operation
            await asyncio.sleep(2)  # Simulate processing time
            
            # Mock result based on operation
            if request.operation == "code_analysis":
                result.result = {
                    "analysis": "Code analysis completed successfully",
                    "issues_found": 0,
                    "suggestions": []
                }
            elif request.operation == "optimization":
                result.result = {
                    "optimized": True,
                    "performance_gain": "15%",
                    "memory_saved": "200MB"
                }
            else:
                result.result = {"status": "completed", "operation": request.operation}
            
            result.status = ComputeStatus.COMPLETED
            result.resources_used = {
                "cpu_seconds": 2.5,
                "memory_peak_mb": 512,
                "network_bytes": 1024
            }
            
        except asyncio.TimeoutError:
            result.status = ComputeStatus.TIMEOUT
            result.error = f"Operation timed out after {request.timeout} seconds"
            
        except Exception as e:
            result.status = ComputeStatus.FAILED
            result.error = str(e)
            self.logger.error(f"Compute request {request_id} failed: {e}")
            
        finally:
            # Calculate duration
            end_time = datetime.utcnow()
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            self.logger.info(
                f"Compute request {request_id} completed with status: {result.status} "
                f"in {result.duration_seconds:.2f} seconds"
            )
    
    def is_compute_available(self) -> bool:
        """Check if compute features are available."""
        return self.config.enabled and self.config.is_max_plan()
    
    def get_compute_limits(self) -> Dict[str, Any]:
        """Get current compute limits based on plan."""
        if not self.config.is_max_plan():
            return {
                "available": False,
                "reason": "Compute features require Claude Code Max Plan"
            }
            
        return {
            "available": True,
            "timeout_seconds": self.config.compute_timeout,
            "max_memory_mb": self.config.compute_max_memory,
            "max_cpu_cores": self.config.compute_max_cpu,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens
        }


# Singleton instance getter
_claude_code_service = None


def get_claude_code_service() -> ClaudeCodeService:
    """Get the global Claude Code service instance."""
    global _claude_code_service
    if _claude_code_service is None:
        _claude_code_service = ClaudeCodeService()
    return _claude_code_service