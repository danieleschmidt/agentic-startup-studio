"""
Advanced OpenTelemetry instrumentation for AI/ML workloads
Provides distributed tracing across multi-agent workflows
"""

import os
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.util.http import get_excluded_urls


class AIAgentTracer:
    """Advanced tracing for AI agents with business context"""
    
    def __init__(self, service_name: str = "agentic-startup-studio"):
        self.service_name = service_name
        self.tracer = None
        self.meter = None
        self._setup_tracing()
        self._setup_metrics()
        self._setup_auto_instrumentation()
    
    def _setup_tracing(self):
        """Configure OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": os.getenv("APP_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            "ai.framework": "langchain+langgraph",
            "ai.runtime": "python",
        })
        
        # Set up trace provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self.tracer = trace.get_tracer(__name__)
        
        # Configure exporters
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
            agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
        )
        
        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317"),
            insecure=True,
        )
        
        # Add span processors
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
        
        if os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
    
    def _setup_metrics(self):
        """Configure OpenTelemetry metrics"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": os.getenv("APP_VERSION", "1.0.0"),
        })
        
        metrics.set_meter_provider(MeterProvider(resource=resource))
        self.meter = metrics.get_meter(__name__)
        
        # Create AI-specific metrics
        self.ai_operation_counter = self.meter.create_counter(
            name="ai_operations_total",
            description="Total number of AI operations",
            unit="1"
        )
        
        self.ai_operation_duration = self.meter.create_histogram(
            name="ai_operation_duration_seconds",
            description="Duration of AI operations",
            unit="s"
        )
        
        self.ai_token_usage = self.meter.create_counter(
            name="ai_tokens_consumed_total",
            description="Total tokens consumed by AI operations",
            unit="1"
        )
        
        self.ai_cost_tracker = self.meter.create_counter(
            name="ai_cost_usd_total",
            description="Total cost of AI operations in USD",
            unit="1"
        )
    
    def _setup_auto_instrumentation(self):
        """Configure automatic instrumentation"""
        # Instrument HTTP libraries
        RequestsInstrumentor().instrument()
        URLLib3Instrumentor().instrument()
        
        # Instrument database libraries
        Psycopg2Instrumentor().instrument()
        RedisInstrumentor().instrument()
    
    def trace_ai_agent(
        self,
        agent_type: str,
        operation: str,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """Decorator for tracing AI agent operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                span_name = f"agent.{agent_type}.{operation}"
                
                with self.tracer.start_as_current_span(span_name) as span:
                    # Add standard attributes
                    span.set_attributes({
                        "ai.agent.type": agent_type,
                        "ai.operation.name": operation,
                        "ai.model.name": model_name or "unknown",
                        "code.function": func.__name__,
                        "code.namespace": func.__module__,
                    })
                    
                    # Add custom attributes
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"ai.param.{key}", value)
                    
                    # Add baggage context
                    baggage.set_baggage("agent.type", agent_type)
                    baggage.set_baggage("operation", operation)
                    
                    start_time = time.time()
                    
                    try:
                        # Execute the function
                        result = func(*args, **kwargs)
                        
                        # Add result attributes
                        if hasattr(result, 'usage'):
                            # OpenAI-style usage
                            usage = result.usage
                            span.set_attribute("ai.tokens.prompt", usage.prompt_tokens)
                            span.set_attribute("ai.tokens.completion", usage.completion_tokens)
                            span.set_attribute("ai.tokens.total", usage.total_tokens)
                            
                            # Track metrics
                            self.ai_token_usage.add(
                                usage.total_tokens,
                                {"agent_type": agent_type, "model": model_name or "unknown"}
                            )
                        
                        # Mark span as successful
                        span.set_status(Status(StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record exception details
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        
                        # Add error attributes
                        span.set_attributes({
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                        })
                        
                        raise
                    
                    finally:
                        # Record operation metrics
                        duration = time.time() - start_time
                        
                        self.ai_operation_counter.add(
                            1,
                            {
                                "agent_type": agent_type,
                                "operation": operation,
                                "status": "success" if span.status.status_code == StatusCode.OK else "error"
                            }
                        )
                        
                        self.ai_operation_duration.record(
                            duration,
                            {"agent_type": agent_type, "operation": operation}
                        )
                        
                        span.set_attribute("ai.operation.duration_ms", duration * 1000)
            
            return wrapper
        return decorator
    
    def trace_workflow_stage(self, stage_name: str, workflow_id: Optional[str] = None):
        """Decorator for tracing workflow stages"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                span_name = f"workflow.{stage_name}"
                
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attributes({
                        "workflow.stage": stage_name,
                        "workflow.id": workflow_id or "unknown",
                        "code.function": func.__name__,
                    })
                    
                    # Propagate workflow context
                    baggage.set_baggage("workflow.stage", stage_name)
                    if workflow_id:
                        baggage.set_baggage("workflow.id", workflow_id)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return wrapper
        return decorator
    
    @contextmanager
    def trace_business_operation(
        self,
        operation_name: str,
        business_context: Dict[str, Any] = None
    ):
        """Context manager for tracing business operations with rich context"""
        span_name = f"business.{operation_name}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            # Add business context
            if business_context:
                for key, value in business_context.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"business.{key}", value)
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def inject_trace_context(self, headers: Dict[str, str] = None) -> Dict[str, str]:
        """Inject trace context into headers for downstream services"""
        if headers is None:
            headers = {}
        
        inject(headers)
        return headers
    
    def extract_trace_context(self, headers: Dict[str, str]):
        """Extract trace context from incoming headers"""
        return extract(headers)


# Global tracer instance
tracer = AIAgentTracer()

# Convenience decorators
def trace_ai_agent(agent_type: str, operation: str, model_name: str = None):
    """Convenience decorator for AI agent tracing"""
    return tracer.trace_ai_agent(agent_type, operation, model_name)

def trace_workflow_stage(stage_name: str, workflow_id: str = None):
    """Convenience decorator for workflow stage tracing"""
    return tracer.trace_workflow_stage(stage_name, workflow_id)

def trace_business_operation(operation_name: str, business_context: Dict[str, Any] = None):
    """Convenience context manager for business operation tracing"""
    return tracer.trace_business_operation(operation_name, business_context)


# Example usage patterns
if __name__ == "__main__":
    # Example AI agent operation
    @trace_ai_agent("ceo", "pitch_generation", "gpt-4")
    def generate_pitch(idea: str, target_audience: str):
        # Simulate AI operation
        time.sleep(0.1)
        return {"pitch": "Great startup idea!", "confidence": 0.95}
    
    # Example workflow stage
    @trace_workflow_stage("idea_validation", "workflow-123")
    def validate_idea(idea: str):
        # Simulate validation workflow
        with trace_business_operation("market_research", {"idea_category": "fintech"}):
            time.sleep(0.2)
        return {"valid": True, "score": 0.8}
    
    # Example usage
    result = generate_pitch("AI-powered fintech", "VCs")
    validation = validate_idea("AI-powered fintech")