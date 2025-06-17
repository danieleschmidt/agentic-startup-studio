"""Performance monitor with alerting capabilities."""

import asyncio


class PerformanceMonitor:
    """Real-time performance monitoring with alerting."""
    
    def __init__(
        self, 
        metrics_collector=None, 
        alert_manager=None, 
        cache_hit_rate_threshold: float = 0.8
    ):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.cache_hit_rate_threshold = cache_hit_rate_threshold
    
    async def check_cache_performance(self) -> None:
        """Check cache performance and trigger alerts if needed."""
        if not self.metrics_collector or not self.alert_manager:
            return
        
        hit_rate = self.metrics_collector.get_cache_hit_rate()
        
        if hit_rate < self.cache_hit_rate_threshold:
            self.alert_manager.trigger_cache_performance_alert()