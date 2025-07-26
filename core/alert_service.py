# core/alert_service.py
"""
Interface-compliant alert service implementation.
Wraps existing AlertManager to provide IAlertManager interface.
"""
from typing import List, Dict, Any, Optional
from core.interfaces import IAlertManager
from core.alert_manager import AlertManager


class AlertService(IAlertManager):
    """
    Service implementation of IAlertManager interface.
    Wraps the existing AlertManager for backward compatibility.
    """
    
    def __init__(self, log_file_path: Optional[str] = None):
        self._alert_manager = AlertManager(log_file_path)
        self._structured_alerts: List[Dict[str, Any]] = []
        
    def record_alert(self, message: str, severity: str = "warning") -> None:
        """Record an alert with specified severity"""
        # Use existing AlertManager for logging and file output
        self._alert_manager.record_alert(
            message=message, 
            level=severity.upper(),
            source="AlertService"
        )
        
        # Store structured format for interface compliance
        self._structured_alerts.append({
            "message": message,
            "severity": severity
        })
        
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded alerts in structured format"""
        return self._structured_alerts.copy()
        
    def clear_alerts(self) -> None:
        """Clear all alerts"""
        self._alert_manager.clear_logged_alerts()
        self._structured_alerts.clear()
        
    def get_logged_alerts(self) -> List[str]:
        """Get raw logged alerts (backward compatibility)"""
        return self._alert_manager.get_logged_alerts()
        
    def get_underlying_manager(self) -> AlertManager:
        """Get the underlying AlertManager instance"""
        return self._alert_manager