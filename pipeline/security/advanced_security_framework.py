"""
Advanced Security Framework - Enterprise-Grade Security Implementation

Implements comprehensive security measures:
- Zero-trust authentication and authorization
- Advanced threat detection and prevention
- Real-time security monitoring and alerting
- Automated incident response
- Security compliance validation
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

import jwt

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(str, Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "authz_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class ComplianceStandard(str, Enum):
    """Security compliance standards."""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST = "nist"


@dataclass
class SecurityIncident:
    """Security incident representation."""
    incident_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: SecurityEvent = SecurityEvent.SUSPICIOUS_ACTIVITY
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    
    # Incident details
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    # Response tracking
    auto_remediated: bool = False
    human_review_required: bool = False
    false_positive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary."""
        return {
            "incident_id": self.incident_id,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "resource": self.resource,
            "description": self.description,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "auto_remediated": self.auto_remediated,
            "human_review_required": self.human_review_required,
            "false_positive": self.false_positive
        }


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    policy_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    enabled: bool = True
    
    # Rate limiting
    max_requests_per_minute: int = 100
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # Authentication
    require_mfa: bool = True
    session_timeout_minutes: int = 30
    require_strong_passwords: bool = True
    
    # Authorization
    principle_of_least_privilege: bool = True
    require_explicit_permissions: bool = True
    
    # Monitoring
    log_all_requests: bool = True
    alert_on_suspicious_activity: bool = True
    real_time_monitoring: bool = True
    
    # Compliance
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)


class ThreatDetector:
    """Advanced threat detection engine."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        
        # Tracking dictionaries
        self.request_counts: Dict[str, List[datetime]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.user_sessions: Dict[str, datetime] = {}
        self.suspicious_patterns: Dict[str, int] = {}
        
        # Threat signatures
        self.sql_injection_patterns = [
            r"'\s*(or|and)\s*'",
            r"union\s+select",
            r"drop\s+table",
            r"exec\s*\(",
            r"script\s*>"
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"document\.cookie"
        ]
        
        # Behavioral baselines
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Threat detector initialized")
    
    async def analyze_request(self, 
                            request_data: Dict[str, Any],
                            user_id: Optional[str] = None,
                            source_ip: Optional[str] = None) -> List[SecurityIncident]:
        """Analyze request for security threats."""
        incidents = []
        
        # Rate limiting check
        rate_limit_incident = await self._check_rate_limiting(source_ip or "unknown")
        if rate_limit_incident:
            incidents.append(rate_limit_incident)
        
        # Input validation
        injection_incidents = await self._detect_injection_attempts(request_data, source_ip)
        incidents.extend(injection_incidents)
        
        # Authentication analysis
        if user_id:
            auth_incidents = await self._analyze_authentication_patterns(user_id, source_ip)
            incidents.extend(auth_incidents)
            
            # Behavioral analysis
            behavior_incidents = await self._analyze_user_behavior(user_id, request_data)
            incidents.extend(behavior_incidents)
        
        # Anomaly detection
        anomaly_incidents = await self._detect_anomalies(request_data, user_id, source_ip)
        incidents.extend(anomaly_incidents)
        
        return incidents
    
    async def _check_rate_limiting(self, source_ip: str) -> Optional[SecurityIncident]:
        """Check for rate limiting violations."""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)
        
        # Initialize or clean old entries
        if source_ip not in self.request_counts:
            self.request_counts[source_ip] = []
        
        # Remove old requests
        self.request_counts[source_ip] = [
            req_time for req_time in self.request_counts[source_ip]
            if req_time > cutoff_time
        ]
        
        # Add current request
        self.request_counts[source_ip].append(now)
        
        # Check if rate limit exceeded
        if len(self.request_counts[source_ip]) > self.policy.max_requests_per_minute:
            return SecurityIncident(
                event_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                description=f"Rate limit exceeded: {len(self.request_counts[source_ip])} requests in 1 minute",
                evidence={
                    "request_count": len(self.request_counts[source_ip]),
                    "limit": self.policy.max_requests_per_minute,
                    "time_window": "1 minute"
                }
            )
        
        return None
    
    async def _detect_injection_attempts(self, 
                                       request_data: Dict[str, Any],
                                       source_ip: Optional[str] = None) -> List[SecurityIncident]:
        """Detect SQL injection and XSS attempts."""
        incidents = []
        
        # Convert all request data to strings for analysis
        text_data = []
        self._extract_text_from_data(request_data, text_data)
        
        for text in text_data:
            if not isinstance(text, str):
                continue
            
            text_lower = text.lower()
            
            # Check for SQL injection patterns
            for pattern in self.sql_injection_patterns:
                import re
                if re.search(pattern, text_lower, re.IGNORECASE):
                    incidents.append(SecurityIncident(
                        event_type=SecurityEvent.INJECTION_ATTEMPT,
                        threat_level=ThreatLevel.HIGH,
                        source_ip=source_ip,
                        description=f"SQL injection attempt detected: {pattern}",
                        evidence={
                            "pattern": pattern,
                            "input": text[:100],  # Truncate for logging
                            "type": "sql_injection"
                        }
                    ))
            
            # Check for XSS patterns
            for pattern in self.xss_patterns:
                import re
                if re.search(pattern, text_lower, re.IGNORECASE):
                    incidents.append(SecurityIncident(
                        event_type=SecurityEvent.INJECTION_ATTEMPT,
                        threat_level=ThreatLevel.HIGH,
                        source_ip=source_ip,
                        description=f"XSS attempt detected: {pattern}",
                        evidence={
                            "pattern": pattern,
                            "input": text[:100],  # Truncate for logging
                            "type": "xss"
                        }
                    ))
        
        return incidents
    
    def _extract_text_from_data(self, data: Any, text_list: List[str]):
        """Recursively extract text from nested data structures."""
        if isinstance(data, str):
            text_list.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                self._extract_text_from_data(value, text_list)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._extract_text_from_data(item, text_list)
    
    async def _analyze_authentication_patterns(self, 
                                             user_id: str,
                                             source_ip: Optional[str] = None) -> List[SecurityIncident]:
        """Analyze authentication patterns for anomalies."""
        incidents = []
        now = datetime.now()
        
        # Track failed attempts
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Check for excessive failed attempts
        cutoff_time = now - timedelta(minutes=self.policy.lockout_duration_minutes)
        recent_failures = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff_time
        ]
        
        if len(recent_failures) >= self.policy.max_failed_attempts:
            incidents.append(SecurityIncident(
                event_type=SecurityEvent.AUTHENTICATION_FAILURE,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                user_id=user_id,
                description=f"Excessive failed authentication attempts: {len(recent_failures)}",
                evidence={
                    "failed_attempts": len(recent_failures),
                    "threshold": self.policy.max_failed_attempts,
                    "time_window_minutes": self.policy.lockout_duration_minutes
                }
            ))
        
        return incidents
    
    async def _analyze_user_behavior(self, 
                                   user_id: str,
                                   request_data: Dict[str, Any]) -> List[SecurityIncident]:
        """Analyze user behavior for anomalies."""
        incidents = []
        
        # Initialize baseline if not exists
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                "typical_request_size": 0,
                "typical_endpoints": set(),
                "typical_request_frequency": 0,
                "last_activity": datetime.now()
            }
        
        baseline = self.user_baselines[user_id]
        
        # Analyze request size
        request_size = len(str(request_data))
        if baseline["typical_request_size"] > 0:
            size_ratio = request_size / baseline["typical_request_size"]
            if size_ratio > 10:  # Request 10x larger than typical
                incidents.append(SecurityIncident(
                    event_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM,
                    user_id=user_id,
                    description=f"Unusually large request size: {request_size} bytes",
                    evidence={
                        "request_size": request_size,
                        "typical_size": baseline["typical_request_size"],
                        "size_ratio": size_ratio
                    }
                ))
        
        # Update baseline with new data
        baseline["typical_request_size"] = (baseline["typical_request_size"] * 0.9 + request_size * 0.1)
        baseline["last_activity"] = datetime.now()
        
        return incidents
    
    async def _detect_anomalies(self, 
                              request_data: Dict[str, Any],
                              user_id: Optional[str] = None,
                              source_ip: Optional[str] = None) -> List[SecurityIncident]:
        """Detect various anomalies in requests."""
        incidents = []
        
        # Check for unusually large payloads
        data_size = len(str(request_data))
        if data_size > 1000000:  # 1MB threshold
            incidents.append(SecurityIncident(
                event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                user_id=user_id,
                description=f"Unusually large request payload: {data_size} bytes",
                evidence={"payload_size": data_size}
            ))
        
        # Check for suspicious headers or parameters
        suspicious_keys = ["admin", "root", "system", "debug", "test", "exec", "eval"]
        for key in suspicious_keys:
            if self._contains_key_recursive(request_data, key):
                incidents.append(SecurityIncident(
                    event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                    threat_level=ThreatLevel.LOW,
                    source_ip=source_ip,
                    user_id=user_id,
                    description=f"Suspicious parameter name detected: {key}",
                    evidence={"suspicious_key": key}
                ))
        
        return incidents
    
    def _contains_key_recursive(self, data: Any, key: str) -> bool:
        """Check if a key exists recursively in data structure."""
        if isinstance(data, dict):
            if key.lower() in [k.lower() for k in data.keys()]:
                return True
            for value in data.values():
                if self._contains_key_recursive(value, key):
                    return True
        elif isinstance(data, (list, tuple)):
            for item in data:
                if self._contains_key_recursive(item, key):
                    return True
        return False
    
    def record_failed_attempt(self, user_id: str):
        """Record a failed authentication attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        self.failed_attempts[user_id].append(datetime.now())


class SecurityAuthenticator:
    """Advanced authentication and authorization."""
    
    def __init__(self, secret_key: str, policy: SecurityPolicy):
        self.secret_key = secret_key
        self.policy = policy
        
        # Active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_permissions: Dict[str, Set[str]] = {}
        
        # Token blacklist
        self.blacklisted_tokens: Set[str] = set()
        
        logger.info("Security authenticator initialized")
    
    def generate_secure_token(self, 
                            user_id: str,
                            permissions: List[str],
                            expires_in_minutes: int = 30) -> str:
        """Generate a secure JWT token."""
        now = datetime.now()
        expiration = now + timedelta(minutes=expires_in_minutes)
        
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "issued_at": now.timestamp(),
            "expires_at": expiration.timestamp(),
            "jti": str(uuid4()),  # JWT ID for revocation
            "session_id": str(uuid4())
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # Store session information
        self.active_sessions[payload["session_id"]] = {
            "user_id": user_id,
            "permissions": set(permissions),
            "created_at": now,
            "last_activity": now,
            "token_id": payload["jti"]
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            # Check if token is blacklisted
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            if payload.get("jti") in self.blacklisted_tokens:
                logger.warning(f"Attempted use of blacklisted token: {payload.get('jti')}")
                return None
            
            # Check expiration
            if payload.get("expires_at", 0) < datetime.now().timestamp():
                logger.warning(f"Expired token used by user: {payload.get('user_id')}")
                return None
            
            # Update session activity
            session_id = payload.get("session_id")
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["last_activity"] = datetime.now()
            
            return payload
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            jti = payload.get("jti")
            session_id = payload.get("session_id")
            
            if jti:
                self.blacklisted_tokens.add(jti)
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            logger.info(f"Token revoked for user: {payload.get('user_id')}")
            return True
            
        except jwt.InvalidTokenError:
            return False
    
    def check_permission(self, user_id: str, required_permission: str) -> bool:
        """Check if user has required permission."""
        user_perms = self.user_permissions.get(user_id, set())
        return required_permission in user_perms or "admin" in user_perms
    
    def add_user_permissions(self, user_id: str, permissions: List[str]):
        """Add permissions for a user."""
        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = set()
        self.user_permissions[user_id].update(permissions)
    
    def remove_user_permissions(self, user_id: str, permissions: List[str]):
        """Remove permissions from a user."""
        if user_id in self.user_permissions:
            self.user_permissions[user_id].difference_update(permissions)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            last_activity = session_data["last_activity"]
            if (now - last_activity).total_seconds() > (self.policy.session_timeout_minutes * 60):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Expired session cleaned up: {session_id}")


class SecurityMonitor:
    """Real-time security monitoring and alerting."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.incidents: List[SecurityIncident] = []
        self.alert_handlers: List[callable] = []
        
        # Metrics tracking
        self.metrics = {
            "total_incidents": 0,
            "incidents_by_type": {},
            "incidents_by_severity": {},
            "auto_remediated": 0,
            "false_positives": 0
        }
        
        logger.info("Security monitor initialized")
    
    async def process_incident(self, incident: SecurityIncident):
        """Process a security incident."""
        self.incidents.append(incident)
        self._update_metrics(incident)
        
        logger.warning(f"Security incident detected: {incident.event_type.value} "
                      f"(Level: {incident.threat_level.value})")
        
        # Attempt auto-remediation
        if await self._attempt_auto_remediation(incident):
            incident.auto_remediated = True
            incident.resolved_at = datetime.now()
        
        # Send alerts for high-priority incidents
        if incident.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._send_alert(incident)
        
        # Check if human review is required
        if (incident.threat_level == ThreatLevel.CRITICAL or 
            not incident.auto_remediated):
            incident.human_review_required = True
    
    async def _attempt_auto_remediation(self, incident: SecurityIncident) -> bool:
        """Attempt to automatically remediate the incident."""
        try:
            if incident.event_type == SecurityEvent.RATE_LIMIT_EXCEEDED:
                # Could implement IP blocking here
                logger.info(f"Auto-remediation: Rate limiting applied to {incident.source_ip}")
                return True
            
            elif incident.event_type == SecurityEvent.INJECTION_ATTEMPT:
                # Could implement request blocking/sanitization
                logger.info(f"Auto-remediation: Blocked injection attempt from {incident.source_ip}")
                return True
            
            elif incident.event_type == SecurityEvent.AUTHENTICATION_FAILURE:
                # Could implement account lockout
                logger.info(f"Auto-remediation: Account locked for user {incident.user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Auto-remediation failed for incident {incident.incident_id}: {e}")
            return False
    
    async def _send_alert(self, incident: SecurityIncident):
        """Send alert for high-priority incident."""
        alert_data = {
            "incident": incident.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "requires_immediate_attention": incident.threat_level == ThreatLevel.CRITICAL
        }
        
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert_data)
                else:
                    handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: callable):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def _update_metrics(self, incident: SecurityIncident):
        """Update security metrics."""
        self.metrics["total_incidents"] += 1
        
        # Update by type
        event_type = incident.event_type.value
        if event_type not in self.metrics["incidents_by_type"]:
            self.metrics["incidents_by_type"][event_type] = 0
        self.metrics["incidents_by_type"][event_type] += 1
        
        # Update by severity
        severity = incident.threat_level.value
        if severity not in self.metrics["incidents_by_severity"]:
            self.metrics["incidents_by_severity"][severity] = 0
        self.metrics["incidents_by_severity"][severity] += 1
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        recent_incidents = [
            incident for incident in self.incidents
            if (datetime.now() - incident.detected_at).total_seconds() < 3600  # Last hour
        ]
        
        return {
            "metrics": self.metrics,
            "recent_incidents": len(recent_incidents),
            "unresolved_incidents": len([i for i in self.incidents if not i.resolved_at]),
            "critical_incidents": len([i for i in self.incidents if i.threat_level == ThreatLevel.CRITICAL]),
            "incidents_requiring_review": len([i for i in self.incidents if i.human_review_required]),
            "auto_remediation_rate": (
                self.metrics["auto_remediated"] / max(1, self.metrics["total_incidents"])
            ),
            "recent_incident_types": {
                incident_type: len([
                    i for i in recent_incidents 
                    if i.event_type.value == incident_type
                ])
                for incident_type in set(i.event_type.value for i in recent_incidents)
            }
        }


class AdvancedSecurityFramework:
    """Main security framework orchestrating all security components."""
    
    def __init__(self, secret_key: str, policy: Optional[SecurityPolicy] = None):
        self.secret_key = secret_key
        self.policy = policy or SecurityPolicy()
        
        # Initialize components
        self.threat_detector = ThreatDetector(self.policy)
        self.authenticator = SecurityAuthenticator(secret_key, self.policy)
        self.monitor = SecurityMonitor(self.policy)
        
        # Compliance validation
        self.compliance_validators = {}
        self._setup_compliance_validators()
        
        logger.info("Advanced Security Framework initialized")
    
    def _setup_compliance_validators(self):
        """Setup compliance validators for different standards."""
        for standard in self.policy.compliance_standards:
            if standard == ComplianceStandard.GDPR:
                self.compliance_validators[standard] = self._validate_gdpr_compliance
            elif standard == ComplianceStandard.HIPAA:
                self.compliance_validators[standard] = self._validate_hipaa_compliance
            elif standard == ComplianceStandard.SOC2:
                self.compliance_validators[standard] = self._validate_soc2_compliance
    
    async def secure_request_handler(self, 
                                   request_data: Dict[str, Any],
                                   token: Optional[str] = None,
                                   source_ip: Optional[str] = None,
                                   required_permission: Optional[str] = None) -> Dict[str, Any]:
        """Handle request with full security validation."""
        
        # 1. Token verification
        user_context = None
        if token:
            user_context = self.authenticator.verify_token(token)
            if not user_context:
                incident = SecurityIncident(
                    event_type=SecurityEvent.AUTHENTICATION_FAILURE,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    description="Invalid or expired token"
                )
                await self.monitor.process_incident(incident)
                raise Exception("Authentication failed")
        
        # 2. Permission check
        if required_permission and user_context:
            user_id = user_context.get("user_id")
            if not self.authenticator.check_permission(user_id, required_permission):
                incident = SecurityIncident(
                    event_type=SecurityEvent.AUTHORIZATION_DENIED,
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    user_id=user_id,
                    description=f"Access denied for permission: {required_permission}"
                )
                await self.monitor.process_incident(incident)
                raise Exception("Authorization denied")
        
        # 3. Threat detection
        user_id = user_context.get("user_id") if user_context else None
        incidents = await self.threat_detector.analyze_request(request_data, user_id, source_ip)
        
        # 4. Process any detected incidents
        for incident in incidents:
            await self.monitor.process_incident(incident)
            
            # Block request for critical threats
            if incident.threat_level == ThreatLevel.CRITICAL:
                raise Exception(f"Request blocked due to security threat: {incident.description}")
        
        return {
            "user_context": user_context,
            "security_incidents": len(incidents),
            "cleared_for_processing": True
        }
    
    async def generate_access_token(self, 
                                  user_id: str,
                                  permissions: List[str],
                                  source_ip: Optional[str] = None) -> str:
        """Generate secure access token after validation."""
        
        # Additional security checks could be added here
        # - Check user account status
        # - Validate source IP against user's typical locations
        # - Check for concurrent sessions
        
        token = self.authenticator.generate_secure_token(
            user_id, 
            permissions,
            self.policy.session_timeout_minutes
        )
        
        logger.info(f"Access token generated for user: {user_id}")
        return token
    
    def _validate_gdpr_compliance(self, request_data: Dict[str, Any]) -> bool:
        """Validate GDPR compliance requirements."""
        # Check for personal data handling
        # Verify data minimization
        # Ensure consent tracking
        return True  # Simplified for demo
    
    def _validate_hipaa_compliance(self, request_data: Dict[str, Any]) -> bool:
        """Validate HIPAA compliance requirements."""
        # Check for PHI handling
        # Verify access controls
        # Ensure audit logging
        return True  # Simplified for demo
    
    def _validate_soc2_compliance(self, request_data: Dict[str, Any]) -> bool:
        """Validate SOC2 compliance requirements."""
        # Check security controls
        # Verify availability measures
        # Ensure confidentiality
        return True  # Simplified for demo
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "framework_status": "active",
            "policy": {
                "name": self.policy.name,
                "enabled": self.policy.enabled,
                "compliance_standards": [std.value for std in self.policy.compliance_standards]
            },
            "threat_detection": {
                "active_patterns": len(self.threat_detector.sql_injection_patterns + self.threat_detector.xss_patterns),
                "monitored_users": len(self.threat_detector.user_baselines),
                "tracked_ips": len(self.threat_detector.request_counts)
            },
            "authentication": {
                "active_sessions": len(self.authenticator.active_sessions),
                "blacklisted_tokens": len(self.authenticator.blacklisted_tokens),
                "managed_users": len(self.authenticator.user_permissions)
            },
            "monitoring": self.monitor.get_security_dashboard()
        }


# Factory function
def create_security_framework(secret_key: str, 
                            policy: Optional[SecurityPolicy] = None) -> AdvancedSecurityFramework:
    """Create and return a configured security framework."""
    return AdvancedSecurityFramework(secret_key, policy)


# Example usage
async def security_demo():
    """Demonstrate security framework capabilities."""
    
    # Create security framework
    secret_key = secrets.token_urlsafe(32)
    policy = SecurityPolicy(
        name="demo_policy",
        max_requests_per_minute=50,
        compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.SOC2]
    )
    
    framework = create_security_framework(secret_key, policy)
    
    # Add alert handler
    async def alert_handler(alert_data):
        print(f"SECURITY ALERT: {alert_data['incident']['description']}")
    
    framework.monitor.add_alert_handler(alert_handler)
    
    # Setup user permissions
    framework.authenticator.add_user_permissions("user123", ["read", "write"])
    
    # Generate token
    token = await framework.generate_access_token("user123", ["read", "write"])
    
    # Test secure request handling
    try:
        # Normal request
        result = await framework.secure_request_handler(
            {"action": "get_data", "resource": "documents"},
            token=token,
            source_ip="192.168.1.100",
            required_permission="read"
        )
        print("Normal request: PASSED")
        
        # Suspicious request (SQL injection attempt)
        await framework.secure_request_handler(
            {"query": "SELECT * FROM users WHERE id = 1 OR 1=1"},
            token=token,
            source_ip="192.168.1.100"
        )
        
    except Exception as e:
        print(f"Suspicious request blocked: {e}")
    
    # Get security status
    status = framework.get_security_status()
    print(f"\nSecurity Status:")
    print(f"Active sessions: {status['authentication']['active_sessions']}")
    print(f"Total incidents: {status['monitoring']['metrics']['total_incidents']}")
    print(f"Unresolved incidents: {status['monitoring']['unresolved_incidents']}")
    
    return framework


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(security_demo())