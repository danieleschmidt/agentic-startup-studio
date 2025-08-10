"""
Enhanced Security Module - Advanced threat detection and response
Implements zero-trust security, runtime protection, and adaptive defense mechanisms.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import jwt
import re

from pydantic import BaseModel, Field, validator
from cryptography.fernet import Fernet
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer
from ..core.adaptive_intelligence import get_intelligence, PatternType

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(str, Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    INJECTION_ATTEMPT = "injection_attempt"
    BRUTE_FORCE = "brute_force"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_INPUT = "malicious_input"


class SecurityAction(str, Enum):
    """Security response actions"""
    LOG_ONLY = "log_only"
    RATE_LIMIT = "rate_limit"
    BLOCK_IP = "block_ip"
    REQUIRE_MFA = "require_mfa"
    ISOLATE_SESSION = "isolate_session"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class SecurityEvent:
    """Security event with metadata"""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "related_events": self.related_events
        }


@dataclass
class SecurityRule:
    """Security detection and response rule"""
    rule_id: str
    name: str
    description: str
    pattern: str
    threat_level: ThreatLevel
    action: SecurityAction
    enabled: bool = True
    false_positive_rate: float = 0.0
    detections: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class InputSanitizer:
    """Advanced input sanitization and validation"""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|\/\*|\*\/|'|\")",
        r"(\bOR\b.*=.*\b)",
        r"(1\s*=\s*1|1\s*=\s*'1')",
        r"(\bAND\b.*1\s*=\s*1)"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"(cat|ls|pwd|whoami|id|uname)",
        r"(curl|wget|nc|telnet)"
    ]
    
    def sanitize_input(self, input_data: Any, context: str = "general") -> Any:
        """Sanitize input based on context"""
        if isinstance(input_data, str):
            return self._sanitize_string(input_data, context)
        elif isinstance(input_data, dict):
            return {k: self.sanitize_input(v, context) for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item, context) for item in input_data]
        else:
            return input_data
    
    def _sanitize_string(self, text: str, context: str) -> str:
        """Sanitize string input"""
        # Check for malicious patterns
        threats = self._detect_threats(text)
        if threats:
            logger.warning(f"Malicious input detected: {threats}")
            # Replace with safe alternatives or reject
            text = self._neutralize_threats(text, threats)
        
        # Context-specific sanitization
        if context == "sql":
            text = self._escape_sql(text)
        elif context == "html":
            text = self._escape_html(text)
        elif context == "javascript":
            text = self._escape_javascript(text)
        
        return text
    
    def _detect_threats(self, text: str) -> List[str]:
        """Detect potential security threats in text"""
        threats = []
        
        # SQL injection detection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("sql_injection")
                break
        
        # XSS detection
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append("xss")
                break
        
        # Command injection detection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text):
                threats.append("command_injection")
                break
        
        return threats
    
    def _neutralize_threats(self, text: str, threats: List[str]) -> str:
        """Neutralize detected threats"""
        for threat in threats:
            if threat == "sql_injection":
                text = re.sub(r"[';\"\\]", "", text)
            elif threat == "xss":
                text = re.sub(r"[<>\"']", "", text)
            elif threat == "command_injection":
                text = re.sub(r"[;&|`$()]", "", text)
        
        return text
    
    def _escape_sql(self, text: str) -> str:
        """Escape SQL special characters"""
        return text.replace("'", "''").replace("\\", "\\\\")
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    def _escape_javascript(self, text: str) -> str:
        """Escape JavaScript special characters"""
        return (text.replace("\\", "\\\\")
                   .replace("'", "\\'")
                   .replace('"', '\\"')
                   .replace("\n", "\\n")
                   .replace("\r", "\\r"))


class ThreatDetector:
    """Advanced threat detection using machine learning and heuristics"""
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_patterns: Dict[str, int] = {}
        self.baseline_metrics: Dict[str, float] = {}
        
    def analyze_request(
        self, 
        source_ip: str, 
        user_id: Optional[str], 
        request_data: Dict[str, Any]
    ) -> Optional[SecurityEvent]:
        """Analyze request for security threats"""
        with tracer.start_as_current_span("analyze_request") as span:
            span.set_attributes({
                "source_ip": source_ip,
                "user_id": user_id or "anonymous"
            })
            
            threats = []
            
            # Check for brute force attempts
            if self._detect_brute_force(source_ip):
                threats.append(("brute_force", ThreatLevel.HIGH))
            
            # Check for injection attempts
            injection_threat = self._detect_injection_attempts(request_data)
            if injection_threat:
                threats.append((injection_threat, ThreatLevel.CRITICAL))
            
            # Check for anomalous behavior
            if self._detect_anomalous_behavior(source_ip, user_id, request_data):
                threats.append(("anomalous_behavior", ThreatLevel.MEDIUM))
            
            # Check rate limiting violations
            if self._check_rate_limits(source_ip, user_id):
                threats.append(("rate_limit_violation", ThreatLevel.MEDIUM))
            
            # Return highest severity threat
            if threats:
                threats.sort(key=lambda x: self._threat_priority(x[1]), reverse=True)
                threat_type, threat_level = threats[0]
                
                event = SecurityEvent(
                    event_id=f"threat_{int(time.time())}_{secrets.token_hex(4)}",
                    event_type=SecurityEventType(threat_type) if threat_type in [e.value for e in SecurityEventType] else SecurityEventType.ANOMALOUS_BEHAVIOR,
                    threat_level=threat_level,
                    source_ip=source_ip,
                    user_id=user_id,
                    details={"request_data": request_data, "all_threats": threats}
                )
                
                return event
            
            return None
    
    def _detect_brute_force(self, source_ip: str) -> bool:
        """Detect brute force attempts"""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=5)
        
        if source_ip not in self.failed_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[source_ip]
            if attempt > cutoff
        ]
        
        return len(recent_attempts) > 10
    
    def _detect_injection_attempts(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Detect various injection attempts"""
        data_str = json.dumps(request_data, default=str).lower()
        
        # SQL injection
        sql_patterns = ["union select", "drop table", "insert into", "delete from", "' or 1=1"]
        if any(pattern in data_str for pattern in sql_patterns):
            return "injection_attempt"
        
        # Command injection
        cmd_patterns = ["; cat /", "&& whoami", "| ls -la", "$(curl"]
        if any(pattern in data_str for pattern in cmd_patterns):
            return "injection_attempt"
        
        # LDAP injection
        ldap_patterns = ["*)(uid=*", "*)(cn=*", "admin*"]
        if any(pattern in data_str for pattern in ldap_patterns):
            return "injection_attempt"
        
        return None
    
    def _detect_anomalous_behavior(
        self, 
        source_ip: str, 
        user_id: Optional[str], 
        request_data: Dict[str, Any]
    ) -> bool:
        """Detect anomalous behavior patterns"""
        # Check for unusual request patterns
        request_size = len(json.dumps(request_data, default=str))
        if request_size > 50000:  # Unusually large request
            return True
        
        # Check for unusual parameter combinations
        sensitive_params = ["password", "token", "key", "secret"]
        if len([k for k in request_data.keys() if any(sp in k.lower() for sp in sensitive_params)]) > 3:
            return True
        
        # Check for suspicious file operations
        file_patterns = ["../", "..\\", "/etc/", "C:\\Windows"]
        data_str = json.dumps(request_data, default=str)
        if any(pattern in data_str for pattern in file_patterns):
            return True
        
        return False
    
    def _check_rate_limits(self, source_ip: str, user_id: Optional[str]) -> bool:
        """Check if rate limits are exceeded"""
        # Simple rate limiting check (would be more sophisticated in production)
        key = f"{source_ip}:{user_id or 'anon'}"
        current_minute = int(time.time() // 60)
        
        if key not in self.suspicious_patterns:
            self.suspicious_patterns[key] = 0
        
        # Reset counter every minute
        if current_minute % 60 == 0:
            self.suspicious_patterns[key] = 0
        
        self.suspicious_patterns[key] += 1
        return self.suspicious_patterns[key] > 100  # 100 requests per minute
    
    def _threat_priority(self, threat_level: ThreatLevel) -> int:
        """Get numeric priority for threat level"""
        priorities = {
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return priorities.get(threat_level, 0)
    
    def record_failed_attempt(self, source_ip: str) -> None:
        """Record failed authentication attempt"""
        now = datetime.utcnow()
        if source_ip not in self.failed_attempts:
            self.failed_attempts[source_ip] = []
        
        self.failed_attempts[source_ip].append(now)
        
        # Keep only recent attempts
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[source_ip] = [
            attempt for attempt in self.failed_attempts[source_ip]
            if attempt > cutoff
        ]


class SecurityManager:
    """
    Comprehensive security management system with adaptive responses
    """
    
    def __init__(self):
        self.detector = ThreatDetector()
        self.sanitizer = InputSanitizer()
        self.events: List[SecurityEvent] = []
        self.rules: Dict[str, SecurityRule] = {}
        self.blocked_ips: Set[str] = set()
        self.quarantined_users: Set[str] = set()
        self._load_default_rules()
        
    def _load_default_rules(self) -> None:
        """Load default security rules"""
        default_rules = [
            SecurityRule(
                rule_id="sql_injection",
                name="SQL Injection Detection",
                description="Detects SQL injection attempts in input",
                pattern="sql_injection",
                threat_level=ThreatLevel.CRITICAL,
                action=SecurityAction.BLOCK_IP
            ),
            SecurityRule(
                rule_id="brute_force",
                name="Brute Force Detection",
                description="Detects brute force authentication attempts",
                pattern="brute_force",
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.RATE_LIMIT
            ),
            SecurityRule(
                rule_id="xss_attempt",
                name="Cross-Site Scripting Detection",
                description="Detects XSS attempts in input",
                pattern="xss",
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.LOG_ONLY
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    async def analyze_security_event(
        self, 
        source_ip: str, 
        user_id: Optional[str], 
        request_data: Dict[str, Any]
    ) -> Optional[SecurityEvent]:
        """Analyze request for security threats"""
        with tracer.start_as_current_span("security_analysis") as span:
            span.set_attributes({
                "source_ip": source_ip,
                "user_id": user_id or "anonymous"
            })
            
            # Check if IP is already blocked
            if source_ip in self.blocked_ips:
                return SecurityEvent(
                    event_id=f"blocked_ip_{int(time.time())}",
                    event_type=SecurityEventType.AUTHENTICATION_FAILURE,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    user_id=user_id,
                    details={"reason": "IP blocked due to previous violations"}
                )
            
            # Sanitize input
            sanitized_data = self.sanitizer.sanitize_input(request_data)
            
            # Detect threats
            event = self.detector.analyze_request(source_ip, user_id, sanitized_data)
            
            if event:
                await self._handle_security_event(event)
                self.events.append(event)
                
                # Feed to adaptive intelligence
                intelligence = await get_intelligence()
                await intelligence.ingest_data_point(
                    PatternType.SECURITY,
                    {
                        "event_type": event.event_type.value,
                        "threat_level": event.threat_level.value,
                        "source_ip": event.source_ip,
                        "timestamp": event.timestamp.isoformat()
                    }
                )
            
            return event
    
    async def _handle_security_event(self, event: SecurityEvent) -> None:
        """Handle security event based on rules"""
        with tracer.start_as_current_span("handle_security_event") as span:
            span.set_attributes({
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value
            })
            
            # Find matching rule
            matching_rule = None
            for rule in self.rules.values():
                if rule.enabled and event.event_type.value == rule.pattern:
                    matching_rule = rule
                    break
            
            if not matching_rule:
                logger.warning(f"No rule found for event type: {event.event_type.value}")
                return
            
            # Execute action
            await self._execute_security_action(event, matching_rule)
            
            # Update rule statistics
            matching_rule.detections += 1
    
    async def _execute_security_action(self, event: SecurityEvent, rule: SecurityRule) -> None:
        """Execute security response action"""
        with tracer.start_as_current_span("execute_security_action") as span:
            span.set_attributes({
                "action": rule.action.value,
                "rule_id": rule.rule_id
            })
            
            logger.warning(f"Executing security action: {rule.action.value} for event: {event.event_id}")
            
            if rule.action == SecurityAction.BLOCK_IP:
                self.blocked_ips.add(event.source_ip)
                logger.critical(f"IP blocked: {event.source_ip}")
                
            elif rule.action == SecurityAction.RATE_LIMIT:
                # Implement rate limiting (would integrate with actual rate limiter)
                logger.warning(f"Rate limiting applied to IP: {event.source_ip}")
                
            elif rule.action == SecurityAction.ISOLATE_SESSION:
                if event.user_id:
                    self.quarantined_users.add(event.user_id)
                    logger.warning(f"User session isolated: {event.user_id}")
                    
            elif rule.action == SecurityAction.REQUIRE_MFA:
                logger.info(f"MFA required for user: {event.user_id}")
                
            elif rule.action == SecurityAction.EMERGENCY_SHUTDOWN:
                logger.critical("EMERGENCY SHUTDOWN TRIGGERED - Critical security threat detected")
                # Would trigger actual system shutdown procedures
                
            # Always log the event
            logger.info(f"Security event logged: {event.to_dict()}")
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token with enhanced security"""
        try:
            # Decode with verification
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=["HS256"],
                options={"verify_exp": True, "verify_iat": True}
            )
            
            # Additional security checks
            if "user_id" not in payload:
                raise jwt.InvalidTokenError("Missing user_id in token")
            
            if payload.get("user_id") in self.quarantined_users:
                raise jwt.InvalidTokenError("User session is quarantined")
            
            # Check token age
            issued_at = payload.get("iat", 0)
            if time.time() - issued_at > 86400:  # 24 hours
                raise jwt.ExpiredSignatureError("Token too old")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise
    
    def generate_secure_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        """Generate secure JWT token"""
        now = time.time()
        payload = {
            "user_id": user_id,
            "iat": now,
            "exp": now + 3600,  # 1 hour expiry
            "jti": secrets.token_hex(16),  # Unique token ID
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        key = settings.SECRET_KEY.encode()[:32]  # Use first 32 bytes of secret key
        f = Fernet(key)
        return f.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        key = settings.SECRET_KEY.encode()[:32]
        f = Fernet(key)
        return f.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{hashed.hex()}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = hashed_password.split(':')
            hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hashed.hex() == hash_hex
        except ValueError:
            return False
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def is_user_quarantined(self, user_id: str) -> bool:
        """Check if user is quarantined"""
        return user_id in self.quarantined_users
    
    def unblock_ip(self, ip: str) -> bool:
        """Unblock IP address"""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            logger.info(f"IP unblocked: {ip}")
            return True
        return False
    
    def release_user_quarantine(self, user_id: str) -> bool:
        """Release user from quarantine"""
        if user_id in self.quarantined_users:
            self.quarantined_users.remove(user_id)
            logger.info(f"User released from quarantine: {user_id}")
            return True
        return False
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        with tracer.start_as_current_span("security_report"):
            recent_events = [
                event for event in self.events
                if event.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]
            
            events_by_type = {}
            events_by_level = {}
            
            for event in recent_events:
                event_type = event.event_type.value
                threat_level = event.threat_level.value
                
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                events_by_level[threat_level] = events_by_level.get(threat_level, 0) + 1
            
            rule_performance = {}
            for rule in self.rules.values():
                rule_performance[rule.name] = {
                    "detections": rule.detections,
                    "false_positive_rate": rule.false_positive_rate,
                    "enabled": rule.enabled
                }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_events_24h": len(recent_events),
                "events_by_type": events_by_type,
                "events_by_threat_level": events_by_level,
                "blocked_ips": len(self.blocked_ips),
                "quarantined_users": len(self.quarantined_users),
                "active_rules": len([r for r in self.rules.values() if r.enabled]),
                "rule_performance": rule_performance
            }


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create the global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


async def analyze_request_security(
    source_ip: str, 
    user_id: Optional[str], 
    request_data: Dict[str, Any]
) -> Optional[SecurityEvent]:
    """Convenience function to analyze request security"""
    security_manager = get_security_manager()
    return await security_manager.analyze_security_event(source_ip, user_id, request_data)