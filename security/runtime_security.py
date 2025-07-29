"""
Runtime Application Security Protection (RASP)
Advanced runtime security monitoring, threat detection, and automatic response system.
"""

import asyncio
import time
import json
import logging
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import inspect
import traceback
from pathlib import Path
import uuid

# IP address validation
import ipaddress
import socket


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of security attacks detected."""
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    SUSPICIOUS_PAYLOAD = "suspicious_payload"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    BRUTE_FORCE = "brute_force"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class SecurityAction(Enum):
    """Security response actions."""
    LOG = "log"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ALERT = "alert"
    TERMINATE = "terminate"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    timestamp: datetime
    attack_type: AttackType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    endpoint: str
    payload: str
    action_taken: SecurityAction
    blocked: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'attack_type': self.attack_type.value,
            'threat_level': self.threat_level.value,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'endpoint': self.endpoint,
            'payload': self.payload[:500] + '...' if len(self.payload) > 500 else self.payload,
            'action_taken': self.action_taken.value,
            'blocked': self.blocked,
            'metadata': self.metadata
        }


class SecurityPatterns:
    """Security threat detection patterns."""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b.*\b=\b.*\bOR\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\bINSERT\b.*\bINTO\b.*\bVALUES\b)",
        r"(\bDELETE\b.*\bFROM\b.*\bWHERE\b)",
        r"(\b1\b.*\b=\b.*\b1\b)",
        r"(\'\s*OR\s*\'\w*\'\s*=\s*\'\w*)",
        r"(\'\s*;\s*DROP\s*TABLE)",
        r"(\bEXEC\s*\(\s*CHAR\b)",
        r"(\bCONCAT\s*\(\s*CHAR\b)"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"(<script[^>]*>.*?</script>)",
        r"(<img[^>]+src[^>]*=.*?javascript:)",
        r"(<iframe[^>]*>)",
        r"(on\w+\s*=\s*[\"'].*?[\"'])",
        r"(javascript:.*)",
        r"(<object[^>]*>.*?</object>)",
        r"(<embed[^>]*>)",
        r"(<link[^>]*>)",
        r"(<meta[^>]*>)",
        r"(expression\s*\()"
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"(\.\.\/)",
        r"(\.\.\\)",
        r"(%2e%2e%2f)",
        r"(%2e%2e%5c)",
        r"(\.\.%2f)",
        r"(\.\.%5c)",
        r"(%252e%252e%252f)",
        r"(%c0%ae%c0%ae%c0%af)",
        r"(\.\.\/.*\/etc\/passwd)",
        r"(\.\.\\.*\\windows\\system32)"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"(\||\&|\;|\$\(|\`)",
        r"(\bnc\b.*\-l)",
        r"(\bwget\b|\bcurl\b)",
        r"(\bchmod\b|\bchown\b)",
        r"(\brm\b.*\-rf)",
        r"(\bcat\b.*\/etc\/passwd)",
        r"(\buname\b|\bwhoami\b|\bid\b)",
        r"(\bping\b|\bnslookup\b)",
        r"(\bnetcat\b|\btelnet\b)",
        r"(\bexport\b|\benv\b)"
    ]
    
    # Suspicious payload patterns
    SUSPICIOUS_PATTERNS = [
        r"(password.*=.*admin)",
        r"(admin.*password)",
        r"(\beval\b.*\()",
        r"(\bexec\b.*\()",
        r"(\bsystem\b.*\()",
        r"(\bshell_exec\b.*\()",
        r"(\bpassthru\b.*\()",
        r"(base64_decode.*\()",
        r"(\bfile_get_contents\b.*http)",
        r"(\bcreate_function\b.*\()"
    ]


class ThreatDetector:
    """Core threat detection engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ThreatDetector")
        self.compiled_patterns = self._compile_patterns()
        
        # Learning mode - adapts to normal traffic patterns
        self.baseline_patterns = defaultdict(set)
        self.anomaly_threshold = 0.7
        
    def _compile_patterns(self) -> Dict[AttackType, List]:
        """Compile regex patterns for performance."""
        compiled = {}
        
        compiled[AttackType.SQL_INJECTION] = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in SecurityPatterns.SQL_INJECTION_PATTERNS
        ]
        
        compiled[AttackType.XSS] = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in SecurityPatterns.XSS_PATTERNS
        ]
        
        compiled[AttackType.PATH_TRAVERSAL] = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SecurityPatterns.PATH_TRAVERSAL_PATTERNS
        ]
        
        compiled[AttackType.COMMAND_INJECTION] = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SecurityPatterns.COMMAND_INJECTION_PATTERNS
        ]
        
        compiled[AttackType.SUSPICIOUS_PAYLOAD] = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SecurityPatterns.SUSPICIOUS_PATTERNS
        ]
        
        return compiled
    
    def analyze_request(self, request_data: Dict[str, Any]) -> Tuple[List[AttackType], ThreatLevel]:
        """Analyze request for security threats."""
        detected_attacks = []
        max_threat_level = ThreatLevel.LOW
        
        # Extract data to analyze
        payload = self._extract_payload(request_data)
        
        # Check against each attack pattern
        for attack_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(payload):
                    detected_attacks.append(attack_type)
                    
                    # Determine threat level based on attack type
                    threat_level = self._get_threat_level(attack_type)
                    if self._is_higher_threat(threat_level, max_threat_level):
                        max_threat_level = threat_level
                    
                    self.logger.warning(f"Detected {attack_type.value} attack in payload")
                    break
        
        # Check for anomalous behavior
        if self._is_anomalous_request(request_data):
            detected_attacks.append(AttackType.ANOMALOUS_BEHAVIOR)
            if max_threat_level == ThreatLevel.LOW:
                max_threat_level = ThreatLevel.MEDIUM
        
        return detected_attacks, max_threat_level
    
    def _extract_payload(self, request_data: Dict[str, Any]) -> str:
        """Extract analyzable payload from request."""
        payload_parts = []
        
        # URL parameters
        if 'query_params' in request_data:
            payload_parts.extend(str(v) for v in request_data['query_params'].values())
        
        # POST/PUT body
        if 'body' in request_data:
            payload_parts.append(str(request_data['body']))
        
        # Headers (selective)
        if 'headers' in request_data:
            suspicious_headers = ['user-agent', 'referer', 'x-forwarded-for']
            for header in suspicious_headers:
                if header in request_data['headers']:
                    payload_parts.append(request_data['headers'][header])
        
        # Cookies
        if 'cookies' in request_data:
            payload_parts.extend(str(v) for v in request_data['cookies'].values())
        
        return ' '.join(payload_parts)
    
    def _get_threat_level(self, attack_type: AttackType) -> ThreatLevel:
        """Determine threat level for attack type."""
        critical_attacks = [
            AttackType.SQL_INJECTION,
            AttackType.COMMAND_INJECTION,
            AttackType.PRIVILEGE_ESCALATION
        ]
        
        high_attacks = [
            AttackType.XSS,
            AttackType.PATH_TRAVERSAL,
            AttackType.DATA_EXFILTRATION
        ]
        
        medium_attacks = [
            AttackType.SUSPICIOUS_PAYLOAD,
            AttackType.BRUTE_FORCE,
            AttackType.RATE_LIMIT_ABUSE
        ]
        
        if attack_type in critical_attacks:
            return ThreatLevel.CRITICAL
        elif attack_type in high_attacks:
            return ThreatLevel.HIGH
        elif attack_type in medium_attacks:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _is_higher_threat(self, new_level: ThreatLevel, current_level: ThreatLevel) -> bool:
        """Check if new threat level is higher than current."""
        threat_hierarchy = {
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return threat_hierarchy[new_level] > threat_hierarchy[current_level]
    
    def _is_anomalous_request(self, request_data: Dict[str, Any]) -> bool:
        """Detect anomalous behavior patterns."""
        # Simple anomaly detection based on request characteristics
        anomaly_score = 0.0
        
        # Check payload size
        payload = self._extract_payload(request_data)
        if len(payload) > 10000:  # Large payload
            anomaly_score += 0.3
        
        # Check for unusual characters
        suspicious_chars = len(re.findall(r'[<>"\'\(\)\{\}\[\]\\]', payload))
        if suspicious_chars > 20:
            anomaly_score += 0.4
        
        # Check for encoded content
        if re.search(r'%[0-9a-fA-F]{2}', payload):
            anomaly_score += 0.2
        
        # Check request frequency (simplified)
        source_ip = request_data.get('source_ip', '')
        if source_ip in self.baseline_patterns:
            # More sophisticated pattern analysis would go here
            pass
        
        return anomaly_score >= self.anomaly_threshold


class RateLimitProtection:
    """Rate limiting and abuse protection."""
    
    def __init__(self, default_limit: int = 100, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.request_counts = defaultdict(lambda: deque())
        self.blocked_ips = {}
        self.logger = logging.getLogger(f"{__name__}.RateLimitProtection")
        
        # Different limits for different endpoints
        self.endpoint_limits = {
            '/api/auth/login': 5,  # Login attempts
            '/api/auth/register': 10,  # Registration
            '/api/search': 50,  # Search operations
            '/api/upload': 20,  # File uploads
        }
    
    def check_rate_limit(self, source_ip: str, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request should be rate limited."""
        current_time = time.time()
        
        # Check if IP is currently blocked
        if source_ip in self.blocked_ips:
            if current_time < self.blocked_ips[source_ip]:
                return False, {
                    'reason': 'IP temporarily blocked',
                    'blocked_until': self.blocked_ips[source_ip],
                    'remaining_time': self.blocked_ips[source_ip] - current_time
                }
            else:
                # Unblock IP
                del self.blocked_ips[source_ip]
        
        # Get rate limit for endpoint
        limit = self.endpoint_limits.get(endpoint, self.default_limit)
        
        # Clean old requests outside the window
        request_key = f"{source_ip}:{endpoint}"
        request_times = self.request_counts[request_key]
        
        while request_times and current_time - request_times[0] > self.window_seconds:
            request_times.popleft()
        
        # Check if limit exceeded
        if len(request_times) >= limit:
            # Block IP for extended period on severe abuse
            if len(request_times) > limit * 2:
                self.blocked_ips[source_ip] = current_time + 3600  # Block for 1 hour
                self.logger.warning(f"Blocked IP {source_ip} for severe rate limit abuse")
            
            return False, {
                'reason': 'Rate limit exceeded',
                'limit': limit,
                'window_seconds': self.window_seconds,
                'current_count': len(request_times),
                'retry_after': self.window_seconds
            }
        
        # Allow request and record it
        request_times.append(current_time)
        
        return True, {
            'allowed': True,
            'remaining': limit - len(request_times),
            'reset_time': current_time + self.window_seconds
        }


class SecurityResponseManager:
    """Manages automated security responses."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SecurityResponseManager")
        self.response_rules = self._initialize_response_rules()
        self.quarantined_sessions = set()
        self.blocked_ips = set()
        
    def _initialize_response_rules(self) -> Dict[Tuple[AttackType, ThreatLevel], SecurityAction]:
        """Initialize automated response rules."""
        return {
            # Critical threats - immediate termination
            (AttackType.SQL_INJECTION, ThreatLevel.CRITICAL): SecurityAction.TERMINATE,
            (AttackType.COMMAND_INJECTION, ThreatLevel.CRITICAL): SecurityAction.TERMINATE,
            (AttackType.PRIVILEGE_ESCALATION, ThreatLevel.CRITICAL): SecurityAction.TERMINATE,
            
            # High threats - block and alert
            (AttackType.SQL_INJECTION, ThreatLevel.HIGH): SecurityAction.BLOCK,
            (AttackType.XSS, ThreatLevel.HIGH): SecurityAction.BLOCK,
            (AttackType.PATH_TRAVERSAL, ThreatLevel.HIGH): SecurityAction.BLOCK,
            (AttackType.DATA_EXFILTRATION, ThreatLevel.HIGH): SecurityAction.BLOCK,
            
            # Medium threats - quarantine
            (AttackType.SUSPICIOUS_PAYLOAD, ThreatLevel.MEDIUM): SecurityAction.QUARANTINE,
            (AttackType.ANOMALOUS_BEHAVIOR, ThreatLevel.MEDIUM): SecurityAction.QUARANTINE,
            (AttackType.BRUTE_FORCE, ThreatLevel.MEDIUM): SecurityAction.QUARANTINE,
            
            # Low threats - log and alert
            (AttackType.RATE_LIMIT_ABUSE, ThreatLevel.LOW): SecurityAction.ALERT,
        }
    
    def determine_response(self, attack_types: List[AttackType], threat_level: ThreatLevel) -> SecurityAction:
        """Determine appropriate security response."""
        max_action = SecurityAction.LOG
        
        for attack_type in attack_types:
            rule_key = (attack_type, threat_level)
            if rule_key in self.response_rules:
                action = self.response_rules[rule_key]
                if self._is_stronger_action(action, max_action):
                    max_action = action
        
        return max_action
    
    def _is_stronger_action(self, new_action: SecurityAction, current_action: SecurityAction) -> bool:
        """Check if new action is stronger than current."""
        action_strength = {
            SecurityAction.LOG: 1,
            SecurityAction.ALERT: 2,
            SecurityAction.QUARANTINE: 3,
            SecurityAction.BLOCK: 4,
            SecurityAction.TERMINATE: 5
        }
        return action_strength[new_action] > action_strength[current_action]
    
    def execute_response(self, action: SecurityAction, request_data: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Execute security response action."""
        source_ip = request_data.get('source_ip', 'unknown')
        session_id = request_data.get('session_id', 'unknown')
        
        try:
            if action == SecurityAction.TERMINATE:
                self.blocked_ips.add(source_ip)
                self.quarantined_sessions.add(session_id)
                self.logger.critical(f"TERMINATED connection from {source_ip} - session {session_id}")
                return True
                
            elif action == SecurityAction.BLOCK:
                self.blocked_ips.add(source_ip)
                self.logger.warning(f"BLOCKED IP {source_ip}")
                return True
                
            elif action == SecurityAction.QUARANTINE:
                self.quarantined_sessions.add(session_id)
                self.logger.warning(f"QUARANTINED session {session_id} from {source_ip}")
                return True
                
            elif action == SecurityAction.ALERT:
                self.logger.info(f"SECURITY ALERT: {metadata.get('attack_types', [])} from {source_ip}")
                # Would send to monitoring system in production
                return True
                
            elif action == SecurityAction.LOG:
                self.logger.debug(f"Security event logged for {source_ip}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to execute security response {action}: {e}")
            return False
        
        return False


class RuntimeSecurityProtection:
    """Main RASP (Runtime Application Security Protection) system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.RuntimeSecurityProtection")
        
        # Initialize components
        self.threat_detector = ThreatDetector()
        self.rate_limiter = RateLimitProtection()
        self.response_manager = SecurityResponseManager()
        
        # Event storage
        self.security_events: deque = deque(maxlen=10000)
        self.active_threats = defaultdict(int)
        
        # Performance metrics
        self.total_requests = 0
        self.blocked_requests = 0
        self.detection_time_ms = deque(maxlen=1000)
        
        # Enable/disable protection
        self.protection_enabled = self.config.get('enabled', True)
        
        self.logger.info("Runtime Security Protection initialized")
    
    async def protect_request(self, request_data: Dict[str, Any]) -> Tuple[bool, Optional[SecurityEvent]]:
        """Main protection function - analyze and protect against request."""
        if not self.protection_enabled:
            return True, None
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            source_ip = request_data.get('source_ip', 'unknown')
            endpoint = request_data.get('endpoint', '/')
            
            # Step 1: Rate limiting check
            rate_allowed, rate_info = self.rate_limiter.check_rate_limit(source_ip, endpoint)
            if not rate_allowed:
                security_event = self._create_security_event(
                    attack_types=[AttackType.RATE_LIMIT_ABUSE],
                    threat_level=ThreatLevel.MEDIUM,
                    request_data=request_data,
                    action_taken=SecurityAction.BLOCK,
                    blocked=True,
                    metadata={'rate_limit_info': rate_info}
                )
                
                self.security_events.append(security_event)
                self.blocked_requests += 1
                return False, security_event
            
            # Step 2: Threat detection
            detected_attacks, threat_level = self.threat_detector.analyze_request(request_data)
            
            if detected_attacks:
                # Step 3: Determine response
                response_action = self.response_manager.determine_response(detected_attacks, threat_level)
                
                # Step 4: Execute response
                blocked = response_action in [SecurityAction.BLOCK, SecurityAction.TERMINATE]
                success = self.response_manager.execute_response(
                    response_action, 
                    request_data, 
                    {'attack_types': detected_attacks, 'threat_level': threat_level}
                )
                
                # Step 5: Create security event
                security_event = self._create_security_event(
                    attack_types=detected_attacks,
                    threat_level=threat_level,
                    request_data=request_data,
                    action_taken=response_action,
                    blocked=blocked,
                    metadata={
                        'response_success': success,
                        'detection_patterns': [attack.value for attack in detected_attacks]
                    }
                )
                
                self.security_events.append(security_event)
                
                # Update threat counters
                for attack in detected_attacks:
                    self.active_threats[attack] += 1
                
                if blocked:
                    self.blocked_requests += 1
                
                # Record detection time
                detection_time = (time.time() - start_time) * 1000
                self.detection_time_ms.append(detection_time)
                
                return not blocked, security_event
            
            # Clean request - allow through
            detection_time = (time.time() - start_time) * 1000
            self.detection_time_ms.append(detection_time)
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error in security protection: {e}")
            self.logger.error(traceback.format_exc())
            # Fail open - allow request but log error
            return True, None
    
    def _create_security_event(
        self,
        attack_types: List[AttackType],
        threat_level: ThreatLevel,
        request_data: Dict[str, Any],
        action_taken: SecurityAction,
        blocked: bool,
        metadata: Dict[str, Any]
    ) -> SecurityEvent:
        """Create security event record."""
        return SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            attack_type=attack_types[0] if attack_types else AttackType.ANOMALOUS_BEHAVIOR,
            threat_level=threat_level,
            source_ip=request_data.get('source_ip', 'unknown'),
            user_id=request_data.get('user_id'),
            endpoint=request_data.get('endpoint', '/'),
            payload=self.threat_detector._extract_payload(request_data),
            action_taken=action_taken,
            blocked=blocked,
            metadata={
                **metadata,
                'all_attack_types': [attack.value for attack in attack_types],
                'user_agent': request_data.get('headers', {}).get('user-agent', ''),
                'request_method': request_data.get('method', 'GET')
            }
        )
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        recent_events = [e for e in self.security_events if e.timestamp >= datetime.now() - timedelta(hours=24)]
        
        attack_counts = defaultdict(int)
        threat_level_counts = defaultdict(int)
        blocked_count = 0
        
        for event in recent_events:
            attack_counts[event.attack_type.value] += 1
            threat_level_counts[event.threat_level.value] += 1
            if event.blocked:
                blocked_count += 1
        
        avg_detection_time = sum(self.detection_time_ms) / len(self.detection_time_ms) if self.detection_time_ms else 0
        
        return {
            'protection_enabled': self.protection_enabled,
            'total_requests_processed': self.total_requests,
            'total_requests_blocked': self.blocked_requests,
            'block_rate_percentage': (self.blocked_requests / max(self.total_requests, 1)) * 100,
            'events_last_24h': len(recent_events),
            'blocked_last_24h': blocked_count,
            'attack_type_distribution': dict(attack_counts),
            'threat_level_distribution': dict(threat_level_counts),
            'average_detection_time_ms': avg_detection_time,
            'active_blocked_ips': len(self.response_manager.blocked_ips),
            'quarantined_sessions': len(self.response_manager.quarantined_sessions),
            'rate_limiting_active': True
        }
    
    def get_recent_events(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp >= cutoff_time]
        
        # Sort by timestamp (most recent first) and limit
        recent_events.sort(key=lambda x: x.timestamp, reverse=True)
        return [event.to_dict() for event in recent_events[:limit]]
    
    def enable_protection(self):
        """Enable security protection."""
        self.protection_enabled = True
        self.logger.info("Runtime security protection ENABLED")
    
    def disable_protection(self):
        """Disable security protection."""
        self.protection_enabled = False
        self.logger.warning("Runtime security protection DISABLED")


# Standalone security validation
async def validate_runtime_security():
    """Validate runtime security protection system."""
    print("🛡️  Runtime Security Protection Validation")
    print("=" * 50)
    
    # Initialize RASP system
    rasp = RuntimeSecurityProtection()
    
    print("\n1. Testing Clean Requests")
    clean_requests = [
        {
            'source_ip': '192.168.1.100',
            'endpoint': '/api/users',
            'method': 'GET',
            'query_params': {'page': '1', 'limit': '10'},
            'headers': {'user-agent': 'Mozilla/5.0 (legitimate browser)'},
            'session_id': 'clean_session_123'
        },
        {
            'source_ip': '192.168.1.101',
            'endpoint': '/api/search',
            'method': 'POST',
            'body': {'query': 'machine learning tutorials', 'category': 'technology'},
            'headers': {'user-agent': 'PostmanRuntime/7.26.8'},
            'session_id': 'clean_session_456'
        }
    ]
    
    clean_allowed = 0
    for request in clean_requests:
        allowed, event = await rasp.protect_request(request)
        if allowed:
            clean_allowed += 1
    
    print(f"   ✅ Clean requests allowed: {clean_allowed}/{len(clean_requests)}")
    
    print("\n2. Testing SQL Injection Attacks")
    sql_attacks = [
        {
            'source_ip': '10.0.0.50',
            'endpoint': '/api/login',
            'method': 'POST',
            'body': {'username': "admin' OR '1'='1", 'password': 'any'},
            'session_id': 'attack_session_1'
        },
        {
            'source_ip': '10.0.0.51',
            'endpoint': '/api/search',
            'method': 'GET',
            'query_params': {'q': "'; DROP TABLE users; --"},
            'session_id': 'attack_session_2'
        },
        {
            'source_ip': '10.0.0.52',
            'endpoint': '/api/data',
            'method': 'GET',
            'query_params': {'id': '1 UNION SELECT password FROM users'},
            'session_id': 'attack_session_3'
        }
    ]
    
    sql_blocked = 0
    for attack in sql_attacks:
        allowed, event = await rasp.protect_request(attack)
        if not allowed and event:
            sql_blocked += 1
            print(f"   🚨 Blocked SQL injection from {attack['source_ip']}: {event.attack_type.value}")
    
    print(f"   🛡️  SQL injection attacks blocked: {sql_blocked}/{len(sql_attacks)}")
    
    print("\n3. Testing XSS Attacks")
    xss_attacks = [
        {
            'source_ip': '10.0.0.60',
            'endpoint': '/api/comments',
            'method': 'POST',
            'body': {'comment': '<script>alert("XSS")</script>'},
            'session_id': 'xss_session_1'
        },
        {
            'source_ip': '10.0.0.61',
            'endpoint': '/api/profile',
            'method': 'PUT',
            'body': {'bio': '<img src=x onerror=alert(1)>'},
            'session_id': 'xss_session_2'
        }
    ]
    
    xss_blocked = 0
    for attack in xss_attacks:
        allowed, event = await rasp.protect_request(attack)
        if not allowed and event:
            xss_blocked += 1
            print(f"   🚨 Blocked XSS attack from {attack['source_ip']}: {event.payload[:50]}...")
    
    print(f"   🛡️  XSS attacks blocked: {xss_blocked}/{len(xss_attacks)}")
    
    print("\n4. Testing Rate Limiting")
    # Simulate rapid requests from same IP
    rate_limit_ip = '10.0.0.99'
    blocked_by_rate_limit = 0
    
    for i in range(15):  # Exceed default limit of 10 for login endpoint
        request = {
            'source_ip': rate_limit_ip,
            'endpoint': '/api/auth/login',
            'method': 'POST',
            'body': {'username': f'user{i}', 'password': 'test'},
            'session_id': f'rate_limit_session_{i}'
        }
        
        allowed, event = await rasp.protect_request(request)
        if not allowed and event and event.attack_type == AttackType.RATE_LIMIT_ABUSE:
            blocked_by_rate_limit += 1
    
    print(f"   🚨 Requests blocked by rate limiting: {blocked_by_rate_limit}")
    
    print("\n5. Testing Command Injection")
    cmd_attacks = [
        {
            'source_ip': '10.0.0.70',
            'endpoint': '/api/system',
            'method': 'POST',
            'body': {'command': 'ls; cat /etc/passwd'},
            'session_id': 'cmd_session_1'
        },
        {
            'source_ip': '10.0.0.71',
            'endpoint': '/api/backup',
            'method': 'GET',
            'query_params': {'file': '../../../etc/passwd'},
            'session_id': 'cmd_session_2'
        }
    ]
    
    cmd_blocked = 0
    for attack in cmd_attacks:
        allowed, event = await rasp.protect_request(attack)
        if not allowed and event:
            cmd_blocked += 1
            print(f"   🚨 Blocked command injection from {attack['source_ip']}")
    
    print(f"   🛡️  Command injection attacks blocked: {cmd_blocked}/{len(cmd_attacks)}")
    
    print("\n6. Security Statistics")
    stats = rasp.get_security_statistics()
    
    print(f"   Total Requests Processed: {stats['total_requests_processed']}")
    print(f"   Total Requests Blocked: {stats['total_requests_blocked']}")
    print(f"   Block Rate: {stats['block_rate_percentage']:.1f}%")
    print(f"   Average Detection Time: {stats['average_detection_time_ms']:.2f}ms")
    print(f"   Active Blocked IPs: {stats['active_blocked_ips']}")
    print(f"   Events Last 24h: {stats['events_last_24h']}")
    
    print("\n7. Attack Type Distribution")
    for attack_type, count in stats['attack_type_distribution'].items():
        print(f"   {attack_type.replace('_', ' ').title()}: {count}")
    
    print("\n8. Recent Security Events")
    recent_events = rasp.get_recent_events(hours=1, limit=5)
    for event in recent_events[:3]:  # Show first 3
        print(f"   🔍 {event['timestamp'][:19]} - {event['attack_type']} from {event['source_ip']} - {'BLOCKED' if event['blocked'] else 'ALLOWED'}")
    
    # Validation criteria
    total_attacks = len(sql_attacks) + len(xss_attacks) + len(cmd_attacks)
    total_blocked = sql_blocked + xss_blocked + cmd_blocked
    
    attack_detection_rate = (total_blocked / total_attacks) * 100 if total_attacks > 0 else 0
    clean_pass_rate = (clean_allowed / len(clean_requests)) * 100 if clean_requests else 0
    
    print(f"\n✅ Runtime Security Protection Validation Complete")
    print(f"🎯 Attack Detection Rate: {attack_detection_rate:.1f}%")
    print(f"✅ Clean Request Pass Rate: {clean_pass_rate:.1f}%")
    print(f"⚡ Average Detection Time: {stats['average_detection_time_ms']:.2f}ms")
    print(f"🚫 Rate Limiting Active: {'Yes' if blocked_by_rate_limit > 0 else 'No'}")
    
    # Success criteria: >80% attack detection, >95% clean pass rate, <50ms detection time
    success = (
        attack_detection_rate >= 80 and
        clean_pass_rate >= 95 and
        stats['average_detection_time_ms'] < 50
    )
    
    return success


if __name__ == "__main__":
    # Run standalone validation
    compliance = asyncio.run(validate_runtime_security())
    print(f"\n🏆 Runtime Security Target: {'✅ ACHIEVED' if compliance else '❌ NOT MET'}")
    exit(0 if compliance else 1)