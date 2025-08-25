"""
Zero Trust Security Framework - Never Trust, Always Verify
Comprehensive security framework with continuous verification and threat detection.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import ipaddress
from collections import defaultdict, deque
import jwt
import re

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Trust levels for entities"""
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class ThreatLevel(str, Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(str, Enum):
    """Types of security events"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_FAILURE = "login_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TOKEN_VALIDATION_FAILED = "token_validation_failed"
    ANOMALY_DETECTED = "anomaly_detected"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityContext:
    """Security context for requests"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    trust_level: TrustLevel
    permissions: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_verified: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThreatIndicator:
    """Security threat indicator"""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False


@dataclass
class AccessAttempt:
    """Record of access attempt"""
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    method: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time: float = 0.0
    additional_context: Dict[str, Any] = field(default_factory=dict)


class ZeroTrustFramework:
    """
    Comprehensive Zero Trust Security Framework.
    
    Principles:
    - Never trust, always verify
    - Assume breach scenarios
    - Principle of least privilege
    - Continuous monitoring and verification
    - Dynamic risk assessment
    """
    
    def __init__(
        self,
        secret_key: str,
        max_failed_attempts: int = 5,
        lockout_duration: int = 900,  # 15 minutes
        session_timeout: int = 3600,  # 1 hour
        anomaly_threshold: float = 0.7
    ):
        self.secret_key = secret_key.encode('utf-8')
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        self.session_timeout = session_timeout
        self.anomaly_threshold = anomaly_threshold
        
        # Security state tracking
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: defaultdict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Dict[str, datetime] = {}
        self.threat_indicators: deque[ThreatIndicator] = deque(maxlen=10000)
        self.access_log: deque[AccessAttempt] = deque(maxlen=50000)
        
        # Behavioral analysis
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.ip_reputation: Dict[str, float] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, deque[datetime]] = defaultdict(lambda: deque(maxlen=100))
        
        # Threading
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
        
    async def start_monitoring(self):
        """Start background security monitoring tasks"""
        
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_data())
            
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._continuous_monitoring())
            
        self.logger.info("Zero Trust security monitoring started")
        
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            
        self.logger.info("Zero Trust security monitoring stopped")
    
    async def authenticate_user(
        self, 
        user_id: str, 
        credentials: Dict[str, Any],
        ip_address: str,
        user_agent: str
    ) -> Optional[SecurityContext]:
        """
        Authenticate user with zero trust principles.
        
        Args:
            user_id: User identifier
            credentials: Authentication credentials
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            SecurityContext if authentication successful, None otherwise
        """
        
        with self._lock:
            
            # Check if IP is blocked
            if await self._is_ip_blocked(ip_address):
                await self._record_threat(
                    SecurityEvent.LOGIN_ATTEMPT,
                    ThreatLevel.HIGH,
                    ip_address,
                    user_id,
                    "Login attempt from blocked IP",
                    {"user_agent": user_agent}
                )
                return None
            
            # Check rate limiting
            if not await self._check_rate_limit(ip_address, "login", 10, 300):  # 10 attempts per 5 minutes
                await self._record_threat(
                    SecurityEvent.RATE_LIMIT_EXCEEDED,
                    ThreatLevel.MEDIUM,
                    ip_address,
                    user_id,
                    "Login rate limit exceeded",
                    {"user_agent": user_agent}
                )
                return None
            
            # Validate credentials (placeholder - integrate with actual auth system)
            if not await self._validate_credentials(user_id, credentials):
                await self._record_failed_attempt(user_id, ip_address, user_agent)
                await self._record_threat(
                    SecurityEvent.LOGIN_FAILURE,
                    ThreatLevel.MEDIUM,
                    ip_address,
                    user_id,
                    "Invalid credentials",
                    {"user_agent": user_agent}
                )
                return None
            
            # Calculate initial trust level
            trust_level = await self._calculate_trust_level(user_id, ip_address, user_agent)
            
            # Create security context
            session_id = secrets.token_urlsafe(32)
            context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                trust_level=trust_level,
                permissions=await self._get_user_permissions(user_id)
            )
            
            self.active_sessions[session_id] = context
            
            # Update user profile
            await self._update_user_profile(user_id, ip_address, user_agent)
            
            # Record successful access
            await self._record_access_attempt(
                user_id, ip_address, user_agent, "authentication", "POST", True
            )
            
            self.logger.info(f"User {user_id} authenticated successfully with trust level {trust_level.value}")
            
            return context
    
    async def verify_request(
        self, 
        session_id: str,
        resource: str,
        method: str,
        ip_address: str,
        user_agent: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[SecurityContext]]:
        """
        Verify request with continuous security assessment.
        
        Args:
            session_id: Session identifier
            resource: Requested resource
            method: HTTP method
            ip_address: Client IP address
            user_agent: Client user agent
            additional_context: Additional context for verification
            
        Returns:
            Tuple of (authorized, security_context)
        """
        
        with self._lock:
            
            # Check if session exists and is valid
            context = self.active_sessions.get(session_id)
            if not context:
                await self._record_threat(
                    SecurityEvent.TOKEN_VALIDATION_FAILED,
                    ThreatLevel.HIGH,
                    ip_address,
                    None,
                    "Invalid session ID",
                    {"resource": resource, "method": method}
                )
                return False, None
            
            # Check session timeout
            if datetime.utcnow() - context.last_verified > timedelta(seconds=self.session_timeout):
                del self.active_sessions[session_id]
                await self._record_threat(
                    SecurityEvent.TOKEN_VALIDATION_FAILED,
                    ThreatLevel.MEDIUM,
                    ip_address,
                    context.user_id,
                    "Session expired",
                    {"resource": resource, "method": method}
                )
                return False, None
            
            # Verify request consistency
            if not await self._verify_request_consistency(context, ip_address, user_agent):
                await self._record_threat(
                    SecurityEvent.SUSPICIOUS_ACTIVITY,
                    ThreatLevel.HIGH,
                    ip_address,
                    context.user_id,
                    "Request consistency check failed",
                    {"expected_ip": context.ip_address, "actual_ip": ip_address}
                )
                # Don't immediately reject - might be legitimate (mobile, proxy, etc.)
                context.trust_level = self._downgrade_trust_level(context.trust_level)
            
            # Check permissions
            required_permission = self._get_required_permission(resource, method)
            if required_permission and required_permission not in context.permissions:
                await self._record_threat(
                    SecurityEvent.UNAUTHORIZED_ACCESS,
                    ThreatLevel.HIGH,
                    ip_address,
                    context.user_id,
                    f"Insufficient permissions for {resource}",
                    {"required": required_permission, "user_permissions": list(context.permissions)}
                )
                await self._record_access_attempt(
                    context.user_id, ip_address, user_agent, resource, method, False
                )
                return False, context
            
            # Rate limiting per user
            user_rate_key = f"user:{context.user_id}"
            if not await self._check_rate_limit(user_rate_key, "api", 1000, 3600):  # 1000 requests per hour
                await self._record_threat(
                    SecurityEvent.RATE_LIMIT_EXCEEDED,
                    ThreatLevel.MEDIUM,
                    ip_address,
                    context.user_id,
                    "User rate limit exceeded",
                    {"resource": resource, "method": method}
                )
                return False, context
            
            # Anomaly detection
            if await self._detect_anomalies(context, resource, method, additional_context):
                await self._record_threat(
                    SecurityEvent.ANOMALY_DETECTED,
                    ThreatLevel.MEDIUM,
                    ip_address,
                    context.user_id,
                    "Anomalous behavior detected",
                    {"resource": resource, "method": method, "context": additional_context}
                )
                context.trust_level = self._downgrade_trust_level(context.trust_level)
            
            # Update verification timestamp
            context.last_verified = datetime.utcnow()
            
            # Record successful access
            await self._record_access_attempt(
                context.user_id, ip_address, user_agent, resource, method, True
            )
            
            return True, context
    
    async def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is currently blocked"""
        
        if ip_address in self.blocked_ips:
            block_time = self.blocked_ips[ip_address]
            if datetime.utcnow() - block_time < timedelta(seconds=self.lockout_duration):
                return True
            else:
                # Unblock expired IPs
                del self.blocked_ips[ip_address]
                
        return False
    
    async def _check_rate_limit(
        self, 
        identifier: str, 
        action: str, 
        limit: int, 
        window: int
    ) -> bool:
        """Check if identifier is within rate limit"""
        
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window)
        
        key = f"{identifier}:{action}"
        timestamps = self.rate_limits[key]
        
        # Remove old timestamps
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
            
        # Check if limit exceeded
        if len(timestamps) >= limit:
            return False
            
        # Add current timestamp
        timestamps.append(now)
        return True
    
    async def _validate_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate user credentials (placeholder implementation)"""
        
        # This would integrate with your actual authentication system
        # For demo purposes, we'll do basic validation
        
        password = credentials.get('password', '')
        token = credentials.get('token', '')
        
        # Basic checks
        if not password and not token:
            return False
            
        # Token validation
        if token:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
                return payload.get('user_id') == user_id
            except jwt.InvalidTokenError:
                return False
        
        # Password validation (placeholder)
        if password:
            # In real implementation, compare with hashed password from database
            return len(password) >= 8  # Minimum length check
            
        return False
    
    async def _calculate_trust_level(
        self, 
        user_id: str, 
        ip_address: str, 
        user_agent: str
    ) -> TrustLevel:
        """Calculate initial trust level for user"""
        
        trust_score = 0.5  # Start with neutral trust
        
        # Check IP reputation
        ip_rep = self.ip_reputation.get(ip_address, 0.5)
        trust_score = (trust_score + ip_rep) / 2
        
        # Check user history
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            # Known IP addresses
            known_ips = profile.get('known_ips', set())
            if ip_address in known_ips:
                trust_score += 0.2
            
            # Known user agents
            known_agents = profile.get('known_agents', set())
            if user_agent in known_agents:
                trust_score += 0.1
            
            # Recent successful logins
            recent_success = profile.get('recent_success', 0)
            trust_score += min(recent_success * 0.1, 0.3)
            
        # Check for recent failed attempts
        recent_failures = len([
            attempt for attempt in self.failed_attempts.get(user_id, [])
            if datetime.utcnow() - attempt < timedelta(hours=24)
        ])
        
        if recent_failures > 0:
            trust_score -= min(recent_failures * 0.1, 0.4)
            
        # Convert score to trust level
        trust_score = max(0.0, min(1.0, trust_score))
        
        if trust_score >= 0.8:
            return TrustLevel.HIGH
        elif trust_score >= 0.6:
            return TrustLevel.MEDIUM
        elif trust_score >= 0.4:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED
    
    async def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get user permissions (placeholder implementation)"""
        
        # This would integrate with your authorization system
        # For demo purposes, return basic permissions
        
        base_permissions = {"read:basic", "write:own"}
        
        # Admin users get additional permissions
        if user_id.startswith("admin_"):
            base_permissions.update({"admin:read", "admin:write", "system:monitor"})
            
        # API users get API permissions
        if user_id.startswith("api_"):
            base_permissions.update({"api:read", "api:write"})
            
        return base_permissions
    
    def _get_required_permission(self, resource: str, method: str) -> Optional[str]:
        """Get required permission for resource and method"""
        
        # Define permission mapping
        permission_map = {
            ("/admin", "GET"): "admin:read",
            ("/admin", "POST"): "admin:write",
            ("/admin", "PUT"): "admin:write",
            ("/admin", "DELETE"): "admin:write",
            ("/api", "GET"): "api:read", 
            ("/api", "POST"): "api:write",
            ("/user", "GET"): "read:basic",
            ("/user", "POST"): "write:own",
        }
        
        # Simple prefix matching
        for (path_prefix, req_method), permission in permission_map.items():
            if resource.startswith(path_prefix) and method == req_method:
                return permission
                
        # Default read permission for GET requests
        if method == "GET":
            return "read:basic"
            
        return None
    
    async def _verify_request_consistency(
        self, 
        context: SecurityContext, 
        ip_address: str, 
        user_agent: str
    ) -> bool:
        """Verify request consistency with established context"""
        
        # Check IP address consistency
        if context.ip_address != ip_address:
            
            # Check if both IPs are in the same subnet (could be NAT/proxy)
            try:
                context_ip = ipaddress.ip_address(context.ip_address)
                request_ip = ipaddress.ip_address(ip_address)
                
                # Allow if in same /24 subnet (common for corporate networks)
                if context_ip.version == request_ip.version:
                    if context_ip.version == 4:
                        context_network = ipaddress.ip_network(f"{context_ip}/24", strict=False)
                        if request_ip in context_network:
                            return True
                            
            except ValueError:
                pass  # Invalid IP addresses
                
            return False
        
        # Check user agent consistency (allow minor variations)
        if context.user_agent != user_agent:
            # Basic similarity check (could be more sophisticated)
            common_parts = set(context.user_agent.split()) & set(user_agent.split())
            if len(common_parts) < 3:  # Too different
                return False
                
        return True
    
    async def _detect_anomalies(
        self, 
        context: SecurityContext,
        resource: str,
        method: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Detect anomalous behavior patterns"""
        
        anomaly_score = 0.0
        
        # Check access patterns
        user_accesses = [
            access for access in self.access_log
            if access.user_id == context.user_id and 
               access.timestamp >= datetime.utcnow() - timedelta(hours=1)
        ]
        
        if len(user_accesses) > 0:
            
            # Unusual resource access
            recent_resources = set(access.resource for access in user_accesses[-10:])
            if len(recent_resources) > 8:  # Accessing many different resources
                anomaly_score += 0.3
                
            # High request frequency
            if len(user_accesses) > 100:  # More than 100 requests in last hour
                anomaly_score += 0.4
                
            # Unusual timing patterns
            access_times = [access.timestamp.hour for access in user_accesses[-20:]]
            if len(set(access_times)) == 1 and len(access_times) > 10:
                # All requests at same hour (potential bot behavior)
                anomaly_score += 0.2
                
        # Check user profile deviation
        if context.user_id in self.user_profiles:
            profile = self.user_profiles[context.user_id]
            
            # Unusual resource for this user
            common_resources = profile.get('common_resources', set())
            if common_resources and resource not in common_resources and len(common_resources) > 5:
                anomaly_score += 0.2
                
            # Unusual time of access
            common_hours = profile.get('common_hours', set())
            current_hour = datetime.utcnow().hour
            if common_hours and current_hour not in common_hours and len(common_hours) > 2:
                anomaly_score += 0.1
        
        return anomaly_score >= self.anomaly_threshold
    
    def _downgrade_trust_level(self, current_level: TrustLevel) -> TrustLevel:
        """Downgrade trust level by one step"""
        
        downgrade_map = {
            TrustLevel.VERIFIED: TrustLevel.HIGH,
            TrustLevel.HIGH: TrustLevel.MEDIUM,
            TrustLevel.MEDIUM: TrustLevel.LOW,
            TrustLevel.LOW: TrustLevel.UNTRUSTED,
            TrustLevel.UNTRUSTED: TrustLevel.UNTRUSTED
        }
        
        return downgrade_map.get(current_level, TrustLevel.UNTRUSTED)
    
    async def _record_failed_attempt(self, user_id: str, ip_address: str, user_agent: str):
        """Record failed authentication attempt"""
        
        now = datetime.utcnow()
        
        # Record for user
        self.failed_attempts[user_id].append(now)
        
        # Check if user should be locked out
        recent_failures = [
            attempt for attempt in self.failed_attempts[user_id]
            if now - attempt < timedelta(seconds=self.lockout_duration)
        ]
        
        if len(recent_failures) >= self.max_failed_attempts:
            self.blocked_ips[ip_address] = now
            self.logger.warning(f"IP {ip_address} blocked due to failed attempts for user {user_id}")
            
        # Update IP reputation
        current_rep = self.ip_reputation.get(ip_address, 0.5)
        self.ip_reputation[ip_address] = max(current_rep - 0.1, 0.0)
    
    async def _update_user_profile(self, user_id: str, ip_address: str, user_agent: str):
        """Update user behavioral profile"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'known_ips': set(),
                'known_agents': set(),
                'common_resources': set(),
                'common_hours': set(),
                'recent_success': 0,
                'last_login': None
            }
            
        profile = self.user_profiles[user_id]
        
        # Update known IPs (keep last 10)
        profile['known_ips'].add(ip_address)
        if len(profile['known_ips']) > 10:
            # Remove oldest (this is simplified, real implementation would track timestamps)
            profile['known_ips'].pop()
            
        # Update known user agents (keep last 5)
        profile['known_agents'].add(user_agent)
        if len(profile['known_agents']) > 5:
            profile['known_agents'].pop()
            
        # Update success count
        profile['recent_success'] = min(profile['recent_success'] + 1, 10)
        profile['last_login'] = datetime.utcnow()
        
        # Update IP reputation positively
        current_rep = self.ip_reputation.get(ip_address, 0.5)
        self.ip_reputation[ip_address] = min(current_rep + 0.05, 1.0)
    
    async def _record_access_attempt(
        self, 
        user_id: Optional[str], 
        ip_address: str,
        user_agent: str,
        resource: str,
        method: str,
        success: bool
    ):
        """Record access attempt for analysis"""
        
        attempt = AccessAttempt(
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            method=method,
            success=success
        )
        
        self.access_log.append(attempt)
        
        # Update user profile for successful accesses
        if success and user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile['common_resources'].add(resource)
            profile['common_hours'].add(datetime.utcnow().hour)
    
    async def _record_threat(
        self,
        event_type: SecurityEvent,
        threat_level: ThreatLevel,
        source_ip: str,
        user_id: Optional[str],
        description: str,
        evidence: Dict[str, Any]
    ):
        """Record security threat indicator"""
        
        threat = ThreatIndicator(
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            evidence=evidence
        )
        
        self.threat_indicators.append(threat)
        
        # Log based on severity
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.error(f"Security threat detected: {description} from {source_ip}")
        elif threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(f"Security event: {description} from {source_ip}")
        else:
            self.logger.info(f"Security event: {description} from {source_ip}")
    
    async def _cleanup_expired_data(self):
        """Background task to clean up expired data"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                now = datetime.utcnow()
                
                with self._lock:
                    
                    # Clean up expired sessions
                    expired_sessions = [
                        session_id for session_id, context in self.active_sessions.items()
                        if now - context.last_verified > timedelta(seconds=self.session_timeout * 2)
                    ]
                    
                    for session_id in expired_sessions:
                        del self.active_sessions[session_id]
                        
                    # Clean up expired IP blocks
                    expired_blocks = [
                        ip for ip, block_time in self.blocked_ips.items()
                        if now - block_time > timedelta(seconds=self.lockout_duration)
                    ]
                    
                    for ip in expired_blocks:
                        del self.blocked_ips[ip]
                        
                    # Clean up old failed attempts
                    for user_id in list(self.failed_attempts.keys()):
                        self.failed_attempts[user_id] = [
                            attempt for attempt in self.failed_attempts[user_id]
                            if now - attempt < timedelta(days=7)
                        ]
                        
                        if not self.failed_attempts[user_id]:
                            del self.failed_attempts[user_id]
                            
                    self.logger.debug(f"Cleanup completed: removed {len(expired_sessions)} sessions, {len(expired_blocks)} IP blocks")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    async def _continuous_monitoring(self):
        """Background task for continuous security monitoring"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Analyze recent threats
                await self._analyze_threat_patterns()
                
                # Update IP reputation based on recent activity
                await self._update_ip_reputation()
                
                # Check for coordinated attacks
                await self._detect_coordinated_attacks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring task: {e}")
    
    async def _analyze_threat_patterns(self):
        """Analyze patterns in recent threats"""
        
        recent_threats = [
            threat for threat in self.threat_indicators
            if datetime.utcnow() - threat.timestamp < timedelta(hours=1)
        ]
        
        if len(recent_threats) > 20:  # High threat activity
            
            # Check for threat escalation
            critical_threats = [t for t in recent_threats if t.threat_level == ThreatLevel.CRITICAL]
            if len(critical_threats) > 5:
                self.logger.critical("Critical threat escalation detected - consider emergency response")
                
            # Check for IP-based attacks
            ip_counts = defaultdict(int)
            for threat in recent_threats:
                ip_counts[threat.source_ip] += 1
                
            for ip, count in ip_counts.items():
                if count >= 10:  # Many threats from same IP
                    self.blocked_ips[ip] = datetime.utcnow()
                    self.logger.warning(f"Auto-blocked IP {ip} due to threat pattern ({count} threats)")
    
    async def _update_ip_reputation(self):
        """Update IP reputation scores based on recent activity"""
        
        # Decay reputation scores over time
        for ip in list(self.ip_reputation.keys()):
            current_score = self.ip_reputation[ip]
            # Gradual decay toward neutral (0.5)
            if current_score > 0.5:
                self.ip_reputation[ip] = max(current_score - 0.01, 0.5)
            elif current_score < 0.5:
                self.ip_reputation[ip] = min(current_score + 0.01, 0.5)
    
    async def _detect_coordinated_attacks(self):
        """Detect coordinated attacks from multiple sources"""
        
        recent_access = [
            access for access in self.access_log
            if datetime.utcnow() - access.timestamp < timedelta(minutes=10)
        ]
        
        # Check for distributed brute force
        failed_attempts = [access for access in recent_access if not access.success]
        if len(failed_attempts) > 50:  # Many failures in short time
            
            unique_ips = set(access.ip_address for access in failed_attempts)
            if len(unique_ips) > 10:  # From many different IPs
                await self._record_threat(
                    SecurityEvent.SUSPICIOUS_ACTIVITY,
                    ThreatLevel.HIGH,
                    "multiple",
                    None,
                    f"Coordinated attack detected: {len(failed_attempts)} failures from {len(unique_ips)} IPs",
                    {"failure_count": len(failed_attempts), "unique_ips": len(unique_ips)}
                )
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data"""
        
        with self._lock:
            now = datetime.utcnow()
            
            # Recent activity metrics
            recent_threats = [
                t for t in self.threat_indicators
                if now - t.timestamp < timedelta(hours=24)
            ]
            
            recent_access = [
                a for a in self.access_log
                if now - a.timestamp < timedelta(hours=24)
            ]
            
            # Calculate metrics
            threat_by_level = defaultdict(int)
            for threat in recent_threats:
                threat_by_level[threat.threat_level.value] += 1
                
            access_success_rate = (
                len([a for a in recent_access if a.success]) / max(len(recent_access), 1)
            ) * 100
            
            return {
                "overview": {
                    "active_sessions": len(self.active_sessions),
                    "blocked_ips": len(self.blocked_ips),
                    "total_threats_24h": len(recent_threats),
                    "total_access_attempts_24h": len(recent_access),
                    "access_success_rate": round(access_success_rate, 2)
                },
                "threats": {
                    "by_level": dict(threat_by_level),
                    "recent": [
                        {
                            "type": t.event_type.value,
                            "level": t.threat_level.value,
                            "source_ip": t.source_ip,
                            "description": t.description,
                            "timestamp": t.timestamp.isoformat()
                        }
                        for t in sorted(recent_threats, key=lambda x: x.timestamp, reverse=True)[:10]
                    ]
                },
                "network": {
                    "top_source_ips": {
                        ip: count for ip, count in 
                        sorted(
                            defaultdict(int, 
                                {a.ip_address: len([x for x in recent_access if x.ip_address == a.ip_address])
                                 for a in recent_access}
                            ).items(),
                            key=lambda x: x[1], reverse=True
                        )[:10]
                    },
                    "blocked_ips": list(self.blocked_ips.keys())
                },
                "users": {
                    "active_users": len(set(s.user_id for s in self.active_sessions.values())),
                    "users_with_failures": len(self.failed_attempts),
                    "top_users": {
                        user: count for user, count in
                        sorted(
                            defaultdict(int,
                                {a.user_id: len([x for x in recent_access if x.user_id == a.user_id and a.user_id])
                                 for a in recent_access if a.user_id}
                            ).items(),
                            key=lambda x: x[1], reverse=True
                        )[:10]
                    }
                },
                "timestamp": now.isoformat()
            }
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get threat intelligence summary"""
        
        with self._lock:
            
            # Analyze threat patterns
            threat_by_type = defaultdict(int)
            threat_by_ip = defaultdict(int)
            recent_threats = [
                t for t in self.threat_indicators
                if datetime.utcnow() - t.timestamp < timedelta(days=7)
            ]
            
            for threat in recent_threats:
                threat_by_type[threat.event_type.value] += 1
                threat_by_ip[threat.source_ip] += 1
                
            # Top threat sources
            top_threat_ips = sorted(threat_by_ip.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Threat trends (simplified)
            daily_threats = defaultdict(int)
            for threat in recent_threats:
                day_key = threat.timestamp.strftime("%Y-%m-%d")
                daily_threats[day_key] += 1
                
            return {
                "summary": {
                    "total_threats_7d": len(recent_threats),
                    "unique_threat_sources": len(threat_by_ip),
                    "most_common_threat": max(threat_by_type.items(), key=lambda x: x[1])[0] if threat_by_type else None
                },
                "threat_types": dict(threat_by_type),
                "top_threat_sources": [
                    {"ip": ip, "threat_count": count} 
                    for ip, count in top_threat_ips
                ],
                "daily_trends": dict(sorted(daily_threats.items())),
                "ip_reputation": {
                    ip: score for ip, score in 
                    sorted(self.ip_reputation.items(), key=lambda x: x[1])[:20]
                    if score < 0.3  # Only show low reputation IPs
                }
            }


# Global zero trust framework instance
_zero_trust_framework: Optional[ZeroTrustFramework] = None
_framework_lock = threading.Lock()


def get_zero_trust_framework(secret_key: Optional[str] = None) -> ZeroTrustFramework:
    """Get global zero trust framework instance"""
    
    global _zero_trust_framework
    
    if _zero_trust_framework is None:
        with _framework_lock:
            if _zero_trust_framework is None:
                if secret_key is None:
                    secret_key = secrets.token_urlsafe(32)
                    
                _zero_trust_framework = ZeroTrustFramework(secret_key)
                
    return _zero_trust_framework


async def initialize_zero_trust(secret_key: Optional[str] = None) -> ZeroTrustFramework:
    """Initialize and start zero trust framework"""
    
    framework = get_zero_trust_framework(secret_key)
    await framework.start_monitoring()
    return framework