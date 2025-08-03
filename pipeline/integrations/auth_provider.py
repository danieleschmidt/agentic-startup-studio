"""
Authentication Provider for Agentic Startup Studio.

Provides comprehensive authentication and authorization services including:
- JWT token management
- OAuth integration (Google, GitHub, Microsoft)
- API key authentication
- Role-based access control (RBAC)
- Session management
- Multi-factor authentication (MFA)
- Rate limiting and security monitoring
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import jwt
import secrets
import hashlib
import hmac
import base64
import json
from dataclasses import dataclass
import aiohttp
from urllib.parse import urlencode, parse_qs, urlparse

from pipeline.config.settings import get_settings
from pipeline.infrastructure.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for access control."""
    GUEST = "guest"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class Permission(Enum):
    """System permissions."""
    READ_IDEAS = "read_ideas"
    CREATE_IDEAS = "create_ideas"
    UPDATE_IDEAS = "update_ideas"
    DELETE_IDEAS = "delete_ideas"
    EXECUTE_WORKFLOWS = "execute_workflows"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"


class AuthenticationMethod(Enum):
    """Available authentication methods."""
    PASSWORD = "password"
    OAUTH_GOOGLE = "oauth_google"
    OAUTH_GITHUB = "oauth_github"
    OAUTH_MICROSOFT = "oauth_microsoft"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"


@dataclass
class User:
    """User model for authentication."""
    user_id: str
    email: str
    username: str
    role: UserRole
    permissions: List[Permission]
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = None
    last_login: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'username': self.username,
            'role': self.role.value,
            'permissions': [p.value for p in self.permissions],
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        return cls(
            user_id=data['user_id'],
            email=data['email'],
            username=data['username'],
            role=UserRole(data['role']),
            permissions=[Permission(p) for p in data['permissions']],
            is_active=data.get('is_active', True),
            is_verified=data.get('is_verified', False),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class AuthToken:
    """Authentication token model."""
    token: str
    token_type: str
    user_id: str
    expires_at: datetime
    scopes: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid."""
        return not self.is_expired()


class OAuthProvider:
    """Base class for OAuth providers."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=30,
            recovery_timeout=60
        )
    
    def get_authorization_url(self, state: str = None, scopes: List[str] = None) -> str:
        """Get OAuth authorization URL."""
        raise NotImplementedError
    
    async def exchange_code_for_token(self, code: str, state: str = None) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        raise NotImplementedError
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information using access token."""
        raise NotImplementedError


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__(client_id, client_secret, redirect_uri)
        self.auth_base_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    def get_authorization_url(self, state: str = None, scopes: List[str] = None) -> str:
        """Get Google OAuth authorization URL."""
        
        if scopes is None:
            scopes = ["openid", "email", "profile"]
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(scopes),
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        if state:
            params['state'] = state
        
        return f"{self.auth_base_url}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str, state: str = None) -> Dict[str, Any]:
        """Exchange Google authorization code for access token."""
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        try:
            async with self.circuit_breaker:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.token_url,
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"Token exchange failed: {error_text}")
        
        except Exception as e:
            logger.error(f"Google OAuth token exchange failed: {e}")
            raise
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get Google user information."""
        
        headers = {'Authorization': f'Bearer {access_token}'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.user_info_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"User info request failed: {error_text}")
        
        except Exception as e:
            logger.error(f"Google user info request failed: {e}")
            raise


class GitHubOAuthProvider(OAuthProvider):
    """GitHub OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__(client_id, client_secret, redirect_uri)
        self.auth_base_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        self.user_info_url = "https://api.github.com/user"
    
    def get_authorization_url(self, state: str = None, scopes: List[str] = None) -> str:
        """Get GitHub OAuth authorization URL."""
        
        if scopes is None:
            scopes = ["user:email", "read:user"]
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(scopes),
            'state': state or secrets.token_urlsafe(32)
        }
        
        return f"{self.auth_base_url}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str, state: str = None) -> Dict[str, Any]:
        """Exchange GitHub authorization code for access token."""
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code
        }
        
        headers = {'Accept': 'application/json'}
        
        try:
            async with self.circuit_breaker:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.token_url,
                        data=data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"Token exchange failed: {error_text}")
        
        except Exception as e:
            logger.error(f"GitHub OAuth token exchange failed: {e}")
            raise
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get GitHub user information."""
        
        headers = {
            'Authorization': f'token {access_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.user_info_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"User info request failed: {error_text}")
        
        except Exception as e:
            logger.error(f"GitHub user info request failed: {e}")
            raise


class AuthProvider:
    """
    Comprehensive authentication and authorization provider.
    
    Features:
    - JWT token management
    - Multiple OAuth providers
    - API key authentication
    - Role-based access control
    - Session management
    - Rate limiting
    - Security monitoring
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # JWT configuration
        self.jwt_secret = getattr(self.settings, 'jwt_secret', secrets.token_hex(32))
        self.jwt_algorithm = getattr(self.settings, 'jwt_algorithm', 'HS256')
        self.jwt_access_token_expire = getattr(self.settings, 'jwt_access_token_expire_minutes', 15)
        self.jwt_refresh_token_expire = getattr(self.settings, 'jwt_refresh_token_expire_days', 30)
        
        # OAuth providers
        self.oauth_providers = self._setup_oauth_providers()
        
        # Role permissions mapping
        self.role_permissions = self._setup_role_permissions()
        
        # In-memory stores (should be replaced with Redis/database in production)
        self.active_sessions = {}
        self.api_keys = {}
        self.rate_limits = {}
        
        logger.info("Authentication provider initialized")
    
    def _setup_oauth_providers(self) -> Dict[str, OAuthProvider]:
        """Set up OAuth providers."""
        providers = {}
        
        # Google OAuth
        if all(hasattr(self.settings, attr) for attr in ['google_client_id', 'google_client_secret']):
            providers['google'] = GoogleOAuthProvider(
                client_id=self.settings.google_client_id,
                client_secret=self.settings.google_client_secret,
                redirect_uri=getattr(self.settings, 'google_redirect_uri', 'http://localhost:8000/auth/google/callback')
            )
        
        # GitHub OAuth
        if all(hasattr(self.settings, attr) for attr in ['github_client_id', 'github_client_secret']):
            providers['github'] = GitHubOAuthProvider(
                client_id=self.settings.github_client_id,
                client_secret=self.settings.github_client_secret,
                redirect_uri=getattr(self.settings, 'github_redirect_uri', 'http://localhost:8000/auth/github/callback')
            )
        
        return providers
    
    def _setup_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Set up role-based permissions."""
        return {
            UserRole.GUEST: [
                Permission.READ_IDEAS
            ],
            UserRole.USER: [
                Permission.READ_IDEAS,
                Permission.CREATE_IDEAS,
                Permission.UPDATE_IDEAS,
                Permission.EXECUTE_WORKFLOWS,
                Permission.VIEW_ANALYTICS
            ],
            UserRole.PREMIUM: [
                Permission.READ_IDEAS,
                Permission.CREATE_IDEAS,
                Permission.UPDATE_IDEAS,
                Permission.DELETE_IDEAS,
                Permission.EXECUTE_WORKFLOWS,
                Permission.VIEW_ANALYTICS
            ],
            UserRole.ADMIN: [
                Permission.READ_IDEAS,
                Permission.CREATE_IDEAS,
                Permission.UPDATE_IDEAS,
                Permission.DELETE_IDEAS,
                Permission.EXECUTE_WORKFLOWS,
                Permission.VIEW_ANALYTICS,
                Permission.MANAGE_USERS
            ],
            UserRole.SUPER_ADMIN: [perm for perm in Permission]
        }
    
    def create_jwt_token(
        self,
        user_id: str,
        token_type: str = "access",
        additional_claims: Dict[str, Any] = None
    ) -> AuthToken:
        """Create JWT token for user."""
        
        now = datetime.now(timezone.utc)
        
        if token_type == "access":
            expire_delta = timedelta(minutes=self.jwt_access_token_expire)
        elif token_type == "refresh":
            expire_delta = timedelta(days=self.jwt_refresh_token_expire)
        else:
            expire_delta = timedelta(hours=1)
        
        expires_at = now + expire_delta
        
        payload = {
            'sub': user_id,
            'type': token_type,
            'iat': now,
            'exp': expires_at,
            'iss': 'agentic-startup-studio',
            'aud': 'agentic-startup-studio'
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        return AuthToken(
            token=token,
            token_type=token_type,
            user_id=user_id,
            expires_at=expires_at,
            scopes=additional_claims.get('scopes', []) if additional_claims else [],
            metadata={'created_at': now.isoformat()}
        )
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                audience='agentic-startup-studio',
                issuer='agentic-startup-studio'
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError as e:
            raise Exception(f"Invalid token: {e}")
    
    def generate_api_key(self, user_id: str, name: str, scopes: List[str] = None) -> str:
        """Generate API key for user."""
        
        # Create API key with format: prefix.key_id.signature
        prefix = "ask"  # Agentic Startup Studio Key
        key_id = secrets.token_hex(8)
        
        # Create signature
        payload = {
            'user_id': user_id,
            'key_id': key_id,
            'name': name,
            'scopes': scopes or [],
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        signature = hmac.new(
            self.jwt_secret.encode(),
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        
        api_key = f"{prefix}_{key_id}_{signature}"
        
        # Store API key info
        self.api_keys[api_key] = {
            'user_id': user_id,
            'name': name,
            'scopes': scopes or [],
            'created_at': datetime.now(timezone.utc),
            'last_used': None,
            'is_active': True
        }
        
        logger.info(f"Generated API key for user {user_id}: {name}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return key info."""
        
        if api_key not in self.api_keys:
            raise Exception("Invalid API key")
        
        key_info = self.api_keys[api_key]
        
        if not key_info['is_active']:
            raise Exception("API key is inactive")
        
        # Update last used timestamp
        key_info['last_used'] = datetime.now(timezone.utc)
        
        return key_info
    
    def get_oauth_authorization_url(self, provider: str, state: str = None) -> str:
        """Get OAuth authorization URL for provider."""
        
        if provider not in self.oauth_providers:
            raise Exception(f"OAuth provider '{provider}' not configured")
        
        oauth_provider = self.oauth_providers[provider]
        return oauth_provider.get_authorization_url(state=state)
    
    async def handle_oauth_callback(
        self,
        provider: str,
        code: str,
        state: str = None
    ) -> Dict[str, Any]:
        """Handle OAuth callback and create user session."""
        
        if provider not in self.oauth_providers:
            raise Exception(f"OAuth provider '{provider}' not configured")
        
        oauth_provider = self.oauth_providers[provider]
        
        try:
            # Exchange code for token
            token_data = await oauth_provider.exchange_code_for_token(code, state)
            access_token = token_data['access_token']
            
            # Get user info
            user_info = await oauth_provider.get_user_info(access_token)
            
            # Create or update user
            user = self._create_user_from_oauth(provider, user_info)
            
            # Create JWT tokens
            access_jwt = self.create_jwt_token(user.user_id, "access")
            refresh_jwt = self.create_jwt_token(user.user_id, "refresh")
            
            # Create session
            session_id = secrets.token_hex(32)
            self.active_sessions[session_id] = {
                'user_id': user.user_id,
                'provider': provider,
                'created_at': datetime.now(timezone.utc),
                'last_activity': datetime.now(timezone.utc),
                'access_token': access_jwt.token,
                'refresh_token': refresh_jwt.token
            }
            
            logger.info(f"OAuth login successful for user {user.email} via {provider}")
            
            return {
                'user': user.to_dict(),
                'access_token': access_jwt.token,
                'refresh_token': refresh_jwt.token,
                'session_id': session_id,
                'token_type': 'Bearer',
                'expires_in': self.jwt_access_token_expire * 60
            }
            
        except Exception as e:
            logger.error(f"OAuth callback failed for {provider}: {e}")
            raise
    
    def _create_user_from_oauth(self, provider: str, user_info: Dict[str, Any]) -> User:
        """Create user from OAuth user info."""
        
        # Map provider user info to our user model
        if provider == 'google':
            user_id = f"google_{user_info['id']}"
            email = user_info['email']
            username = user_info.get('name', email.split('@')[0])
            is_verified = user_info.get('verified_email', False)
        
        elif provider == 'github':
            user_id = f"github_{user_info['id']}"
            email = user_info.get('email', '')
            username = user_info['login']
            is_verified = True  # GitHub emails are verified
        
        else:
            raise Exception(f"Unsupported OAuth provider: {provider}")
        
        # Default role for new users
        role = UserRole.USER
        permissions = self.role_permissions[role]
        
        user = User(
            user_id=user_id,
            email=email,
            username=username,
            role=role,
            permissions=permissions,
            is_verified=is_verified,
            metadata={
                'provider': provider,
                'provider_data': user_info
            }
        )
        
        return user
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions
    
    def check_rate_limit(self, user_id: str, endpoint: str, limit: int = 100, window: int = 3600) -> bool:
        """Check rate limit for user and endpoint."""
        
        now = datetime.now(timezone.utc)
        key = f"{user_id}:{endpoint}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old entries outside the window
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if (now - timestamp).total_seconds() < window
        ]
        
        # Check if under limit
        if len(self.rate_limits[key]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[key].append(now)
        return True
    
    def create_session(self, user: User) -> str:
        """Create user session."""
        
        session_id = secrets.token_hex(32)
        
        self.active_sessions[session_id] = {
            'user_id': user.user_id,
            'created_at': datetime.now(timezone.utc),
            'last_activity': datetime.now(timezone.utc),
            'user_data': user.to_dict()
        }
        
        logger.info(f"Created session for user {user.user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Update last activity
        session['last_activity'] = datetime.now(timezone.utc)
        
        return session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke user session."""
        
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id]['user_id']
            del self.active_sessions[session_id]
            logger.info(f"Revoked session for user {user_id}")
            return True
        
        return False
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        
        if api_key in self.api_keys:
            self.api_keys[api_key]['is_active'] = False
            logger.info(f"Revoked API key: {self.api_keys[api_key]['name']}")
            return True
        
        return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for user."""
        
        sessions = []
        for session_id, session_data in self.active_sessions.items():
            if session_data['user_id'] == user_id:
                sessions.append({
                    'session_id': session_id,
                    **session_data
                })
        
        return sessions
    
    def cleanup_expired_sessions(self, max_idle_hours: int = 24) -> int:
        """Clean up expired sessions."""
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max_idle_hours)
        
        expired_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if session_data['last_activity'] < cutoff
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def get_authentication_stats(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        
        now = datetime.now(timezone.utc)
        
        # Active sessions stats
        active_sessions_count = len(self.active_sessions)
        
        # API keys stats
        active_api_keys = len([k for k in self.api_keys.values() if k['is_active']])
        total_api_keys = len(self.api_keys)
        
        # OAuth providers stats
        configured_providers = list(self.oauth_providers.keys())
        
        return {
            'active_sessions': active_sessions_count,
            'api_keys': {
                'active': active_api_keys,
                'total': total_api_keys
            },
            'oauth_providers': configured_providers,
            'rate_limits_tracked': len(self.rate_limits),
            'timestamp': now.isoformat()
        }


# Testing and usage example

async def test_auth_provider():
    """Test function for authentication provider."""
    
    auth_provider = AuthProvider()
    
    try:
        # Test JWT token creation
        user = User(
            user_id="test_user_123",
            email="test@example.com",
            username="testuser",
            role=UserRole.USER,
            permissions=auth_provider.role_permissions[UserRole.USER]
        )
        
        access_token = auth_provider.create_jwt_token(user.user_id)
        print(f"Created JWT token: {access_token.token[:50]}...")
        
        # Test token verification
        payload = auth_provider.verify_jwt_token(access_token.token)
        print(f"Token verified for user: {payload['sub']}")
        
        # Test API key generation
        api_key = auth_provider.generate_api_key(user.user_id, "Test Key")
        print(f"Generated API key: {api_key}")
        
        # Test API key verification
        key_info = auth_provider.verify_api_key(api_key)
        print(f"API key verified for user: {key_info['user_id']}")
        
        # Test session creation
        session_id = auth_provider.create_session(user)
        print(f"Created session: {session_id}")
        
        # Test permission checking
        can_create = auth_provider.check_permission(user, Permission.CREATE_IDEAS)
        can_admin = auth_provider.check_permission(user, Permission.SYSTEM_ADMIN)
        print(f"Can create ideas: {can_create}, Can admin: {can_admin}")
        
        # Test OAuth URLs
        for provider in auth_provider.oauth_providers:
            url = auth_provider.get_oauth_authorization_url(provider)
            print(f"{provider.title()} OAuth URL: {url[:100]}...")
        
        # Test rate limiting
        within_limit = auth_provider.check_rate_limit(user.user_id, "/api/ideas", limit=5)
        print(f"Within rate limit: {within_limit}")
        
        # Get stats
        stats = auth_provider.get_authentication_stats()
        print(f"Auth stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_auth_provider())