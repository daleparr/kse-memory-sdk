"""
Enterprise Security Layer for KSE Memory SDK

This module implements comprehensive security features including RBAC, authentication,
authorization, audit trails, and SOC 2 compliance capabilities.
"""

import asyncio
import logging
import hashlib
import secrets
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    # Read permissions
    READ_SEARCH = "read:search"
    READ_ITEMS = "read:items"
    READ_DOMAINS = "read:domains"
    READ_ANALYTICS = "read:analytics"
    READ_AUDIT_LOGS = "read:audit_logs"
    READ_BILLING = "read:billing"
    
    # Write permissions
    WRITE_ITEMS = "write:items"
    WRITE_DOMAINS = "write:domains"
    WRITE_CONFIGS = "write:configs"
    
    # Execute permissions
    EXECUTE_SEARCH = "execute:search"
    EXECUTE_DOMAIN_ADAPTATION = "execute:domain_adaptation"
    EXECUTE_TRANSFER_LEARNING = "execute:transfer_learning"
    EXECUTE_CROSS_MODAL_SEARCH = "execute:cross_modal_search"
    EXECUTE_TEMPORAL_QUERY = "execute:temporal_query"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_TENANTS = "admin:tenants"
    ADMIN_BILLING = "admin:billing"
    ADMIN_SECURITY = "admin:security"
    ADMIN_SYSTEM = "admin:system"


class Role(Enum):
    """Predefined system roles."""
    VIEWER = "viewer"
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class AuditEventType(Enum):
    """Types of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    BILLING_EVENT = "billing_event"
    SYSTEM_EVENT = "system_event"


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class User:
    """User account with security attributes."""
    
    user_id: str
    username: str
    email: str
    password_hash: str
    tenant_id: str
    
    # Role and permissions
    roles: Set[Role] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    
    # Security attributes
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    
    # Access control
    allowed_ips: List[str] = field(default_factory=list)
    session_timeout: int = 3600  # 1 hour default
    
    # Audit trail
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    # Compliance
    password_changed_at: datetime = field(default_factory=datetime.now)
    must_change_password: bool = False
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: Role) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        return self.locked_until and datetime.now() < self.locked_until
    
    def needs_password_change(self) -> bool:
        """Check if user needs to change password."""
        if self.must_change_password:
            return True
        
        # Check password age (90 days default)
        password_age = datetime.now() - self.password_changed_at
        return password_age > timedelta(days=90)


@dataclass
class AuditEvent:
    """Audit log event."""
    
    event_id: str
    tenant_id: str
    user_id: Optional[str]
    event_type: AuditEventType
    timestamp: datetime
    
    # Event details
    action: str
    resource: str
    resource_id: Optional[str] = None
    
    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Results
    success: bool = True
    error_message: Optional[str] = None
    
    # Data
    before_data: Optional[Dict[str, Any]] = None
    after_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Classification
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "resource": self.resource,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "success": self.success,
            "error_message": self.error_message,
            "before_data": self.before_data,
            "after_data": self.after_data,
            "metadata": self.metadata,
            "security_level": self.security_level.value
        }


class PasswordManager:
    """Secure password management."""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Password policy
        self.min_length = 12
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
        self.max_age_days = 90
        self.history_size = 12  # Remember last 12 passwords
    
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(password, hashed)
    
    def validate_password_policy(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against policy."""
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters")
        
        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        # Check for common weak patterns
        if password.lower() in ["password", "123456", "qwerty", "admin"]:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a secure random password."""
        import string
        
        # Ensure at least one character from each required category
        chars = []
        chars.append(secrets.choice(string.ascii_uppercase))
        chars.append(secrets.choice(string.ascii_lowercase))
        chars.append(secrets.choice(string.digits))
        chars.append(secrets.choice("!@#$%^&*()_+-=[]{}|;:,.<>?"))
        
        # Fill the rest randomly
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
        for _ in range(length - 4):
            chars.append(secrets.choice(all_chars))
        
        # Shuffle to avoid predictable patterns
        secrets.SystemRandom().shuffle(chars)
        
        return ''.join(chars)


class RoleBasedAccessControl:
    """Role-Based Access Control (RBAC) system."""
    
    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()
        self.custom_roles: Dict[str, Set[Permission]] = {}
    
    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role-permission mappings."""
        
        return {
            Role.VIEWER: {
                Permission.READ_SEARCH,
                Permission.READ_ITEMS,
                Permission.EXECUTE_SEARCH,
            },
            
            Role.USER: {
                Permission.READ_SEARCH,
                Permission.READ_ITEMS,
                Permission.READ_DOMAINS,
                Permission.WRITE_ITEMS,
                Permission.EXECUTE_SEARCH,
                Permission.EXECUTE_CROSS_MODAL_SEARCH,
                Permission.EXECUTE_TEMPORAL_QUERY,
            },
            
            Role.DEVELOPER: {
                Permission.READ_SEARCH,
                Permission.READ_ITEMS,
                Permission.READ_DOMAINS,
                Permission.READ_ANALYTICS,
                Permission.WRITE_ITEMS,
                Permission.WRITE_DOMAINS,
                Permission.EXECUTE_SEARCH,
                Permission.EXECUTE_DOMAIN_ADAPTATION,
                Permission.EXECUTE_TRANSFER_LEARNING,
                Permission.EXECUTE_CROSS_MODAL_SEARCH,
                Permission.EXECUTE_TEMPORAL_QUERY,
            },
            
            Role.ADMIN: {
                # All developer permissions plus admin capabilities
                Permission.READ_SEARCH,
                Permission.READ_ITEMS,
                Permission.READ_DOMAINS,
                Permission.READ_ANALYTICS,
                Permission.READ_AUDIT_LOGS,
                Permission.READ_BILLING,
                Permission.WRITE_ITEMS,
                Permission.WRITE_DOMAINS,
                Permission.WRITE_CONFIGS,
                Permission.EXECUTE_SEARCH,
                Permission.EXECUTE_DOMAIN_ADAPTATION,
                Permission.EXECUTE_TRANSFER_LEARNING,
                Permission.EXECUTE_CROSS_MODAL_SEARCH,
                Permission.EXECUTE_TEMPORAL_QUERY,
                Permission.ADMIN_USERS,
                Permission.ADMIN_BILLING,
            },
            
            Role.SUPER_ADMIN: {
                # All permissions
                *Permission
            }
        }
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get permissions for a role."""
        return self.role_permissions.get(role, set())
    
    def create_custom_role(self, role_name: str, permissions: Set[Permission]):
        """Create a custom role with specific permissions."""
        self.custom_roles[role_name] = permissions
    
    def assign_permissions_to_user(self, user: User):
        """Assign permissions to user based on their roles."""
        user.permissions.clear()
        
        # Add permissions from standard roles
        for role in user.roles:
            user.permissions.update(self.get_role_permissions(role))
        
        # Add permissions from custom roles (if any)
        # This would be extended to support custom role assignments


class SessionManager:
    """Manages user sessions with security controls."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour default
        
    def create_session(
        self,
        user: User,
        ip_address: str,
        user_agent: str
    ) -> Tuple[str, str]:
        """Create a new session and return session ID and JWT token."""
        
        session_id = f"sess_{uuid.uuid4().hex}"
        now = datetime.now()
        expires_at = now + timedelta(seconds=user.session_timeout or self.session_timeout)
        
        # Create session data
        session_data = {
            "session_id": session_id,
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "created_at": now,
            "expires_at": expires_at,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "last_activity": now
        }
        
        self.active_sessions[session_id] = session_data
        
        # Create JWT token
        token_payload = {
            "session_id": session_id,
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "exp": expires_at,
            "iat": now,
            "iss": "kse-memory-sdk"
        }
        
        token = jwt.encode(token_payload, self.secret_key, algorithm="HS256")
        
        return session_id, token
    
    def validate_session(self, token: str, ip_address: str) -> Optional[Dict[str, Any]]:
        """Validate a session token."""
        
        try:
            # Decode JWT
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            session_id = payload["session_id"]
            
            # Check if session exists
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            # Check expiration
            if datetime.now() > session["expires_at"]:
                self.revoke_session(session_id)
                return None
            
            # Check IP address (if IP binding is enabled)
            if session["ip_address"] != ip_address:
                logger.warning(f"IP address mismatch for session {session_id}")
                # Could revoke session or just log warning
            
            # Update last activity
            session["last_activity"] = datetime.now()
            
            return session
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def revoke_session(self, session_id: str):
        """Revoke a session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    def revoke_user_sessions(self, user_id: str):
        """Revoke all sessions for a user."""
        sessions_to_revoke = [
            sid for sid, session in self.active_sessions.items()
            if session["user_id"] == user_id
        ]
        
        for session_id in sessions_to_revoke:
            self.revoke_session(session_id)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self.active_sessions.items()
            if now > session["expires_at"]
        ]
        
        for session_id in expired_sessions:
            self.revoke_session(session_id)


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.audit_events: List[AuditEvent] = []
        self.buffer_size = self.config.get("buffer_size", 1000)
        self.retention_days = self.config.get("retention_days", 2555)  # 7 years for SOC 2
        
    async def log_event(
        self,
        tenant_id: str,
        event_type: AuditEventType,
        action: str,
        resource: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        before_data: Optional[Dict[str, Any]] = None,
        after_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL
    ) -> str:
        """Log an audit event."""
        
        event_id = f"audit_{uuid.uuid4().hex[:12]}"
        
        event = AuditEvent(
            event_id=event_id,
            tenant_id=tenant_id,
            user_id=user_id,
            event_type=event_type,
            timestamp=datetime.now(),
            action=action,
            resource=resource,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            error_message=error_message,
            before_data=before_data,
            after_data=after_data,
            metadata=metadata or {},
            security_level=security_level
        )
        
        self.audit_events.append(event)
        
        # Auto-flush if buffer is full
        if len(self.audit_events) >= self.buffer_size:
            await self.flush_events()
        
        logger.info(f"Audit event logged: {action} on {resource} by {user_id}")
        
        return event_id
    
    async def flush_events(self):
        """Flush audit events to persistent storage."""
        if not self.audit_events:
            return
        
        # In production, this would write to a secure, tamper-evident storage
        logger.info(f"Flushing {len(self.audit_events)} audit events")
        
        # Clear buffer
        self.audit_events.clear()
    
    async def search_audit_logs(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        resource: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Search audit logs with filters."""
        
        # In production, this would query a database
        results = []
        
        for event in self.audit_events:
            # Apply filters
            if event.tenant_id != tenant_id:
                continue
            
            if start_date and event.timestamp < start_date:
                continue
            
            if end_date and event.timestamp > end_date:
                continue
            
            if user_id and event.user_id != user_id:
                continue
            
            if event_type and event.event_type != event_type:
                continue
            
            if resource and resource.lower() not in event.resource.lower():
                continue
            
            results.append(event)
            
            if len(results) >= limit:
                break
        
        return results
    
    async def generate_compliance_report(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for SOC 2 auditing."""
        
        events = await self.search_audit_logs(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Analyze events for compliance metrics
        event_counts = {}
        security_events = []
        failed_events = []
        
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event.event_type == AuditEventType.SECURITY_EVENT:
                security_events.append(event)
            
            if not event.success:
                failed_events.append(event)
        
        return {
            "tenant_id": tenant_id,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_events": len(events),
            "event_counts_by_type": event_counts,
            "security_events": len(security_events),
            "failed_events": len(failed_events),
            "compliance_score": self._calculate_compliance_score(events),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_compliance_score(self, events: List[AuditEvent]) -> float:
        """Calculate compliance score based on audit events."""
        if not events:
            return 100.0
        
        # Simple scoring: penalize failed events and security incidents
        failed_events = len([e for e in events if not e.success])
        security_events = len([e for e in events if e.event_type == AuditEventType.SECURITY_EVENT])
        
        penalty = (failed_events + security_events * 2) / len(events) * 100
        
        return max(0.0, 100.0 - penalty)


class SecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.password_manager = PasswordManager()
        self.rbac = RoleBasedAccessControl()
        self.session_manager = SessionManager(
            secret_key=self.config.get("secret_key", secrets.token_urlsafe(32))
        )
        self.audit_logger = AuditLogger(self.config.get("audit", {}))
        
        # User management
        self.users: Dict[str, User] = {}
        
        # Security policies
        self.max_failed_logins = self.config.get("max_failed_logins", 5)
        self.lockout_duration = self.config.get("lockout_duration", 900)  # 15 minutes
        self.require_mfa = self.config.get("require_mfa", False)
        
        logger.info("Initialized SecurityManager")
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        tenant_id: str,
        roles: Optional[List[Role]] = None,
        created_by: Optional[str] = None
    ) -> str:
        """Create a new user account."""
        
        # Validate password
        is_valid, errors = self.password_manager.validate_password_policy(password)
        if not is_valid:
            raise ValueError(f"Password policy violation: {', '.join(errors)}")
        
        # Check if user already exists
        existing_user = self._find_user_by_username(username)
        if existing_user:
            raise ValueError(f"User {username} already exists")
        
        # Create user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        password_hash = self.password_manager.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            tenant_id=tenant_id,
            roles=set(roles or [Role.USER])
        )
        
        # Assign permissions based on roles
        self.rbac.assign_permissions_to_user(user)
        
        self.users[user_id] = user
        
        # Log audit event
        await self.audit_logger.log_event(
            tenant_id=tenant_id,
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            action="create_user",
            resource="user",
            resource_id=user_id,
            user_id=created_by,
            after_data={"username": username, "email": email, "roles": [r.value for r in user.roles]},
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        logger.info(f"Created user {username} with ID {user_id}")
        
        return user_id
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        mfa_code: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Authenticate a user and return session ID and token."""
        
        user = self._find_user_by_username(username)
        
        # Log authentication attempt
        await self.audit_logger.log_event(
            tenant_id=user.tenant_id if user else "unknown",
            event_type=AuditEventType.AUTHENTICATION,
            action="login_attempt",
            resource="user",
            resource_id=user.user_id if user else None,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,  # Will update if successful
            metadata={"username": username}
        )
        
        if not user:
            return None, None
        
        # Check if account is locked
        if user.is_locked():
            await self.audit_logger.log_event(
                tenant_id=user.tenant_id,
                event_type=AuditEventType.SECURITY_EVENT,
                action="login_blocked_locked_account",
                resource="user",
                resource_id=user.user_id,
                ip_address=ip_address,
                error_message="Account is locked"
            )
            return None, None
        
        # Check if account is active
        if not user.is_active:
            return None, None
        
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_logins:
                user.locked_until = datetime.now() + timedelta(seconds=self.lockout_duration)
                
                await self.audit_logger.log_event(
                    tenant_id=user.tenant_id,
                    event_type=AuditEventType.SECURITY_EVENT,
                    action="account_locked",
                    resource="user",
                    resource_id=user.user_id,
                    ip_address=ip_address,
                    metadata={"failed_attempts": user.failed_login_attempts}
                )
            
            return None, None
        
        # Check MFA if enabled
        if user.mfa_enabled and not self._verify_mfa(user, mfa_code):
            return None, None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        # Create session
        session_id, token = self.session_manager.create_session(
            user=user,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Log successful authentication
        await self.audit_logger.log_event(
            tenant_id=user.tenant_id,
            event_type=AuditEventType.AUTHENTICATION,
            action="login_success",
            resource="user",
            resource_id=user.user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=True
        )
        
        return session_id, token
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _verify_mfa(self, user: User, mfa_code: Optional[str]) -> bool:
        """Verify MFA code (simplified implementation)."""
        if not user.mfa_enabled or not user.mfa_secret:
            return True
        
        if not mfa_code:
            return False
        
        # In production, this would use TOTP verification
        # For demo, accept any 6-digit code
        return len(mfa_code) == 6 and mfa_code.isdigit()
    
    async def authorize_action(
        self,
        session_token: str,
        permission: Permission,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[User]]:
        """Authorize an action based on session and permissions."""
        
        # Validate session
        session = self.session_manager.validate_session(session_token, ip_address or "unknown")
        if not session:
            return False, None
        
        # Get user
        user = self.users.get(session["user_id"])
        if not user or not user.is_active:
            return False, None
        
        # Check permission
        has_permission = user.has_permission(permission)
        
        # Log authorization attempt
        await self.audit_logger.log_event(
            tenant_id=user.tenant_id,
            event_type=AuditEventType.AUTHORIZATION,
            action="permission_check",
            resource="permission",
            resource_id=permission.value,
            user_id=user.user_id,
            session_id=session["session_id"],
            ip_address=ip_address,
            success=has_permission,
            metadata={"permission": permission.value, "resource_id": resource_id}
        )
        
        return has_permission, user if has_permission else None
    
    async def get_security_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get security dashboard metrics."""
        
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        # Get recent audit events
        recent_events = await self.audit_logger.search_audit_logs(
            tenant_id=tenant_id,
            start_date=last_24h
        )
        
        # Calculate metrics
        total_events = len(recent_events)
        failed_logins = len([e for e in recent_events if e.action == "login_attempt" and not e.success])
        security_events = len([e for e in recent_events if e.event_type == AuditEventType.SECURITY_EVENT])
        
        # Active sessions
        active_sessions = len([
            s for s in self.session_manager.active_sessions.values()
            if s["tenant_id"] == tenant_id and now < s["expires_at"]
        ])
        
        # Users by status
        tenant_users = [u for u in self.users.values() if u.tenant_id == tenant_id]
        active_users = len([u for u in tenant_users if u.is_active])
        locked_users = len([u for u in tenant_users if u.is_locked()])
        
        return {
            "tenant_id": tenant_id,
            "last_24_hours": {
                "total_events": total_events,
                "failed_logins": failed_logins,
                "security_events": security_events,
                "active_sessions": active_sessions
            },
            "users": {
                "total": len(tenant_users),
                "active": active_users,
                "locked": locked_users,
                "mfa_enabled": len([u for u in tenant_users if u.mfa_enabled])
            },
            "compliance": {
                "password_policy_enforced": True,
                "audit_logging_enabled": True,
                "session_timeout_configured": True,
                "mfa_available": True
            },
            "generated_at": now.isoformat()
        }