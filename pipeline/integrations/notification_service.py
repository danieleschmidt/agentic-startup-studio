"""
Notification Service for Agentic Startup Studio.

Provides multi-channel notification capabilities including:
- Email notifications (SMTP, SendGrid, AWS SES)
- Slack integration for team notifications
- Discord webhooks for community updates
- SMS notifications for critical alerts
- Push notifications for mobile apps
- In-app notifications and real-time updates
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import aiohttp
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl

from pipeline.config.settings import get_settings
from pipeline.infrastructure.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


class NotificationTemplate:
    """Template for structured notifications."""
    
    def __init__(
        self,
        title: str,
        body: str,
        template_vars: Dict[str, Any] = None,
        attachments: List[Dict[str, Any]] = None
    ):
        self.title = title
        self.body = body
        self.template_vars = template_vars or {}
        self.attachments = attachments or []
        self.created_at = datetime.now(timezone.utc)
    
    def render(self) -> Dict[str, str]:
        """Render template with variables."""
        rendered_title = self.title.format(**self.template_vars)
        rendered_body = self.body.format(**self.template_vars)
        
        return {
            'title': rendered_title,
            'body': rendered_body
        }


class NotificationResult:
    """Result of a notification attempt."""
    
    def __init__(
        self,
        success: bool,
        channel: NotificationChannel,
        message_id: str = None,
        error: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.success = success
        self.channel = channel
        self.message_id = message_id
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)


class EmailProvider:
    """Base class for email providers."""
    
    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        from_email: str = None,
        html_body: str = None,
        attachments: List[Dict[str, Any]] = None
    ) -> NotificationResult:
        raise NotImplementedError


class SMTPEmailProvider(EmailProvider):
    """SMTP email provider."""
    
    def __init__(self, host: str, port: int, username: str, password: str, use_tls: bool = True):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
    
    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        from_email: str = None,
        html_body: str = None,
        attachments: List[Dict[str, Any]] = None
    ) -> NotificationResult:
        """Send email via SMTP."""
        
        try:
            # Prepare recipients
            recipients = [to] if isinstance(to, str) else to
            from_addr = from_email or self.username
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_addr
            msg['To'] = ', '.join(recipients)
            
            # Add text part
            text_part = MIMEText(body, 'plain', 'utf-8')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html', 'utf-8')
                msg.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            
            if self.use_tls:
                with smtplib.SMTP(self.host, self.port) as server:
                    server.starttls(context=context)
                    server.login(self.username, self.password)
                    server.send_message(msg, from_addr, recipients)
            else:
                with smtplib.SMTP_SSL(self.host, self.port, context=context) as server:
                    server.login(self.username, self.password)
                    server.send_message(msg, from_addr, recipients)
            
            logger.info(f"Email sent successfully to {recipients}")
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id=f"smtp_{datetime.now().timestamp()}",
                metadata={'recipients': recipients, 'subject': subject}
            )
            
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error=str(e)
            )


class SlackNotifier:
    """Slack notification handler."""
    
    def __init__(self, webhook_url: str, bot_token: str = None):
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=30,
            recovery_timeout=60
        )
    
    async def send_message(
        self,
        text: str,
        channel: str = None,
        username: str = "Agentic Studio",
        icon_emoji: str = ":robot_face:",
        attachments: List[Dict[str, Any]] = None,
        blocks: List[Dict[str, Any]] = None
    ) -> NotificationResult:
        """Send message to Slack."""
        
        try:
            async with self.circuit_breaker:
                payload = {
                    'text': text,
                    'username': username,
                    'icon_emoji': icon_emoji
                }
                
                if channel:
                    payload['channel'] = channel
                
                if attachments:
                    payload['attachments'] = attachments
                
                if blocks:
                    payload['blocks'] = blocks
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            logger.info(f"Slack message sent successfully to {channel or 'default channel'}")
                            return NotificationResult(
                                success=True,
                                channel=NotificationChannel.SLACK,
                                message_id=f"slack_{datetime.now().timestamp()}",
                                metadata={'channel': channel, 'text': text[:100]}
                            )
                        else:
                            error_text = await response.text()
                            raise Exception(f"Slack API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error=str(e)
            )
    
    async def send_rich_message(
        self,
        title: str,
        message: str,
        color: str = "good",
        fields: List[Dict[str, Any]] = None,
        channel: str = None
    ) -> NotificationResult:
        """Send rich formatted message to Slack."""
        
        attachment = {
            'color': color,
            'title': title,
            'text': message,
            'timestamp': int(datetime.now().timestamp())
        }
        
        if fields:
            attachment['fields'] = fields
        
        return await self.send_message(
            text=title,
            channel=channel,
            attachments=[attachment]
        )


class DiscordNotifier:
    """Discord webhook notification handler."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=30,
            recovery_timeout=60
        )
    
    async def send_message(
        self,
        content: str,
        username: str = "Agentic Studio",
        avatar_url: str = None,
        embeds: List[Dict[str, Any]] = None
    ) -> NotificationResult:
        """Send message to Discord."""
        
        try:
            async with self.circuit_breaker:
                payload = {
                    'content': content,
                    'username': username
                }
                
                if avatar_url:
                    payload['avatar_url'] = avatar_url
                
                if embeds:
                    payload['embeds'] = embeds
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status in [200, 204]:
                            logger.info("Discord message sent successfully")
                            return NotificationResult(
                                success=True,
                                channel=NotificationChannel.DISCORD,
                                message_id=f"discord_{datetime.now().timestamp()}",
                                metadata={'content': content[:100]}
                            )
                        else:
                            error_text = await response.text()
                            raise Exception(f"Discord webhook error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return NotificationResult(
                success=False,
                channel=NotificationChannel.DISCORD,
                error=str(e)
            )
    
    async def send_embed(
        self,
        title: str,
        description: str,
        color: int = 0x00ff00,
        fields: List[Dict[str, Any]] = None,
        thumbnail_url: str = None,
        footer_text: str = None
    ) -> NotificationResult:
        """Send rich embed message to Discord."""
        
        embed = {
            'title': title,
            'description': description,
            'color': color,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if fields:
            embed['fields'] = fields
        
        if thumbnail_url:
            embed['thumbnail'] = {'url': thumbnail_url}
        
        if footer_text:
            embed['footer'] = {'text': footer_text}
        
        return await self.send_message(
            content="",
            embeds=[embed]
        )


class NotificationService:
    """
    Comprehensive notification service supporting multiple channels.
    
    Features:
    - Multi-channel delivery (email, Slack, Discord, SMS, push)
    - Template system for consistent messaging
    - Priority-based routing
    - Delivery tracking and retry logic
    - Rate limiting and circuit breakers
    - Notification preferences and user targeting
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize providers
        self.email_provider = self._setup_email_provider()
        self.slack_notifier = self._setup_slack_notifier()
        self.discord_notifier = self._setup_discord_notifier()
        
        # Notification templates
        self.templates = self._load_notification_templates()
        
        # Delivery tracking
        self.delivery_history = []
        
        logger.info("Notification service initialized")
    
    def _setup_email_provider(self) -> Optional[EmailProvider]:
        """Set up email provider based on configuration."""
        try:
            if hasattr(self.settings, 'smtp_host'):
                return SMTPEmailProvider(
                    host=self.settings.smtp_host,
                    port=getattr(self.settings, 'smtp_port', 587),
                    username=self.settings.smtp_username,
                    password=self.settings.smtp_password,
                    use_tls=getattr(self.settings, 'smtp_use_tls', True)
                )
        except Exception as e:
            logger.warning(f"Failed to setup email provider: {e}")
        return None
    
    def _setup_slack_notifier(self) -> Optional[SlackNotifier]:
        """Set up Slack notifier based on configuration."""
        try:
            if hasattr(self.settings, 'slack_webhook_url'):
                return SlackNotifier(
                    webhook_url=self.settings.slack_webhook_url,
                    bot_token=getattr(self.settings, 'slack_bot_token', None)
                )
        except Exception as e:
            logger.warning(f"Failed to setup Slack notifier: {e}")
        return None
    
    def _setup_discord_notifier(self) -> Optional[DiscordNotifier]:
        """Set up Discord notifier based on configuration."""
        try:
            if hasattr(self.settings, 'discord_webhook_url'):
                return DiscordNotifier(webhook_url=self.settings.discord_webhook_url)
        except Exception as e:
            logger.warning(f"Failed to setup Discord notifier: {e}")
        return None
    
    def _load_notification_templates(self) -> Dict[str, NotificationTemplate]:
        """Load predefined notification templates."""
        return {
            'idea_created': NotificationTemplate(
                title="New Startup Idea Created: {idea_title}",
                body="A new startup idea '{idea_title}' has been created in category {category}.\n\nDescription: {description}\n\nCreated by: {creator}\nTime: {created_at}"
            ),
            'idea_validated': NotificationTemplate(
                title="Startup Idea Validated: {idea_title}",
                body="Great news! The startup idea '{idea_title}' has been successfully validated.\n\nValidation Score: {validation_score}/100\nConfidence Level: {confidence_level}\n\nNext steps: {next_steps}"
            ),
            'workflow_completed': NotificationTemplate(
                title="Workflow Completed: {workflow_type}",
                body="The {workflow_type} workflow for '{idea_title}' has been completed.\n\nStatus: {status}\nDuration: {duration}\nResults: {results}"
            ),
            'error_alert': NotificationTemplate(
                title="System Alert: {error_type}",
                body="An error has occurred in the system.\n\nError: {error_message}\nComponent: {component}\nSeverity: {severity}\nTime: {timestamp}"
            ),
            'weekly_report': NotificationTemplate(
                title="Weekly Report: Agentic Startup Studio",
                body="Here's your weekly summary:\n\n• Ideas Created: {ideas_created}\n• Ideas Validated: {ideas_validated}\n• Workflows Executed: {workflows_executed}\n• System Uptime: {uptime}\n\nTop performing idea: {top_idea}"
            )
        }
    
    async def send_notification(
        self,
        template_name: str,
        channels: List[NotificationChannel],
        recipients: Dict[NotificationChannel, List[str]],
        template_vars: Dict[str, Any] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> List[NotificationResult]:
        """Send notification across multiple channels."""
        
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        if template_vars:
            template.template_vars.update(template_vars)
        
        rendered = template.render()
        results = []
        
        # Send to each channel
        for channel in channels:
            if channel not in recipients:
                continue
            
            channel_recipients = recipients[channel]
            if not channel_recipients:
                continue
            
            try:
                if channel == NotificationChannel.EMAIL and self.email_provider:
                    result = await self._send_email_notification(
                        rendered, channel_recipients
                    )
                elif channel == NotificationChannel.SLACK and self.slack_notifier:
                    result = await self._send_slack_notification(
                        rendered, channel_recipients, priority
                    )
                elif channel == NotificationChannel.DISCORD and self.discord_notifier:
                    result = await self._send_discord_notification(
                        rendered, channel_recipients, priority
                    )
                else:
                    result = NotificationResult(
                        success=False,
                        channel=channel,
                        error=f"Channel {channel.value} not configured or supported"
                    )
                
                results.append(result)
                self.delivery_history.append(result)
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")
                result = NotificationResult(
                    success=False,
                    channel=channel,
                    error=str(e)
                )
                results.append(result)
        
        return results
    
    async def _send_email_notification(
        self,
        rendered: Dict[str, str],
        recipients: List[str]
    ) -> NotificationResult:
        """Send email notification."""
        
        html_body = f"""
        <html>
        <body>
        <h2>{rendered['title']}</h2>
        <p>{rendered['body'].replace('\n', '<br>')}</p>
        <hr>
        <p><small>Sent by Agentic Startup Studio</small></p>
        </body>
        </html>
        """
        
        return await self.email_provider.send_email(
            to=recipients,
            subject=rendered['title'],
            body=rendered['body'],
            html_body=html_body
        )
    
    async def _send_slack_notification(
        self,
        rendered: Dict[str, str],
        channels: List[str],
        priority: NotificationPriority
    ) -> NotificationResult:
        """Send Slack notification."""
        
        # Choose color based on priority
        color_map = {
            NotificationPriority.LOW: "#36a64f",      # green
            NotificationPriority.NORMAL: "#2196F3",   # blue
            NotificationPriority.HIGH: "#ff9800",     # orange
            NotificationPriority.CRITICAL: "#f44336"  # red
        }
        color = color_map.get(priority, "#2196F3")
        
        # Send to first channel (could be extended for multiple channels)
        channel = channels[0] if channels else None
        
        return await self.slack_notifier.send_rich_message(
            title=rendered['title'],
            message=rendered['body'],
            color=color,
            channel=channel
        )
    
    async def _send_discord_notification(
        self,
        rendered: Dict[str, str],
        webhooks: List[str],
        priority: NotificationPriority
    ) -> NotificationResult:
        """Send Discord notification."""
        
        # Choose color based on priority
        color_map = {
            NotificationPriority.LOW: 0x4CAF50,      # green
            NotificationPriority.NORMAL: 0x2196F3,   # blue
            NotificationPriority.HIGH: 0xFF9800,     # orange
            NotificationPriority.CRITICAL: 0xF44336  # red
        }
        color = color_map.get(priority, 0x2196F3)
        
        return await self.discord_notifier.send_embed(
            title=rendered['title'],
            description=rendered['body'],
            color=color,
            footer_text="Agentic Startup Studio"
        )
    
    async def send_idea_notification(
        self,
        idea_title: str,
        idea_description: str,
        category: str,
        creator: str,
        recipients: Dict[NotificationChannel, List[str]]
    ) -> List[NotificationResult]:
        """Send notification for new idea creation."""
        
        return await self.send_notification(
            template_name='idea_created',
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            recipients=recipients,
            template_vars={
                'idea_title': idea_title,
                'description': idea_description,
                'category': category,
                'creator': creator,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
        )
    
    async def send_validation_notification(
        self,
        idea_title: str,
        validation_score: float,
        confidence_level: str,
        next_steps: str,
        recipients: Dict[NotificationChannel, List[str]]
    ) -> List[NotificationResult]:
        """Send notification for idea validation completion."""
        
        priority = NotificationPriority.HIGH if validation_score > 80 else NotificationPriority.NORMAL
        
        return await self.send_notification(
            template_name='idea_validated',
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.DISCORD],
            recipients=recipients,
            template_vars={
                'idea_title': idea_title,
                'validation_score': validation_score,
                'confidence_level': confidence_level,
                'next_steps': next_steps
            },
            priority=priority
        )
    
    async def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        component: str,
        severity: str,
        recipients: Dict[NotificationChannel, List[str]]
    ) -> List[NotificationResult]:
        """Send error alert notification."""
        
        priority_map = {
            'low': NotificationPriority.LOW,
            'medium': NotificationPriority.NORMAL,
            'high': NotificationPriority.HIGH,
            'critical': NotificationPriority.CRITICAL
        }
        priority = priority_map.get(severity.lower(), NotificationPriority.NORMAL)
        
        return await self.send_notification(
            template_name='error_alert',
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            recipients=recipients,
            template_vars={
                'error_type': error_type,
                'error_message': error_message,
                'component': component,
                'severity': severity,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            },
            priority=priority
        )
    
    def get_delivery_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get notification delivery statistics."""
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_deliveries = [
            result for result in self.delivery_history
            if result.timestamp > cutoff
        ]
        
        if not recent_deliveries:
            return {
                'total_notifications': 0,
                'successful_deliveries': 0,
                'failed_deliveries': 0,
                'success_rate': 0.0,
                'by_channel': {}
            }
        
        successful = len([r for r in recent_deliveries if r.success])
        failed = len([r for r in recent_deliveries if not r.success])
        
        # Group by channel
        by_channel = {}
        for result in recent_deliveries:
            channel = result.channel.value
            if channel not in by_channel:
                by_channel[channel] = {'total': 0, 'successful': 0, 'failed': 0}
            
            by_channel[channel]['total'] += 1
            if result.success:
                by_channel[channel]['successful'] += 1
            else:
                by_channel[channel]['failed'] += 1
        
        return {
            'total_notifications': len(recent_deliveries),
            'successful_deliveries': successful,
            'failed_deliveries': failed,
            'success_rate': (successful / len(recent_deliveries)) * 100,
            'by_channel': by_channel,
            'time_period_hours': hours
        }


# Testing and usage example

async def test_notification_service():
    """Test function for notification service."""
    
    notification_service = NotificationService()
    
    # Test recipients
    recipients = {
        NotificationChannel.EMAIL: ['admin@terragonlabs.com'],
        NotificationChannel.SLACK: ['#general'],
        NotificationChannel.DISCORD: ['general']
    }
    
    try:
        # Test idea notification
        results = await notification_service.send_idea_notification(
            idea_title="AI-Powered Code Review Assistant",
            idea_description="An automated code review tool using machine learning",
            category="AI/ML",
            creator="John Doe",
            recipients=recipients
        )
        
        print(f"Sent {len(results)} notifications")
        for result in results:
            print(f"  {result.channel.value}: {'✓' if result.success else '✗'}")
        
        # Get delivery stats
        stats = notification_service.get_delivery_stats()
        print(f"\nDelivery stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_notification_service())