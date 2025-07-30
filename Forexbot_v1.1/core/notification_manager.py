"""
Enhanced Notification Manager with Multiple Channels and Advanced Features
Professional-grade notification system for trading bot
"""

import os
import logging
import smtplib
import requests
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
import threading
from dataclasses import dataclass
import time
from queue import Queue

@dataclass
class NotificationTemplate:
    """Notification template structure"""
    title_template: str
    message_template: str
    priority: str
    channels: List[str]

class EnhancedNotificationManager:
    """
    Enhanced notification manager with advanced features and reliability
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Channel configurations
        self.email_enabled = config.get('notifications.email_enabled', False)
        self.slack_enabled = config.get('notifications.slack_enabled', False)
        self.discord_enabled = config.get('notifications.discord_enabled', False)
        self.telegram_enabled = config.get('notifications.telegram_enabled', False)
        self.console_enabled = config.get('notifications.console_enabled', True)
        
        # Notification settings
        self.rate_limit_seconds = config.get('notifications.rate_limit_seconds', 10)
        self.max_retries = config.get('notifications.max_retries', 3)
        self.timeout_seconds = config.get('notifications.timeout_seconds', 10)
        
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_username = os.getenv('EMAIL_USERNAME', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        self.email_recipient = os.getenv('EMAIL_RECIPIENT', '')
        
        # Slack configuration
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
        self.slack_channel = os.getenv('SLACK_CHANNEL', '#trading-alerts')
        
        # Discord configuration
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')
        
        # Telegram configuration
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Advanced features
        self.notification_queue = Queue()
        self.notification_history = []
        self.rate_limit_tracker = {}
        self.failed_notifications = []
        
        # Threading for async notifications
        self.notification_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.notifications_sent = 0
        self.notifications_failed = 0
        self.channel_success_rates = {}
        
        # Notification templates
        self.templates = self._initialize_templates()
        
        # Start notification processor
        self._start_notification_processor()
        
        self.logger.info("Enhanced NotificationManager initialized")
        self._log_enabled_channels()

    def _initialize_templates(self) -> Dict[str, NotificationTemplate]:
        """Initialize notification templates"""
        return {
            'trade_executed': NotificationTemplate(
                title_template="🤖 Trade Executed: {direction} {symbol}",
                message_template="""
📊 **Trade Execution Details**
━━━━━━━━━━━━━━━━━━━━━
🎯 **Symbol:** {symbol}
📈 **Direction:** {direction}
💰 **Entry Price:** {entry_price:.5f}
📋 **Strategy:** {strategy}
🎲 **Confidence:** {confidence:.1%}
🎪 **Position Size:** {position_size} lots

💼 **Risk Management:**
🛑 **Stop Loss:** {stop_loss:.5f}
🎯 **Take Profit:** {take_profit:.5f}
💸 **Risk Amount:** ${risk_amount:,.2f}
💵 **Expected Profit:** ${expected_profit:,.2f}

⏰ **Execution Time:** {timestamp}
🎫 **Ticket:** {ticket}
━━━━━━━━━━━━━━━━━━━━━
                """,
                priority="normal",
                channels=["email", "slack", "console"]
            ),
            
            'position_closed': NotificationTemplate(
                title_template="🔒 Position Closed: {symbol} - P&L: ${profit:.2f}",
                message_template="""
📈 **Position Closure Summary**
━━━━━━━━━━━━━━━━━━━━━
🎫 **Ticket:** {ticket}
🎯 **Symbol:** {symbol}
💰 **Final P&L:** ${profit:.2f}
📊 **Return:** {return_percent:.2f}%
⏱️ **Hold Duration:** {hold_duration}
🎪 **Close Reason:** {close_reason}

📋 **Trade Details:**
🔽 **Entry Price:** {entry_price:.5f}
🔼 **Exit Price:** {exit_price:.5f}
📈 **Strategy:** {strategy}

⏰ **Close Time:** {close_time}
━━━━━━━━━━━━━━━━━━━━━
                """,
                priority="normal",
                channels=["email", "slack", "console"]
            ),
            
            'system_alert': NotificationTemplate(
                title_template="{priority_emoji} {title}",
                message_template="""
{message}

⏰ **Time:** {timestamp}
🤖 **Source:** Enhanced Trading Bot
🔧 **Priority:** {priority}
                """,
                priority="high",
                channels=["email", "slack", "discord", "telegram", "console"]
            ),
            
            'performance_update': NotificationTemplate(
                title_template="📊 Performance Update - {period}",
                message_template="""
📈 **Trading Performance Report**
━━━━━━━━━━━━━━━━━━━━━
📊 **Overall Statistics:**
💰 **Account Balance:** ${account_balance:,.2f}
📈 **Total Return:** {total_return:.2f}%
📉 **Current Drawdown:** {drawdown:.2f}%
🎯 **Win Rate:** {win_rate:.1f}%

📋 **Trading Activity:**
🔄 **Total Trades:** {total_trades}
✅ **Successful Trades:** {successful_trades}
❌ **Failed Trades:** {failed_trades}
🎪 **Open Positions:** {open_positions}

🤖 **RL Performance:**
🎲 **RL Signals:** {rl_signals}
📊 **RL Success Rate:** {rl_success_rate:.1f}%
🎯 **RL Win Rate:** {rl_win_rate:.1f}%

⏰ **Report Time:** {timestamp}
━━━━━━━━━━━━━━━━━━━━━
                """,
                priority="low",
                channels=["email", "console"]
            ),
            
            'risk_alert': NotificationTemplate(
                title_template="⚠️ Risk Alert: {alert_type}",
                message_template="""
🚨 **RISK MANAGEMENT ALERT**
━━━━━━━━━━━━━━━━━━━━━
⚠️ **Alert Type:** {alert_type}
📊 **Severity:** {severity}
📋 **Details:** {details}

💼 **Current Risk Status:**
📈 **Daily Risk Used:** {daily_risk_percent:.2f}%
📊 **Portfolio Risk:** {portfolio_risk_percent:.2f}%
📉 **Current Drawdown:** {drawdown_percent:.2f}%

🔧 **Recommended Actions:**
{recommended_actions}

⏰ **Alert Time:** {timestamp}
🤖 **Source:** Enhanced Risk Manager
━━━━━━━━━━━━━━━━━━━━━
                """,
                priority="high",
                channels=["email", "slack", "discord", "telegram", "console"]
            ),
            
            'connection_status': NotificationTemplate(
                title_template="🔌 Connection Status: {status}",
                message_template="""
🔗 **Connection Status Update**
━━━━━━━━━━━━━━━━━━━━━
🔌 **Status:** {status}
🏦 **MT5 Connection:** {mt5_status}
📊 **Data Handler:** {data_status}
🤖 **RL Model:** {rl_status}

📋 **Details:**
{details}

⏰ **Status Time:** {timestamp}
━━━━━━━━━━━━━━━━━━━━━
                """,
                priority="normal",
                channels=["email", "slack", "console"]
            )
        }

    def _start_notification_processor(self):
        """Start background notification processor thread"""
        try:
            if self.notification_thread and self.notification_thread.is_alive():
                return
            
            self.notification_thread = threading.Thread(
                target=self._process_notification_queue,
                daemon=True,
                name="NotificationProcessor"
            )
            self.notification_thread.start()
            self.logger.debug("Notification processor thread started")
            
        except Exception as e:
            self.logger.error(f"Error starting notification processor: {e}")

    def _process_notification_queue(self):
        """Process notifications from queue in background"""
        while not self.stop_event.is_set():
            try:
                # Get notification from queue (timeout to allow checking stop_event)
                try:
                    notification = self.notification_queue.get(timeout=1)
                except:
                    continue  # Timeout, check stop_event and continue
                
                # Process the notification
                self._send_notification_internal(notification)
                
                # Mark task as done
                self.notification_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in notification processor: {e}")
                time.sleep(1)

    def send_trade_notification(self, signal: Dict[str, Any], risk_params: Dict[str, Any], 
                              execution_result: Dict[str, Any]) -> None:
        """Send trade execution notification"""
        try:
            template = self.templates['trade_executed']
            
            # Prepare notification data
            notification_data = {
                'symbol': signal.get('symbol', 'Unknown'),
                'direction': signal.get('direction', 'Unknown'),
                'entry_price': execution_result.get('open_price', 0),
                'strategy': signal.get('strategy', 'Unknown'),
                'confidence': signal.get('confidence', 0),
                'position_size': execution_result.get('volume', 0),
                'stop_loss': execution_result.get('stop_loss', 0),
                'take_profit': execution_result.get('take_profit', 0),
                'risk_amount': risk_params.get('risk_amount', 0),
                'expected_profit': risk_params.get('max_gain_amount', 0),
                'timestamp': execution_result.get('timestamp', datetime.now()),
                'ticket': execution_result.get('ticket', 'N/A')
            }
            
            self._queue_notification('trade_executed', notification_data)
            
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {e}")

    def send_position_closed_notification(self, position_data: Dict[str, Any]) -> None:
        """Send position closure notification"""
        try:
            template = self.templates['position_closed']
            
            # Calculate return percentage
            risk_amount = position_data.get('risk_amount', 1)  # Avoid division by zero
            return_percent = (position_data.get('profit', 0) / risk_amount) * 100 if risk_amount > 0 else 0
            
            notification_data = {
                'ticket': position_data.get('ticket', 'N/A'),
                'symbol': position_data.get('symbol', 'Unknown'),
                'profit': position_data.get('profit', 0),
                'return_percent': return_percent,
                'hold_duration': position_data.get('hold_duration', 'Unknown'),
                'close_reason': position_data.get('close_reason', 'Unknown'),
                'entry_price': position_data.get('entry_price', 0),
                'exit_price': position_data.get('exit_price', 0),
                'strategy': position_data.get('strategy', 'Unknown'),
                'close_time': position_data.get('close_time', datetime.now())
            }
            
            self._queue_notification('position_closed', notification_data)
            
        except Exception as e:
            self.logger.error(f"Error sending position closed notification: {e}")

    def send_system_notification(self, title: str, message: str, priority: str = "normal") -> None:
        """Send system notification"""
        try:
            # Priority emojis
            priority_emojis = {
                'low': '💙',
                'normal': '💚',
                'high': '🟡',
                'critical': '🔴'
            }
            
            notification_data = {
                'title': title,
                'message': message,
                'priority': priority,
                'priority_emoji': priority_emojis.get(priority, '💚'),
                'timestamp': datetime.now()
            }
            
            self._queue_notification('system_alert', notification_data)
            
        except Exception as e:
            self.logger.error(f"Error sending system notification: {e}")

    def send_performance_notification(self, performance_data: Dict[str, Any], period: str = "Daily") -> None:
        """Send performance update notification"""
        try:
            notification_data = {
                'period': period,
                'account_balance': performance_data.get('account_balance', 0),
                'total_return': performance_data.get('total_return_percent', 0),
                'drawdown': performance_data.get('current_drawdown_percent', 0),
                'win_rate': performance_data.get('win_rate_percent', 0),
                'total_trades': performance_data.get('total_trades', 0),
                'successful_trades': performance_data.get('successful_trades', 0),
                'failed_trades': performance_data.get('failed_trades', 0),
                'open_positions': performance_data.get('open_positions_count', 0),
                'rl_signals': performance_data.get('rl_signals', 0),
                'rl_success_rate': performance_data.get('rl_success_rate', 0),
                'rl_win_rate': performance_data.get('rl_win_rate', 0),
                'timestamp': datetime.now()
            }
            
            self._queue_notification('performance_update', notification_data)
            
        except Exception as e:
            self.logger.error(f"Error sending performance notification: {e}")

    def send_risk_alert(self, alert_type: str, severity: str, details: str, 
                       risk_data: Dict[str, Any], recommended_actions: str = "") -> None:
        """Send risk management alert"""
        try:
            notification_data = {
                'alert_type': alert_type,
                'severity': severity,
                'details': details,
                'daily_risk_percent': risk_data.get('daily_risk_percent', 0),
                'portfolio_risk_percent': risk_data.get('portfolio_risk_percent', 0),
                'drawdown_percent': risk_data.get('current_drawdown_percent', 0),
                'recommended_actions': recommended_actions or "Review trading parameters and consider reducing position sizes.",
                'timestamp': datetime.now()
            }
            
            self._queue_notification('risk_alert', notification_data)
            
        except Exception as e:
            self.logger.error(f"Error sending risk alert: {e}")

    def send_connection_status(self, status: str, mt5_status: str, data_status: str, 
                             rl_status: str, details: str = "") -> None:
        """Send connection status notification"""
        try:
            notification_data = {
                'status': status,
                'mt5_status': mt5_status,
                'data_status': data_status,
                'rl_status': rl_status,
                'details': details,
                'timestamp': datetime.now()
            }
            
            self._queue_notification('connection_status', notification_data)
            
        except Exception as e:
            self.logger.error(f"Error sending connection status: {e}")

    def _queue_notification(self, template_name: str, data: Dict[str, Any]) -> None:
        """Queue notification for processing"""
        try:
            notification = {
                'template_name': template_name,
                'data': data,
                'queued_at': datetime.now(),
                'priority': self.templates[template_name].priority
            }
            
            self.notification_queue.put(notification)
            
        except Exception as e:
            self.logger.error(f"Error queuing notification: {e}")

    def _send_notification_internal(self, notification: Dict[str, Any]) -> None:
        """Internal method to send notification"""
        try:
            template_name = notification['template_name']
            data = notification['data']
            template = self.templates[template_name]
            
            # Check rate limiting
            if self._is_rate_limited(template_name):
                self.logger.debug(f"Notification {template_name} rate limited")
                return
            
            # Format title and message
            title = template.title_template.format(**data)
            message = template.message_template.format(**data)
            
            # Send to each enabled channel
            success_count = 0
            total_channels = 0
            
            for channel in template.channels:
                if self._is_channel_enabled(channel):
                    total_channels += 1
                    if self._send_to_channel(channel, title, message, template.priority):
                        success_count += 1
            
            # Update statistics
            if success_count > 0:
                self.notifications_sent += 1
                self._update_rate_limit(template_name)
                
                # Add to history
                self.notification_history.append({
                    'template': template_name,
                    'title': title,
                    'sent_at': datetime.now(),
                    'channels_sent': success_count,
                    'total_channels': total_channels
                })
                
                # Keep history manageable
                if len(self.notification_history) > 100:
                    self.notification_history = self.notification_history[-50:]
            else:
                self.notifications_failed += 1
                self.failed_notifications.append({
                    'template': template_name,
                    'title': title,
                    'failed_at': datetime.now(),
                    'reason': 'All channels failed'
                })
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            self.notifications_failed += 1

    def _is_channel_enabled(self, channel: str) -> bool:
        """Check if notification channel is enabled"""
        channel_mapping = {
            'email': self.email_enabled,
            'slack': self.slack_enabled,
            'discord': self.discord_enabled,
            'telegram': self.telegram_enabled,
            'console': self.console_enabled
        }
        return channel_mapping.get(channel, False)

    def _send_to_channel(self, channel: str, title: str, message: str, priority: str) -> bool:
        """Send notification to specific channel"""
        try:
            success = False
            
            if channel == 'email':
                success = self._send_email(title, message)
            elif channel == 'slack':
                success = self._send_slack(title, message)
            elif channel == 'discord':
                success = self._send_discord(title, message)
            elif channel == 'telegram':
                success = self._send_telegram(title, message)
            elif channel == 'console':
                success = self._send_console(title, message, priority)
            
            # Update channel success rates
            if channel not in self.channel_success_rates:
                self.channel_success_rates[channel] = {'sent': 0, 'failed': 0}
            
            if success:
                self.channel_success_rates[channel]['sent'] += 1
            else:
                self.channel_success_rates[channel]['failed'] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending to {channel}: {e}")
            return False

    def _send_email(self, title: str, message: str) -> bool:
        """Send email notification with enhanced error handling"""
        if not all([self.email_username, self.email_password, self.email_recipient]):
            self.logger.debug("Email credentials not configured")
            return False
        
        for attempt in range(self.max_retries):
            try:
                # Create message
                msg = MIMEMultipart()
                msg['From'] = self.email_username
                msg['To'] = self.email_recipient
                msg['Subject'] = f"Trading Bot: {title}"
                
                # Add body
                body = f"""
Enhanced Trading Bot Notification

{title}

{message}

---
This is an automated message from your Enhanced Trading Bot.
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                # Send email
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.email_username, self.email_password)
                    server.send_message(msg)
                
                self.logger.debug(f"Email sent successfully: {title}")
                return True
                
            except Exception as e:
                self.logger.error(f"Email notification failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False

    def _send_slack(self, title: str, message: str) -> bool:
        """Send Slack notification"""
        if not self.slack_webhook_url or self.slack_webhook_url == 'YOUR/SLACK/WEBHOOK':
            self.logger.debug("Slack webhook not configured")
            return False
        
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "channel": self.slack_channel,
                    "username": "Trading Bot",
                    "icon_emoji": ":robot_face:",
                    "text": f"*{title}*\n\n{message}"
                }
                
                response = requests.post(
                    self.slack_webhook_url,
                    json=payload,
                    timeout=self.timeout_seconds
                )
                
                if response.status_code == 200:
                    self.logger.debug(f"Slack notification sent: {title}")
                    return True
                else:
                    self.logger.warning(f"Slack notification failed: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Slack notification failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        
        return False

    def _send_discord(self, title: str, message: str) -> bool:
        """Send Discord notification"""
        if not self.discord_webhook_url:
            self.logger.debug("Discord webhook not configured")
            return False
        
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "username": "Trading Bot",
                    "avatar_url": "https://example.com/bot-avatar.png",
                    "content": f"**{title}**\n\n{message}"
                }
                
                response = requests.post(
                    self.discord_webhook_url,
                    json=payload,
                    timeout=self.timeout_seconds
                )
                
                if response.status_code == 204:
                    self.logger.debug(f"Discord notification sent: {title}")
                    return True
                else:
                    self.logger.warning(f"Discord notification failed: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Discord notification failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        
        return False

    def _send_telegram(self, title: str, message: str) -> bool:
        """Send Telegram notification"""
        if not all([self.telegram_bot_token, self.telegram_chat_id]):
            self.logger.debug("Telegram credentials not configured")
            return False
        
        for attempt in range(self.max_retries):
            try:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"*{title}*\n\n{message}",
                    "parse_mode": "Markdown"
                }
                
                response = requests.post(url, json=payload, timeout=self.timeout_seconds)
                
                if response.status_code == 200:
                    self.logger.debug(f"Telegram notification sent: {title}")
                    return True
                else:
                    self.logger.warning(f"Telegram notification failed: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Telegram notification failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        
        return False

    def _send_console(self, title: str, message: str, priority: str) -> bool:
        """Send console notification"""
        try:
            # Priority-based formatting
            priority_symbols = {
                'low': '💙',
                'normal': '💚',
                'high': '🟡',
                'critical': '🔴'
            }
            
            symbol = priority_symbols.get(priority, '💚')
            
            # Clean message for console (remove markdown formatting)
            clean_message = message.replace('**', '').replace('━', '-').replace('*', '')
            
            notification_text = f"""
{symbol} NOTIFICATION: {title}
{clean_message}
"""
            
            # Use appropriate logging level based on priority
            if priority == 'critical':
                self.logger.critical(notification_text)
            elif priority == 'high':
                self.logger.warning(notification_text)
            else:
                self.logger.info(notification_text)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Console notification failed: {e}")
            return False

    def _is_rate_limited(self, template_name: str) -> bool:
        """Check if notification is rate limited"""
        try:
            if template_name not in self.rate_limit_tracker:
                return False
            
            last_sent = self.rate_limit_tracker[template_name]
            time_elapsed = (datetime.now() - last_sent).seconds
            
            return time_elapsed < self.rate_limit_seconds
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            return False

    def _update_rate_limit(self, template_name: str) -> None:
        """Update rate limit tracker"""
        try:
            self.rate_limit_tracker[template_name] = datetime.now()
        except Exception as e:
            self.logger.error(f"Error updating rate limit: {e}")

    def _log_enabled_channels(self) -> None:
        """Log enabled notification channels"""
        try:
            enabled_channels = []
            
            if self.email_enabled and self._validate_email_config():
                enabled_channels.append("Email")
            if self.slack_enabled and self._validate_slack_config():
                enabled_channels.append("Slack")
            if self.discord_enabled and self._validate_discord_config():
                enabled_channels.append("Discord")
            if self.telegram_enabled and self._validate_telegram_config():
                enabled_channels.append("Telegram")
            if self.console_enabled:
                enabled_channels.append("Console")
            
            if enabled_channels:
                self.logger.info(f"📢 Notification channels enabled: {', '.join(enabled_channels)}")
            else:
                self.logger.warning("No notification channels configured properly")
                
        except Exception as e:
            self.logger.error(f"Error logging enabled channels: {e}")

    def _validate_email_config(self) -> bool:
        """Validate email configuration"""
        return all([
            self.email_username,
            self.email_password,
            self.email_recipient,
            self.smtp_server,
            self.smtp_port
        ])

    def _validate_slack_config(self) -> bool:
        """Validate Slack configuration"""
        return bool(self.slack_webhook_url and 
                   self.slack_webhook_url != 'YOUR/SLACK/WEBHOOK')

    def _validate_discord_config(self) -> bool:
        """Validate Discord configuration"""
        return bool(self.discord_webhook_url)

    def _validate_telegram_config(self) -> bool:
        """Validate Telegram configuration"""
        return all([self.telegram_bot_token, self.telegram_chat_id])

    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive notification statistics"""
        try:
            # Calculate channel success rates
            channel_stats = {}
            for channel, stats in self.channel_success_rates.items():
                total = stats['sent'] + stats['failed']
                success_rate = (stats['sent'] / total * 100) if total > 0 else 0
                channel_stats[channel] = {
                    'sent': stats['sent'],
                    'failed': stats['failed'],
                    'success_rate': success_rate
                }
            
            return {
                'total_sent': self.notifications_sent,
                'total_failed': self.notifications_failed,
                'overall_success_rate': (self.notifications_sent / max(1, self.notifications_sent + self.notifications_failed)) * 100,
                'channel_statistics': channel_stats,
                'enabled_channels': self._get_enabled_channels_list(),
                'queue_size': self.notification_queue.qsize(),
                'history_count': len(self.notification_history),
                'failed_notifications_count': len(self.failed_notifications),
                'rate_limit_active': len(self.rate_limit_tracker),
                'processor_running': self.notification_thread and self.notification_thread.is_alive()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting notification statistics: {e}")
            return {'error': str(e)}

    def _get_enabled_channels_list(self) -> List[str]:
        """Get list of enabled channels"""
        enabled = []
        if self.email_enabled:
            enabled.append('email')
        if self.slack_enabled:
            enabled.append('slack')
        if self.discord_enabled:
            enabled.append('discord')
        if self.telegram_enabled:
            enabled.append('telegram')
        if self.console_enabled:
            enabled.append('console')
        return enabled

    def test_notifications(self) -> Dict[str, bool]:
        """Test all enabled notification channels"""
        try:
            test_title = "🧪 Test Notification"
            test_message = """
This is a test notification from your Enhanced Trading Bot.

If you receive this message, the notification system is working correctly.

Test performed at: {timestamp}
            """.format(timestamp=datetime.now())
            
            results = {}
            
            if self.email_enabled:
                results['email'] = self._send_email(test_title, test_message)
            
            if self.slack_enabled:
                results['slack'] = self._send_slack(test_title, test_message)
            
            if self.discord_enabled:
                results['discord'] = self._send_discord(test_title, test_message)
            
            if self.telegram_enabled:
                results['telegram'] = self._send_telegram(test_title, test_message)
            
            if self.console_enabled:
                results['console'] = self._send_console(test_title, test_message, 'normal')
            
            # Log test results
            successful_channels = [channel for channel, success in results.items() if success]
            failed_channels = [channel for channel, success in results.items() if not success]
            
            if successful_channels:
                self.logger.info(f"✅ Test notifications successful: {', '.join(successful_channels)}")
            
            if failed_channels:
                self.logger.warning(f"❌ Test notifications failed: {', '.join(failed_channels)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error testing notifications: {e}")
            return {'error': str(e)}

    def shutdown(self) -> None:
        """Shutdown notification manager"""
        try:
            self.logger.info("Shutting down Enhanced NotificationManager...")
            
            # Stop processor thread
            self.stop_event.set()
            
            if self.notification_thread and self.notification_thread.is_alive():
                self.notification_thread.join(timeout=5)
                if self.notification_thread.is_alive():
                    self.logger.warning("Notification processor thread did not stop gracefully")
            
            # Process remaining notifications in queue
            remaining_notifications = 0
            while not self.notification_queue.empty():
                try:
                    notification = self.notification_queue.get_nowait()
                    self._send_notification_internal(notification)
                    remaining_notifications += 1
                except:
                    break
            
            if remaining_notifications > 0:
                self.logger.info(f"Processed {remaining_notifications} remaining notifications")
            
            # Log final statistics
            stats = self.get_notification_statistics()
            self.logger.info(f"Final Notification Stats:")
            self.logger.info(f"  Total Sent: {stats.get('total_sent', 0)}")
            self.logger.info(f"  Total Failed: {stats.get('total_failed', 0)}")
            self.logger.info(f"  Success Rate: {stats.get('overall_success_rate', 0):.1f}%")
            
            self.logger.info("✅ Enhanced NotificationManager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during notification manager shutdown: {e}")

    def force_send_notification(self, title: str, message: str, channels: List[str] = None) -> Dict[str, bool]:
        """Force send notification immediately (bypass queue and rate limiting)"""
        try:
            if channels is None:
                channels = self._get_enabled_channels_list()
            
            results = {}
            for channel in channels:
                if self._is_channel_enabled(channel):
                    results[channel] = self._send_to_channel(channel, title, message, 'high')
                else:
                    results[channel] = False
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error force sending notification: {e}")
            return {'error': str(e)}

    def clear_notification_history(self) -> bool:
        """Clear notification history and reset statistics"""
        try:
            self.notification_history.clear()
            self.failed_notifications.clear()
            self.rate_limit_tracker.clear()
            self.channel_success_rates.clear()
            self.notifications_sent = 0
            self.notifications_failed = 0
            
            self.logger.info("Notification history and statistics cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing notification history: {e}")
            return False

    def update_channel_config(self, channel: str, enabled: bool, **kwargs) -> bool:
        """Update notification channel configuration"""
        try:
            if channel == 'email':
                self.email_enabled = enabled
                if kwargs:
                    self.smtp_server = kwargs.get('smtp_server', self.smtp_server)
                    self.smtp_port = kwargs.get('smtp_port', self.smtp_port)
                    self.email_username = kwargs.get('username', self.email_username)
                    self.email_password = kwargs.get('password', self.email_password)
                    self.email_recipient = kwargs.get('recipient', self.email_recipient)
            
            elif channel == 'slack':
                self.slack_enabled = enabled
                if kwargs:
                    self.slack_webhook_url = kwargs.get('webhook_url', self.slack_webhook_url)
                    self.slack_channel = kwargs.get('channel', self.slack_channel)
            
            elif channel == 'discord':
                self.discord_enabled = enabled
                if kwargs:
                    self.discord_webhook_url = kwargs.get('webhook_url', self.discord_webhook_url)
            
            elif channel == 'telegram':
                self.telegram_enabled = enabled
                if kwargs:
                    self.telegram_bot_token = kwargs.get('bot_token', self.telegram_bot_token)
                    self.telegram_chat_id = kwargs.get('chat_id', self.telegram_chat_id)
            
            elif channel == 'console':
                self.console_enabled = enabled
            
            else:
                self.logger.warning(f"Unknown channel: {channel}")
                return False
            
            self.logger.info(f"Channel {channel} configuration updated: {'enabled' if enabled else 'disabled'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating channel config: {e}")
            return False
