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
        self.email_enabled = getattr(config,'notifications.email_enabled', False)
        self.slack_enabled = getattr(config,'notifications.slack_enabled', False)
        self.discord_enabled = getattr(config,'notifications.discord_enabled', False)
        self.telegram_enabled = getattr(config,'notifications.telegram_enabled', False)
        self.console_enabled = getattr(config,'notifications.console_enabled', True)
        
        # Notification settings
        self.rate_limit_seconds = getattr(config,'notifications.rate_limit_seconds', 10)
        self.max_retries = getattr(config,'notifications.max_retries', 3)
        self.timeout_seconds = getattr(config,'notifications.timeout_seconds', 10)
        
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
        
        # Initialize
        self._start_notification_processor()
        
        self.logger.info("Enhanced NotificationManager initialized")
        self._log_enabled_channels()

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

    def send_error_notification(self, error_title: str, error_details: str, exception: Optional[Exception] = None) -> None:
        """Send error notifications with high priority"""
        try:
            title = f"🔴 Trading Bot Error: {error_title}"
            
            message = f"""
**Critical Error Detected**

**Error:** {error_title}
**Details:** {error_details}
"""
            
            if exception:
                message += f"\n**Exception:** {str(exception)}"
            
            message += f"""

**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Action Required:** Check bot status and logs
            """.strip()
            
            self.send_system_notification(title, message, priority='critical')
            
        except Exception as e:
            self.logger.error(f"Error sending error notification: {e}")

    def _queue_notification(self, template_name: str, data: Dict[str, Any]) -> None:
        """Queue notification for processing"""
        try:
            notification = {
                'template_name': template_name,
                'data': data,
                'queued_at': datetime.now(),
                'priority': 'normal'  # Default priority
            }
            
            self.notification_queue.put(notification)
            
        except Exception as e:
            self.logger.error(f"Error queuing notification: {e}")

    def _send_notification_internal(self, notification: Dict[str, Any]) -> None:
        """Internal method to send notification"""
        try:
            template_name = notification['template_name']
            data = notification['data']
            
            # Check rate limiting
            if self._is_rate_limited(template_name):
                self.logger.debug(f"Notification {template_name} rate limited")
                return
            
            # Format title and message based on template
            if template_name == 'trade_executed':
                title = f"🤖 Trade Executed: {data.get('direction')} {data.get('symbol')}"
                message = f"""
📊 **Trade Execution Details**
━━━━━━━━━━━━━━━━━━━━━
🎯 **Symbol:** {data.get('symbol')}
📈 **Direction:** {data.get('direction')}
💰 **Entry Price:** {data.get('entry_price', 0):.5f}
📋 **Strategy:** {data.get('strategy')}
🎲 **Confidence:** {data.get('confidence', 0):.1%}
🎪 **Position Size:** {data.get('position_size')} lots

💼 **Risk Management:**
🛑 **Stop Loss:** {data.get('stop_loss', 0):.5f}
🎯 **Take Profit:** {data.get('take_profit', 0):.5f}
💸 **Risk Amount:** ${data.get('risk_amount', 0):,.2f}
💵 **Expected Profit:** ${data.get('expected_profit', 0):,.2f}

⏰ **Execution Time:** {data.get('timestamp')}
🎫 **Ticket:** {data.get('ticket')}
━━━━━━━━━━━━━━━━━━━━━
                """.strip()
            elif template_name == 'system_alert':
                title = f"{data.get('priority_emoji', '💚')} {data.get('title')}"
                message = f"""
{data.get('message')}

⏰ **Time:** {data.get('timestamp')}
🤖 **Source:** Enhanced Trading Bot
🔧 **Priority:** {data.get('priority')}
                """.strip()
            elif template_name == 'performance_update':
                title = f"📊 Performance Update - {data.get('period')}"
                message = f"""
📈 **Trading Performance Report**
━━━━━━━━━━━━━━━━━━━━━
📊 **Overall Statistics:**
💰 **Account Balance:** ${data.get('account_balance', 0):,.2f}
📈 **Total Return:** {data.get('total_return', 0):.2f}%
📉 **Current Drawdown:** {data.get('drawdown', 0):.2f}%
🎯 **Win Rate:** {data.get('win_rate', 0):.1f}%

📋 **Trading Activity:**
🔄 **Total Trades:** {data.get('total_trades', 0)}
✅ **Successful Trades:** {data.get('successful_trades', 0)}
❌ **Failed Trades:** {data.get('failed_trades', 0)}
🎪 **Open Positions:** {data.get('open_positions', 0)}

🤖 **RL Performance:**
🎲 **RL Signals:** {data.get('rl_signals', 0)}
📊 **RL Success Rate:** {data.get('rl_success_rate', 0):.1f}%
🎯 **RL Win Rate:** {data.get('rl_win_rate', 0):.1f}%

⏰ **Report Time:** {data.get('timestamp')}
━━━━━━━━━━━━━━━━━━━━━
                """.strip()
            else:
                title = f"📢 {template_name}"
                message = str(data)
            
            # Send to enabled channels
            success_count = 0
            total_channels = 0
            
            channels = ['email', 'slack', 'discord', 'telegram', 'console']
            for channel in channels:
                if self._is_channel_enabled(channel):
                    total_channels += 1
                    if self._send_to_channel(channel, title, message, 'normal'):
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
        """Send email notification"""
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
        if not self.slack_webhook_url:
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
        return bool(self.slack_webhook_url)

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


# For backwards compatibility, create alias
NotificationManager = EnhancedNotificationManager
