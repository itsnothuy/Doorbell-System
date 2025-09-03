"""
Telegram notification module
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, List
import telegram
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Handles Telegram bot notifications"""
    
    def __init__(self):
        self.bot = None
        self.chat_id = None
        self.additional_chat_ids = []
        self.initialized = False
        
        self._load_credentials()
        
        if self.bot:
            logger.info("Telegram notifier initialized")
        else:
            logger.warning("Telegram notifier not configured")
    
    def _load_credentials(self):
        """Load Telegram credentials"""
        try:
            # Try to import credentials
            from config.credentials_telegram import (
                TELEGRAM_BOT_TOKEN, 
                TELEGRAM_CHAT_ID,
                ADDITIONAL_CHAT_IDS,
                SEND_PHOTOS,
                MAX_PHOTO_SIZE,
                RETRY_ATTEMPTS,
                RETRY_DELAY
            )
            
            # Validate credentials
            if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or not TELEGRAM_BOT_TOKEN:
                logger.error("Telegram bot token not configured")
                return
            
            if TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE" or not TELEGRAM_CHAT_ID:
                logger.error("Telegram chat ID not configured")
                return
            
            # Initialize bot
            self.bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            self.chat_id = str(TELEGRAM_CHAT_ID)
            self.additional_chat_ids = [str(cid) for cid in ADDITIONAL_CHAT_IDS]
            
            # Configuration
            self.send_photos = SEND_PHOTOS
            self.max_photo_size = MAX_PHOTO_SIZE
            self.retry_attempts = RETRY_ATTEMPTS
            self.retry_delay = RETRY_DELAY
            
            self.initialized = True
            
            # Test connection
            asyncio.run(self._test_connection())
            
        except ImportError:
            logger.error("Telegram credentials not found. Copy credentials_template.py to credentials_telegram.py")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
    
    async def _test_connection(self):
        """Test Telegram bot connection"""
        try:
            bot_info = await self.bot.get_me()
            logger.info(f"Connected to Telegram bot: {bot_info.first_name}")
            return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            self.initialized = False
            return False
    
    def send_alert(self, message: str, image_path: Optional[Path] = None, priority: str = 'normal'):
        """Send alert message with optional image"""
        if not self.initialized:
            logger.error("Telegram notifier not initialized")
            return False
        
        try:
            # Format message based on priority
            formatted_message = self._format_message(message, priority)
            
            # Send to main chat
            success = self._send_to_chat(self.chat_id, formatted_message, image_path)
            
            # Send to additional chats for urgent alerts
            if priority == 'urgent' and self.additional_chat_ids:
                for chat_id in self.additional_chat_ids:
                    self._send_to_chat(chat_id, formatted_message, image_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    def _format_message(self, message: str, priority: str) -> str:
        """Format message based on priority"""
        priority_icons = {
            'low': '🔵',
            'normal': '🟡', 
            'urgent': '🔴'
        }
        
        icon = priority_icons.get(priority, '🟡')
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        formatted = f"{icon} **DOORBELL SECURITY**\n\n"
        formatted += f"{message}\n\n"
        formatted += f"Priority: {priority.upper()}\n"
        formatted += f"Timestamp: {timestamp}"
        
        return formatted
    
    def _send_to_chat(self, chat_id: str, message: str, image_path: Optional[Path] = None) -> bool:
        """Send message to specific chat with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                if image_path and image_path.exists() and self.send_photos:
                    # Send photo with caption
                    success = asyncio.run(self._send_photo_async(chat_id, message, image_path))
                else:
                    # Send text message only
                    success = asyncio.run(self._send_message_async(chat_id, message))
                
                if success:
                    logger.info(f"Alert sent to {chat_id}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Send attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"Failed to send alert to {chat_id} after {self.retry_attempts} attempts")
        return False
    
    async def _send_message_async(self, chat_id: str, message: str) -> bool:
        """Send text message asynchronously"""
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='Markdown'
            )
            return True
        except TelegramError as e:
            logger.error(f"Telegram API error: {e}")
            return False
    
    async def _send_photo_async(self, chat_id: str, caption: str, image_path: Path) -> bool:
        """Send photo with caption asynchronously"""
        try:
            # Check file size
            if image_path.stat().st_size > self.max_photo_size:
                logger.warning(f"Image too large ({image_path.stat().st_size} bytes), sending message only")
                return await self._send_message_async(chat_id, caption)
            
            # Send photo
            with open(image_path, 'rb') as photo_file:
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode='Markdown'
                )
            
            return True
            
        except TelegramError as e:
            logger.error(f"Failed to send photo: {e}")
            # Fallback to text message
            return await self._send_message_async(chat_id, caption)
        except Exception as e:
            logger.error(f"Photo send error: {e}")
            return False
    
    def send_system_status(self, status_info: dict):
        """Send system status information"""
        try:
            message = "🔧 **SYSTEM STATUS**\n\n"
            
            for key, value in status_info.items():
                message += f"• {key}: {value}\n"
            
            self.send_alert(message, priority='low')
            
        except Exception as e:
            logger.error(f"Failed to send system status: {e}")
    
    def send_startup_notification(self):
        """Send notification when system starts"""
        try:
            message = "✅ **SYSTEM STARTED**\n\n"
            message += "Doorbell security system is now active and monitoring."
            
            self.send_alert(message, priority='low')
            
        except Exception as e:
            logger.error(f"Failed to send startup notification: {e}")
    
    def send_shutdown_notification(self):
        """Send notification when system shuts down"""
        try:
            message = "⏹️ **SYSTEM SHUTDOWN**\n\n"
            message += "Doorbell security system has been stopped."
            
            self.send_alert(message, priority='low')
            
        except Exception as e:
            logger.error(f"Failed to send shutdown notification: {e}")
    
    def test_notification(self):
        """Send test notification"""
        try:
            message = "🧪 **TEST NOTIFICATION**\n\n"
            message += "This is a test message from your doorbell security system."
            
            return self.send_alert(message, priority='low')
            
        except Exception as e:
            logger.error(f"Failed to send test notification: {e}")
            return False


class MockTelegramNotifier(TelegramNotifier):
    """Mock Telegram notifier for testing"""
    
    def __init__(self):
        """Initialize mock notifier"""
        self.initialized = True
        self.sent_messages = []
        logger.info("Mock Telegram notifier initialized")
    
    def send_alert(self, message: str, image_path: Optional[Path] = None, priority: str = 'normal'):
        """Mock send alert"""
        formatted_message = self._format_message(message, priority)
        
        alert_data = {
            'message': formatted_message,
            'image_path': str(image_path) if image_path else None,
            'priority': priority,
            'timestamp': time.time()
        }
        
        self.sent_messages.append(alert_data)
        logger.info(f"Mock alert sent: {priority} - {message[:50]}...")
        
        return True
    
    def get_sent_messages(self) -> List[dict]:
        """Get list of sent messages (for testing)"""
        return self.sent_messages.copy()
    
    def clear_messages(self):
        """Clear sent messages history"""
        self.sent_messages.clear()
