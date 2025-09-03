"""
Telegram Bot Credentials Template

Copy this file to 'credentials_telegram.py' and fill in your actual values.
Never commit the actual credentials file to version control!
"""

# Telegram Bot Configuration
# Create a bot by messaging @BotFather on Telegram
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

# Your Telegram Chat ID
# Get this by messaging @userinfobot on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# Optional: Additional chat IDs for multiple recipients
ADDITIONAL_CHAT_IDS = [
    # "FAMILY_MEMBER_CHAT_ID",
    # "EMERGENCY_CONTACT_CHAT_ID"
]

# Notification Settings
SEND_PHOTOS = True
MAX_PHOTO_SIZE = 1048576  # 1MB max
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds
