#!/usr/bin/env python3
"""
Unit tests for credentials_template module.

Tests credentials template structure and constants.
"""

import pytest
from config.credentials_template import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    ADDITIONAL_CHAT_IDS,
    SEND_PHOTOS,
    MAX_PHOTO_SIZE,
    RETRY_ATTEMPTS,
    RETRY_DELAY,
)


class TestCredentialsTemplate:
    """Test credentials template constants."""

    def test_telegram_bot_token_defined(self):
        """Test TELEGRAM_BOT_TOKEN is defined."""
        assert TELEGRAM_BOT_TOKEN is not None
        assert isinstance(TELEGRAM_BOT_TOKEN, str)

    def test_telegram_chat_id_defined(self):
        """Test TELEGRAM_CHAT_ID is defined."""
        assert TELEGRAM_CHAT_ID is not None
        assert isinstance(TELEGRAM_CHAT_ID, str)

    def test_additional_chat_ids_is_list(self):
        """Test ADDITIONAL_CHAT_IDS is a list."""
        assert isinstance(ADDITIONAL_CHAT_IDS, list)

    def test_send_photos_is_bool(self):
        """Test SEND_PHOTOS is a boolean."""
        assert isinstance(SEND_PHOTOS, bool)

    def test_max_photo_size_is_positive(self):
        """Test MAX_PHOTO_SIZE is positive."""
        assert isinstance(MAX_PHOTO_SIZE, int)
        assert MAX_PHOTO_SIZE > 0

    def test_retry_attempts_is_positive(self):
        """Test RETRY_ATTEMPTS is positive."""
        assert isinstance(RETRY_ATTEMPTS, int)
        assert RETRY_ATTEMPTS >= 0

    def test_retry_delay_is_positive(self):
        """Test RETRY_DELAY is positive."""
        assert isinstance(RETRY_DELAY, (int, float))
        assert RETRY_DELAY >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
