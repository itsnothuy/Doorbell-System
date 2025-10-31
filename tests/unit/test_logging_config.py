#!/usr/bin/env python3
"""
Unit tests for logging configuration module.

Tests logging setup, configuration, and handler management.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from config.logging_config import setup_logging


class TestLoggingConfiguration:
    """Test logging configuration functions."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_setup_logging_with_debug_level(self):
        """Test logging setup with DEBUG level."""
        setup_logging(level=logging.DEBUG)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_info_level(self):
        """Test logging setup with INFO level."""
        setup_logging(level=logging.INFO)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_logging_with_warning_level(self):
        """Test logging setup with WARNING level."""
        setup_logging(level=logging.WARNING)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_creates_console_handler(self):
        """Test that console handler is created."""
        setup_logging()
        root_logger = logging.getLogger()
        
        # Check that at least one StreamHandler exists
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) for h in root_logger.handlers
        )
        assert has_stream_handler

    def test_setup_logging_with_file_path(self, tmp_path):
        """Test logging setup with file handler."""
        log_file = tmp_path / "test.log"
        setup_logging(level=logging.INFO, log_file_path=log_file)
        
        root_logger = logging.getLogger()
        
        # Check that file handler exists
        has_file_handler = any(
            isinstance(h, logging.FileHandler) for h in root_logger.handlers
        )
        assert has_file_handler
        assert log_file.exists()

    def test_setup_logging_creates_log_directory(self, tmp_path):
        """Test that log directory is created if it doesn't exist."""
        log_dir = tmp_path / "logs" / "subdir"
        log_file = log_dir / "test.log"
        
        setup_logging(level=logging.INFO, log_file_path=log_file)
        
        assert log_dir.exists()
        assert log_file.exists()

    def test_setup_logging_clears_existing_handlers(self):
        """Test that existing handlers are cleared on setup."""
        # Add a handler
        root_logger = logging.getLogger()
        initial_handler = logging.StreamHandler()
        root_logger.addHandler(initial_handler)
        initial_count = len(root_logger.handlers)
        
        # Setup logging (should clear existing handlers)
        setup_logging()
        
        # Verify old handler is not present
        assert initial_handler not in root_logger.handlers

    def test_setup_logging_debug_mode_sets_module_levels(self):
        """Test that DEBUG mode sets specific module log levels."""
        setup_logging(level=logging.DEBUG)
        
        # Check specific modules have DEBUG level
        camera_logger = logging.getLogger('src.camera_handler')
        assert camera_logger.level == logging.DEBUG

    def test_setup_logging_non_debug_suppresses_external_libs(self):
        """Test that INFO mode suppresses external library logging."""
        setup_logging(level=logging.INFO)
        
        # Check external libraries are set to WARNING
        werkzeug_logger = logging.getLogger('werkzeug')
        assert werkzeug_logger.level == logging.WARNING
        
        urllib3_logger = logging.getLogger('urllib3')
        assert urllib3_logger.level == logging.WARNING

    def test_setup_logging_formatter_includes_timestamp(self):
        """Test that log formatter includes timestamp."""
        setup_logging()
        root_logger = logging.getLogger()
        
        # Check that handlers have formatters
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            format_str = handler.formatter._fmt
            assert '%(asctime)s' in format_str


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_logging_to_file_works(self, tmp_path):
        """Test that logging to file actually writes content."""
        log_file = tmp_path / "integration_test.log"
        setup_logging(level=logging.INFO, log_file_path=log_file)
        
        # Log a message
        test_message = "Integration test message"
        logger = logging.getLogger('test_integration')
        logger.info(test_message)
        
        # Verify file contains message
        assert log_file.exists()
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_message in content

    def test_logging_multiple_times_accumulates(self, tmp_path):
        """Test that multiple log calls accumulate in file."""
        log_file = tmp_path / "multi_test.log"
        setup_logging(level=logging.INFO, log_file_path=log_file)
        
        logger = logging.getLogger('test_multi')
        messages = ["Message 1", "Message 2", "Message 3"]
        
        for msg in messages:
            logger.info(msg)
        
        # Verify all messages are in file
        with open(log_file, 'r') as f:
            content = f.read()
            for msg in messages:
                assert msg in content

    def test_logging_different_levels(self, tmp_path):
        """Test logging different severity levels."""
        log_file = tmp_path / "levels_test.log"
        setup_logging(level=logging.DEBUG, log_file_path=log_file)
        
        logger = logging.getLogger('test_levels')
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Verify all levels are logged
        with open(log_file, 'r') as f:
            content = f.read()
            assert "DEBUG" in content
            assert "INFO" in content
            assert "WARNING" in content
            assert "ERROR" in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
