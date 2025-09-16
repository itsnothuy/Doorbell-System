import logging
import os
from pathlib import Path

def setup_logging(level=logging.INFO, log_file_path: Path = None):
    """Configures logging for the application.
    Args:
        level: The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
        log_file_path: Optional. Path to a file where logs should also be written.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file_path:
        # Ensure the directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)

    # Explicitly set levels for specific modules if debug is enabled
    if level == logging.DEBUG:
        logging.getLogger('src.camera_handler').setLevel(logging.DEBUG)
        logging.getLogger('src.platform_detector').setLevel(logging.DEBUG)
        logging.getLogger('src.gpio_handler').setLevel(logging.DEBUG)
        logging.getLogger('src.web_interface').setLevel(logging.DEBUG)

    # Suppress verbose logging from external libraries if not in debug mode
    if level != logging.DEBUG:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

