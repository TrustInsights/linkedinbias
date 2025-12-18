# start src/logger.py
"""Logging configuration for LinkedIn Bias Auditor.

Sets up dual logging to both file and console with consistent formatting.
"""

import logging
import sys
from pathlib import Path

from src.config import get_settings


def setup_logging() -> None:
    """Configure application logging with file and console handlers.

    Creates a logs directory if it doesn't exist and configures the root
    logger with both file and console output. Uses the log level from
    application configuration.

    The log format includes timestamp, level, and message with emoji
    indicators for different severity levels.
    """
    settings = get_settings()
    log_level = settings.app.log_level

    # Create logs directory if not exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "app.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []  # Clear any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("🟢 Logging initialized")


# end src/logger.py
