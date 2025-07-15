import logging
from datetime import datetime
from pathlib import Path


def setup_agent_logger(
    log_file: str = "obsidian_agent.log", agent_id: str = None
) -> logging.Logger:
    """Setup logging configuration for the agent."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Setup logger with unique name
    logger_name = f"ObsidianAgent_{agent_id or id(object())}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler for detailed logging
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log setup completion
    logger.info("=" * 60)
    logger.info(f"Logging initialized at {datetime.now()}")
    logger.info("=" * 60)

    return logger


def setup_cli_logger() -> logging.Logger:
    """Setup logging for CLI operations."""
    main_logger = logging.getLogger("ObsidianAgent_CLI")
    main_logger.setLevel(logging.INFO)

    # Add console handler if not exists
    if not main_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        main_logger.addHandler(console_handler)

    return main_logger
