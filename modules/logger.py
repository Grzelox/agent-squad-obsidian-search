import logging
from datetime import datetime
from pathlib import Path
import sys
from typing_extensions import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with simplified color scheme."""

    # ANSI color codes
    COLORS = {
        "GREEN": "\033[32m",  # Green for standard logs
        "BLUE": "\033[94m",  # Blue for verbose logs
        "RESET": "\033[0m",  # Reset to default
    }

    def __init__(self, fmt=None, verbose_mode=False):
        super().__init__(fmt)
        self.verbose_mode = verbose_mode

    def format(self, record):
        formatted = super().format(record)

        if sys.stderr.isatty():
            if self.verbose_mode and hasattr(record, "verbose") and record.verbose:
                record.levelname = (
                    f"[VERBOSE] {record.levelname.replace('[VERBOSE] ', '')}"
                )
                formatted = super().format(record)  # Re-format with updated levelname
                return f"{self.COLORS['BLUE']}{formatted}{self.COLORS['RESET']}"
            else:
                return f"{self.COLORS['GREEN']}{formatted}{self.COLORS['RESET']}"

        if self.verbose_mode and hasattr(record, "verbose") and record.verbose:
            record.levelname = f"[VERBOSE] {record.levelname.replace('[VERBOSE] ', '')}"
            formatted = super().format(record)

        return formatted


def setup_agent_logger(
    log_file: str,
    agent_id: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = False,
) -> logging.Logger:
    """Setup logging configuration for the agent."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger_name = f"ObsidianAgent_{agent_id or id(object())}"
    logger = logging.getLogger(logger_name)

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.handlers.clear()

    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if not quiet:
        console_handler = logging.StreamHandler()
        if verbose:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)

        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", verbose_mode=verbose
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.verbose_mode = verbose
    logger.quiet_mode = quiet

    if not quiet:
        logger.info("=" * 60)
        logger.info(f"Logging initialized at {datetime.now()}")
        if verbose:
            logger.debug("Verbose logging enabled", extra={"verbose": True})
        logger.info("=" * 60)

    return logger


def setup_cli_logger(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Setup logging for CLI operations."""
    main_logger = logging.getLogger("ObsidianAgent_CLI")

    if verbose:
        main_logger.setLevel(logging.DEBUG)
    else:
        main_logger.setLevel(logging.INFO)

    main_logger.handlers.clear()

    if not quiet:
        console_handler = logging.StreamHandler()
        if verbose:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)

        formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", verbose_mode=verbose
        )
        console_handler.setFormatter(formatter)
        main_logger.addHandler(console_handler)

    main_logger.verbose_mode = verbose
    main_logger.quiet_mode = quiet

    return main_logger


def log_verbose(logger, message: str):
    """Helper function to log verbose messages with special formatting."""
    if (
        hasattr(logger, "verbose_mode")
        and logger.verbose_mode
        and not getattr(logger, "quiet_mode", False)
    ):
        logger.debug(message, extra={"verbose": True})
