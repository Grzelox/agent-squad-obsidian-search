import pytest
import logging
from unittest.mock import Mock, patch, call
from datetime import datetime
import os
import pathlib

from modules.logger import (
    ColoredFormatter,
    setup_agent_logger,
    setup_cli_logger,
    log_verbose,
)


class TestColoredFormatter:
    """Test cases for ColoredFormatter class."""

    @pytest.fixture
    def mock_record(self):
        """Create a mock log record."""
        record = Mock()
        record.levelname = "INFO"
        record.getMessage.return_value = "Test message"
        return record

    def test_colored_formatter_initialization_default(self):
        """Test ColoredFormatter initialization with default values."""
        formatter = ColoredFormatter()

        assert formatter.verbose_mode is False
        assert hasattr(formatter, "COLORS")
        assert "GREEN" in formatter.COLORS
        assert "BLUE" in formatter.COLORS
        assert "RESET" in formatter.COLORS

    def test_colored_formatter_initialization_verbose(self):
        """Test ColoredFormatter initialization with verbose mode."""
        formatter = ColoredFormatter(verbose_mode=True)

        assert formatter.verbose_mode is True

    def test_colored_formatter_initialization_with_format(self):
        """Test ColoredFormatter initialization with custom format."""
        custom_format = "%(levelname)s - %(message)s"
        formatter = ColoredFormatter(fmt=custom_format, verbose_mode=True)

        assert formatter.verbose_mode is True

    def test_colors_defined(self):
        """Test that all required colors are defined."""
        formatter = ColoredFormatter()

        assert "GREEN" in formatter.COLORS
        assert "BLUE" in formatter.COLORS
        assert "RESET" in formatter.COLORS

        # Verify ANSI codes
        assert formatter.COLORS["GREEN"] == "\033[32m"
        assert formatter.COLORS["BLUE"] == "\033[94m"
        assert formatter.COLORS["RESET"] == "\033[0m"

    @patch("sys.stderr.isatty")
    @patch.object(logging.Formatter, "format")
    def test_format_non_tty_normal_mode(
        self, mock_super_format, mock_isatty, mock_record
    ):
        """Test format method when not in TTY and normal mode."""
        mock_isatty.return_value = False
        mock_super_format.return_value = "Formatted message"

        formatter = ColoredFormatter(verbose_mode=False)
        result = formatter.format(mock_record)

        assert result == "Formatted message"
        mock_super_format.assert_called_once_with(mock_record)

    @patch("sys.stderr.isatty")
    @patch.object(logging.Formatter, "format")
    def test_format_non_tty_verbose_mode_without_verbose_attr(
        self, mock_super_format, mock_isatty, mock_record
    ):
        """Test format method when not in TTY, verbose mode, but no verbose attribute."""
        mock_isatty.return_value = False
        mock_super_format.return_value = "Formatted message"
        # Explicitly ensure no verbose attribute
        if hasattr(mock_record, "verbose"):
            delattr(mock_record, "verbose")

        formatter = ColoredFormatter(verbose_mode=True)
        result = formatter.format(mock_record)

        assert result == "Formatted message"
        # Called once initially, no second call since no verbose attribute
        mock_super_format.assert_called_once_with(mock_record)

    @patch("sys.stderr.isatty")
    @patch.object(logging.Formatter, "format")
    def test_format_non_tty_verbose_mode_with_verbose_attr(
        self, mock_super_format, mock_isatty, mock_record
    ):
        """Test format method when not in TTY, verbose mode, with verbose attribute."""
        mock_isatty.return_value = False
        mock_super_format.side_effect = ["First call", "Second call"]
        mock_record.verbose = True
        mock_record.levelname = "DEBUG"

        formatter = ColoredFormatter(verbose_mode=True)
        result = formatter.format(mock_record)

        assert result == "Second call"
        assert mock_record.levelname == "[VERBOSE] DEBUG"
        assert mock_super_format.call_count == 2

    @patch("sys.stderr.isatty")
    @patch.object(logging.Formatter, "format")
    def test_format_tty_normal_mode(self, mock_super_format, mock_isatty, mock_record):
        """Test format method when in TTY and normal mode."""
        mock_isatty.return_value = True
        mock_super_format.return_value = "Formatted message"

        formatter = ColoredFormatter(verbose_mode=False)
        result = formatter.format(mock_record)

        expected = (
            f"{formatter.COLORS['GREEN']}Formatted message{formatter.COLORS['RESET']}"
        )
        assert result == expected

    @patch("sys.stderr.isatty")
    @patch.object(logging.Formatter, "format")
    def test_format_tty_verbose_mode_with_verbose_attr(
        self, mock_super_format, mock_isatty, mock_record
    ):
        """Test format method when in TTY, verbose mode, with verbose attribute."""
        mock_isatty.return_value = True
        mock_super_format.side_effect = ["First call", "Second call"]
        mock_record.verbose = True
        mock_record.levelname = "DEBUG"

        formatter = ColoredFormatter(verbose_mode=True)
        result = formatter.format(mock_record)

        expected = f"{formatter.COLORS['BLUE']}Second call{formatter.COLORS['RESET']}"
        assert result == expected
        assert mock_record.levelname == "[VERBOSE] DEBUG"

    @patch("sys.stderr.isatty")
    @patch.object(logging.Formatter, "format")
    def test_format_tty_verbose_mode_without_verbose_attr(
        self, mock_super_format, mock_isatty, mock_record
    ):
        """Test format method when in TTY, verbose mode, but no verbose attribute."""
        mock_isatty.return_value = True
        mock_super_format.return_value = "Formatted message"
        # Explicitly ensure no verbose attribute
        if hasattr(mock_record, "verbose"):
            delattr(mock_record, "verbose")

        formatter = ColoredFormatter(verbose_mode=True)
        result = formatter.format(mock_record)

        expected = (
            f"{formatter.COLORS['GREEN']}Formatted message{formatter.COLORS['RESET']}"
        )
        assert result == expected

    @patch("sys.stderr.isatty")
    @patch.object(logging.Formatter, "format")
    def test_format_verbose_levelname_replacement(
        self, mock_super_format, mock_isatty, mock_record
    ):
        """Test that [VERBOSE] prefix is properly handled in levelname."""
        mock_isatty.return_value = False
        mock_super_format.side_effect = ["First call", "Second call"]
        mock_record.verbose = True
        mock_record.levelname = "[VERBOSE] INFO"  # Already has prefix

        formatter = ColoredFormatter(verbose_mode=True)
        result = formatter.format(mock_record)

        # Should remove existing prefix before adding new one
        assert mock_record.levelname == "[VERBOSE] INFO"


class TestSetupAgentLogger:
    """Test cases for setup_agent_logger function."""

    @pytest.fixture
    def mock_logging(self):
        """Mock logging module components."""
        with patch("modules.logger.logging") as mock:
            mock_logger = Mock()
            mock.getLogger.return_value = mock_logger
            yield mock, mock_logger

    @patch("modules.logger.datetime")
    def test_setup_agent_logger_default_parameters(self, mock_datetime, mock_logging):
        """Test setup_agent_logger with default parameters."""
        mock_logging_module, mock_logger = mock_logging
        mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)

        mock_file_handler = Mock()
        mock_console_handler = Mock()
        mock_logging_module.FileHandler.return_value = mock_file_handler
        mock_logging_module.StreamHandler.return_value = mock_console_handler

        result = setup_agent_logger("test.log")

        # Test return value
        assert result == mock_logger

        # Test logger configuration
        mock_logging_module.getLogger.assert_called_once()
        mock_logger.setLevel.assert_called_once_with(mock_logging_module.INFO)

        # Test file handler configuration
        mock_logging_module.FileHandler.assert_called_once()
        mock_file_handler.setLevel.assert_called_once_with(mock_logging_module.DEBUG)

        # Test console handler configuration
        mock_logging_module.StreamHandler.assert_called_once()
        mock_console_handler.setLevel.assert_called_once_with(mock_logging_module.INFO)

    def test_setup_agent_logger_with_custom_parameters(self, mock_logging):
        """Test setup_agent_logger with custom parameters."""
        mock_logging_module, mock_logger = mock_logging

        setup_agent_logger(
            log_file="custom.log", agent_id="test_agent", verbose=True, quiet=True
        )

        mock_logging_module.getLogger.assert_called_once_with(
            "ObsidianAgent_test_agent"
        )
        mock_logger.setLevel.assert_called_once_with(mock_logging_module.DEBUG)

    def test_setup_agent_logger_verbose_mode(self, mock_logging):
        """Test setup_agent_logger in verbose mode."""
        mock_logging_module, mock_logger = mock_logging

        mock_file_handler = Mock()
        mock_console_handler = Mock()
        mock_logging_module.FileHandler.return_value = mock_file_handler
        mock_logging_module.StreamHandler.return_value = mock_console_handler

        result = setup_agent_logger("test.log", verbose=True)

        # Test that verbose mode sets DEBUG level
        mock_logging_module.getLogger.assert_called_once()
        mock_logger.setLevel.assert_called_once_with(mock_logging_module.DEBUG)

    def test_setup_agent_logger_quiet_mode(self, mock_logging):
        """Test setup_agent_logger in quiet mode."""
        mock_logging_module, mock_logger = mock_logging

        mock_file_handler = Mock()
        mock_logging_module.FileHandler.return_value = mock_file_handler

        result = setup_agent_logger("test.log", quiet=True)

        # Test that quiet mode uses INFO level and disables console handler
        mock_logging_module.getLogger.assert_called_once()
        mock_logger.setLevel.assert_called_once_with(mock_logging_module.INFO)

        # Verify console handler is not added in quiet mode
        mock_logging_module.StreamHandler.assert_not_called()

    @patch("modules.logger.ColoredFormatter")
    def test_setup_agent_logger_file_handler_setup(
        self, mock_colored_formatter, mock_logging
    ):
        """Test file handler setup in setup_agent_logger."""
        mock_logging_module, mock_logger = mock_logging
        mock_file_handler = Mock()
        mock_logging_module.FileHandler.return_value = mock_file_handler
        mock_file_formatter = Mock()
        mock_logging_module.Formatter.return_value = mock_file_formatter

        setup_agent_logger(log_file="test.log")

        # Verify file handler setup
        log_path = "test.log"  # No mock_path, so use direct path
        mock_logging_module.FileHandler.assert_called_once_with(log_path)
        mock_file_handler.setLevel.assert_called_once_with(mock_logging_module.DEBUG)
        mock_file_handler.setFormatter.assert_called_once_with(mock_file_formatter)
        mock_logger.addHandler.assert_any_call(mock_file_handler)

    @patch("modules.logger.ColoredFormatter")
    def test_setup_agent_logger_console_handler_setup(
        self, mock_colored_formatter, mock_logging
    ):
        """Test console handler setup in setup_agent_logger."""
        mock_logging_module, mock_logger = mock_logging
        mock_console_handler = Mock()
        mock_logging_module.StreamHandler.return_value = mock_console_handler
        mock_console_formatter = Mock()
        mock_colored_formatter.return_value = mock_console_formatter

        setup_agent_logger("test.log", verbose=True, quiet=False)

        # Verify ColoredFormatter is used for console handler with format string
        mock_colored_formatter.assert_called_once_with(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", verbose_mode=True
        )
        mock_console_handler.setFormatter.assert_called_once_with(
            mock_console_formatter
        )

    @patch("modules.logger.datetime")
    def test_setup_agent_logger_initialization_logging(
        self, mock_datetime, mock_logging
    ):
        """Test initialization logging messages."""
        mock_logging_module, mock_logger = mock_logging
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = test_time

        setup_agent_logger("test.log", verbose=True, quiet=False)

        # Verify the actual initialization messages that are logged
        expected_calls = [
            call("=" * 60),
            call("Logging initialized at 2023-01-01 12:00:00"),
            call("=" * 60),
        ]
        mock_logger.info.assert_has_calls(expected_calls, any_order=False)

    def test_setup_agent_logger_no_agent_id(self, mock_logging):
        """Test setup_agent_logger without agent_id generates unique name."""
        mock_logging_module, mock_logger = mock_logging

        with patch("builtins.id", return_value=12345):
            setup_agent_logger("test.log")

        # Verify logger name generation with id
        expected_logger_name = "ObsidianAgent_12345"
        mock_logging_module.getLogger.assert_called_once_with(expected_logger_name)


class TestSetupCliLogger:
    """Test cases for setup_cli_logger function."""

    @pytest.fixture
    def mock_logging(self):
        """Mock logging module components."""
        with patch("modules.logger.logging") as mock:
            mock_logger = Mock()
            mock.getLogger.return_value = mock_logger
            yield mock, mock_logger

    def test_setup_cli_logger_default_parameters(self, mock_logging):
        """Test setup_cli_logger with default parameters."""
        mock_logging_module, mock_logger = mock_logging

        result = setup_cli_logger()

        assert result == mock_logger
        mock_logging_module.getLogger.assert_called_once_with("ObsidianAgent_CLI")
        mock_logger.setLevel.assert_called_once_with(mock_logging_module.INFO)
        mock_logger.handlers.clear.assert_called_once()

    def test_setup_cli_logger_verbose_mode(self, mock_logging):
        """Test setup_cli_logger in verbose mode."""
        mock_logging_module, mock_logger = mock_logging
        mock_console_handler = Mock()
        mock_logging_module.StreamHandler.return_value = mock_console_handler

        result = setup_cli_logger(verbose=True)

        mock_logger.setLevel.assert_called_once_with(mock_logging_module.DEBUG)
        mock_console_handler.setLevel.assert_called_once_with(mock_logging_module.DEBUG)
        assert result.verbose_mode is True
        assert result.quiet_mode is False

    def test_setup_cli_logger_quiet_mode(self, mock_logging):
        """Test setup_cli_logger in quiet mode."""
        mock_logging_module, mock_logger = mock_logging

        result = setup_cli_logger(quiet=True)

        # Should not create console handler in quiet mode
        mock_logging_module.StreamHandler.assert_not_called()
        assert result.quiet_mode is True

    @patch("modules.logger.ColoredFormatter")
    def test_setup_cli_logger_console_handler_setup(
        self, mock_colored_formatter, mock_logging
    ):
        """Test console handler setup in setup_cli_logger."""
        mock_logging_module, mock_logger = mock_logging
        mock_console_handler = Mock()
        mock_logging_module.StreamHandler.return_value = mock_console_handler
        mock_console_formatter = Mock()
        mock_colored_formatter.return_value = mock_console_formatter

        setup_cli_logger(verbose=False, quiet=False)

        # Verify console handler setup
        mock_logging_module.StreamHandler.assert_called_once()
        mock_console_handler.setLevel.assert_called_once_with(mock_logging_module.INFO)
        mock_colored_formatter.assert_called_once_with(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", verbose_mode=False
        )
        mock_console_handler.setFormatter.assert_called_once_with(
            mock_console_formatter
        )
        mock_logger.addHandler.assert_called_once_with(mock_console_handler)


class TestLogVerbose:
    """Test cases for log_verbose helper function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock()

    def test_log_verbose_with_verbose_mode_enabled(self, mock_logger):
        """Test log_verbose when verbose mode is enabled."""
        mock_logger.verbose_mode = True
        mock_logger.quiet_mode = False

        log_verbose(mock_logger, "Test verbose message")

        mock_logger.debug.assert_called_once_with(
            "Test verbose message", extra={"verbose": True}
        )

    def test_log_verbose_with_verbose_mode_disabled(self, mock_logger):
        """Test log_verbose when verbose mode is disabled."""
        mock_logger.verbose_mode = False

        log_verbose(mock_logger, "Test verbose message")

        mock_logger.debug.assert_not_called()

    def test_log_verbose_with_quiet_mode_enabled(self, mock_logger):
        """Test log_verbose when quiet mode is enabled."""
        mock_logger.verbose_mode = True
        mock_logger.quiet_mode = True

        log_verbose(mock_logger, "Test verbose message")

        mock_logger.debug.assert_not_called()

    def test_log_verbose_without_verbose_mode_attribute(self, mock_logger):
        """Test log_verbose when logger doesn't have verbose_mode attribute."""
        # Don't set verbose_mode attribute
        del mock_logger.verbose_mode

        log_verbose(mock_logger, "Test verbose message")

        mock_logger.debug.assert_not_called()

    def test_log_verbose_without_quiet_mode_attribute(self, mock_logger):
        """Test log_verbose when logger doesn't have quiet_mode attribute."""
        mock_logger.verbose_mode = True
        # Don't set quiet_mode attribute (should default to False)
        del mock_logger.quiet_mode

        log_verbose(mock_logger, "Test verbose message")

        mock_logger.debug.assert_called_once_with(
            "Test verbose message", extra={"verbose": True}
        )

    def test_log_verbose_with_none_verbose_mode(self, mock_logger):
        """Test log_verbose when verbose_mode is None."""
        mock_logger.verbose_mode = None

        log_verbose(mock_logger, "Test verbose message")

        mock_logger.debug.assert_not_called()

    def test_log_verbose_with_empty_message(self, mock_logger):
        """Test log_verbose with empty message."""
        mock_logger.verbose_mode = True
        mock_logger.quiet_mode = False

        log_verbose(mock_logger, "")

        mock_logger.debug.assert_called_once_with("", extra={"verbose": True})

    def test_log_verbose_with_multiline_message(self, mock_logger):
        """Test log_verbose with multiline message."""
        mock_logger.verbose_mode = True
        mock_logger.quiet_mode = False
        multiline_message = "Line 1\nLine 2\nLine 3"

        log_verbose(mock_logger, multiline_message)

        mock_logger.debug.assert_called_once_with(
            multiline_message, extra={"verbose": True}
        )


class TestLoggerIntegration:
    """Integration tests for logger module."""

    @patch("modules.logger.os.makedirs")
    @patch("modules.logger.os.path.exists")
    @patch("modules.logger.logging")
    def test_agent_logger_creates_log_directory(
        self, mock_logging, mock_exists, mock_makedirs
    ):
        """Test that setup_agent_logger creates log directory."""
        mock_exists.return_value = False
        mock_logger = Mock()
        mock_logging.getLogger.return_value = mock_logger
        mock_logging.FileHandler.return_value = Mock()
        mock_logging.Formatter.return_value = Mock()

        setup_agent_logger("logs/test.log")

        mock_exists.assert_called_once_with("logs")
        mock_makedirs.assert_called_once_with("logs", exist_ok=True)

    def test_colored_formatter_colors_are_valid_ansi(self):
        """Test that ColoredFormatter uses valid ANSI color codes."""
        formatter = ColoredFormatter()

        # Test that colors are valid ANSI escape sequences
        assert formatter.COLORS["GREEN"].startswith("\033[")
        assert formatter.COLORS["BLUE"].startswith("\033[")
        assert formatter.COLORS["RESET"] == "\033[0m"

    @patch("modules.logger.os.makedirs")
    @patch("modules.logger.os.path.exists")
    @patch("modules.logger.logging")
    def test_multiple_logger_setup_calls(
        self, mock_logging, mock_exists, mock_makedirs
    ):
        """Test that multiple setup calls work correctly."""
        mock_exists.return_value = True
        mock_logger1 = Mock()
        mock_logger2 = Mock()
        mock_logging.getLogger.side_effect = [mock_logger1, mock_logger2]
        mock_logging.FileHandler.return_value = Mock()
        mock_logging.Formatter.return_value = Mock()

        setup_agent_logger("logs/test1.log")
        setup_agent_logger("logs/test2.log")

        assert mock_logging.getLogger.call_count == 2
