import pytest
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from modules.vault_copy_service import VaultCopyService, VaultInfo, VaultInfoBuilder


class TestVaultInfo:
    """Test cases for VaultInfo dataclass."""

    def test_vault_info_creation(self):
        """Test VaultInfo creation with all parameters."""
        vault_info = VaultInfo(
            exists=True,
            is_directory=True,
            file_count=10,
            md_file_count=5,
            path="/test/vault",
            error=None,
        )

        assert vault_info.exists is True
        assert vault_info.is_directory is True
        assert vault_info.file_count == 10
        assert vault_info.md_file_count == 5
        assert vault_info.path == "/test/vault"
        assert vault_info.error is None

    def test_vault_info_creation_with_defaults(self):
        """Test VaultInfo creation with minimal parameters."""
        vault_info = VaultInfo(
            exists=False, is_directory=False, file_count=0, md_file_count=0
        )

        assert vault_info.exists is False
        assert vault_info.is_directory is False
        assert vault_info.file_count == 0
        assert vault_info.md_file_count == 0
        assert vault_info.path is None
        assert vault_info.error is None

    def test_vault_info_with_error(self):
        """Test VaultInfo creation with error."""
        vault_info = VaultInfo(
            exists=False,
            is_directory=False,
            file_count=0,
            md_file_count=0,
            error="Access denied",
        )

        assert vault_info.error == "Access denied"

    def test_vault_info_builder_static_method(self):
        """Test VaultInfo.builder() static method returns VaultInfoBuilder."""
        builder = VaultInfo.builder()
        assert isinstance(builder, VaultInfoBuilder)


class TestVaultInfoBuilder:
    """Test cases for VaultInfoBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a fresh VaultInfoBuilder for each test."""
        return VaultInfoBuilder()

    def test_builder_initialization(self, builder):
        """Test builder initializes with default values."""
        vault_info = builder.build()

        assert vault_info.exists is False
        assert vault_info.is_directory is False
        assert vault_info.file_count == 0
        assert vault_info.md_file_count == 0
        assert vault_info.path is None
        assert vault_info.error is None

    def test_builder_vault_exists(self, builder):
        """Test vault_exists method."""
        result = builder.vault_exists(True)
        assert result is builder  # Should return self for chaining

        vault_info = builder.build()
        assert vault_info.exists is True

    def test_builder_vault_exists_default(self, builder):
        """Test vault_exists method with default parameter."""
        builder.vault_exists()
        vault_info = builder.build()
        assert vault_info.exists is True

    def test_builder_vault_exists_false(self, builder):
        """Test vault_exists method with False."""
        builder.vault_exists(False)
        vault_info = builder.build()
        assert vault_info.exists is False

    def test_builder_as_directory(self, builder):
        """Test as_directory method."""
        result = builder.as_directory(True)
        assert result is builder

        vault_info = builder.build()
        assert vault_info.is_directory is True

    def test_builder_as_directory_default(self, builder):
        """Test as_directory method with default parameter."""
        builder.as_directory()
        vault_info = builder.build()
        assert vault_info.is_directory is True

    def test_builder_with_file_count(self, builder):
        """Test with_file_count method."""
        result = builder.with_file_count(25)
        assert result is builder

        vault_info = builder.build()
        assert vault_info.file_count == 25

    def test_builder_with_md_file_count(self, builder):
        """Test with_md_file_count method."""
        result = builder.with_md_file_count(12)
        assert result is builder

        vault_info = builder.build()
        assert vault_info.md_file_count == 12

    def test_builder_with_path(self, builder):
        """Test with_path method."""
        result = builder.with_path("/test/vault/path")
        assert result is builder

        vault_info = builder.build()
        assert vault_info.path == "/test/vault/path"

    def test_builder_with_error(self, builder):
        """Test with_error method."""
        result = builder.with_error("Test error message")
        assert result is builder

        vault_info = builder.build()
        assert vault_info.error == "Test error message"

    def test_builder_reset_to_defaults(self, builder):
        """Test reset_to_defaults method."""
        # Set some values
        builder.vault_exists().as_directory().with_file_count(10).with_md_file_count(
            5
        ).with_path("/test").with_error("error")

        # Reset to defaults
        result = builder.reset_to_defaults()
        assert result is builder

        vault_info = builder.build()
        assert vault_info.exists is False
        assert vault_info.is_directory is False
        assert vault_info.file_count == 0
        assert vault_info.md_file_count == 0
        assert vault_info.path is None
        assert vault_info.error is None

    def test_builder_fluent_interface(self, builder):
        """Test fluent interface chaining."""
        vault_info = (
            builder.vault_exists()
            .as_directory()
            .with_file_count(20)
            .with_md_file_count(8)
            .with_path("/my/vault")
            .build()
        )

        assert vault_info.exists is True
        assert vault_info.is_directory is True
        assert vault_info.file_count == 20
        assert vault_info.md_file_count == 8
        assert vault_info.path == "/my/vault"
        assert vault_info.error is None


class TestVaultCopyService:
    """Test cases for VaultCopyService class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        return Mock()

    @pytest.fixture
    def service(self, mock_logger):
        """Create a VaultCopyService instance for testing."""
        return VaultCopyService(mock_logger)

    @pytest.fixture
    def mock_source_path(self):
        """Create a mock source path."""
        return "/test/source/vault"

    @pytest.fixture
    def mock_destination_path(self):
        """Create a mock destination path."""
        return "/test/destination/vault"

    def test_service_initialization(self, mock_logger):
        """Test VaultCopyService initialization."""
        service = VaultCopyService(mock_logger)
        assert service.logger == mock_logger

    @patch("modules.vault_copy_service.Path")
    def test_copy_vault_success(
        self, mock_path_class, service, mock_source_path, mock_destination_path
    ):
        """Test successful vault copying."""
        # Setup mocks
        mock_source = Mock()
        mock_destination = Mock()
        mock_path_class.side_effect = [mock_source, mock_destination]

        with (
            patch.object(service, "_validate_source_path") as mock_validate,
            patch.object(service, "_prepare_destination") as mock_prepare,
            patch.object(service, "_perform_copy") as mock_copy,
        ):

            result = service.copy_vault(mock_source_path, mock_destination_path)

            # Verify method calls
            mock_validate.assert_called_once_with(mock_source)
            mock_prepare.assert_called_once_with(mock_destination)
            mock_copy.assert_called_once_with(mock_source, mock_destination)

            # Verify result and logging
            assert result == mock_destination
            service.logger.info.assert_any_call(
                f"Starting vault copy operation: {mock_source} -> {mock_destination}"
            )
            service.logger.info.assert_any_call(
                f"Successfully copied vault to {mock_destination}"
            )

    @patch("modules.vault_copy_service.Path")
    def test_copy_vault_validation_error(
        self, mock_path_class, service, mock_source_path, mock_destination_path
    ):
        """Test vault copying with validation error."""
        mock_source = Mock()
        mock_destination = Mock()
        mock_path_class.side_effect = [mock_source, mock_destination]

        with patch.object(
            service, "_validate_source_path", side_effect=Exception("Source not found")
        ):
            with pytest.raises(
                Exception, match="Failed to copy vault from .* to .*: Source not found"
            ):
                service.copy_vault(mock_source_path, mock_destination_path)

            service.logger.error.assert_called_once()

    @patch("modules.vault_copy_service.Path")
    def test_copy_vault_preparation_error(
        self, mock_path_class, service, mock_source_path, mock_destination_path
    ):
        """Test vault copying with destination preparation error."""
        mock_source = Mock()
        mock_destination = Mock()
        mock_path_class.side_effect = [mock_source, mock_destination]

        with (
            patch.object(service, "_validate_source_path"),
            patch.object(
                service,
                "_prepare_destination",
                side_effect=Exception("Cannot create directory"),
            ),
        ):

            with pytest.raises(
                Exception,
                match="Failed to copy vault from .* to .*: Cannot create directory",
            ):
                service.copy_vault(mock_source_path, mock_destination_path)

    @patch("modules.vault_copy_service.Path")
    def test_copy_vault_copy_error(
        self, mock_path_class, service, mock_source_path, mock_destination_path
    ):
        """Test vault copying with copy operation error."""
        mock_source = Mock()
        mock_destination = Mock()
        mock_path_class.side_effect = [mock_source, mock_destination]

        with (
            patch.object(service, "_validate_source_path"),
            patch.object(service, "_prepare_destination"),
            patch.object(
                service, "_perform_copy", side_effect=Exception("Copy failed")
            ),
        ):

            with pytest.raises(
                Exception, match="Failed to copy vault from .* to .*: Copy failed"
            ):
                service.copy_vault(mock_source_path, mock_destination_path)

    def test_validate_source_path_success(self, service):
        """Test successful source path validation."""
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True

        # Should not raise any exception
        service._validate_source_path(mock_path)

        mock_path.exists.assert_called_once()
        mock_path.is_dir.assert_called_once()
        service.logger.debug.assert_called()

    def test_validate_source_path_not_exists(self, service):
        """Test source path validation when path doesn't exist."""
        mock_path = Mock()
        mock_path.exists.return_value = False

        with pytest.raises(Exception, match="Source vault path does not exist"):
            service._validate_source_path(mock_path)

    def test_validate_source_path_not_directory(self, service):
        """Test source path validation when path is not a directory."""
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = False

        with pytest.raises(Exception, match="Source vault path is not a directory"):
            service._validate_source_path(mock_path)

    @patch("modules.vault_copy_service.shutil")
    def test_prepare_destination_success(self, mock_shutil, service):
        """Test successful destination preparation."""
        mock_path = Mock()
        mock_path.parent = Mock()
        mock_path.exists.return_value = False

        service._prepare_destination(mock_path)

        mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_path.exists.assert_called_once()
        mock_shutil.rmtree.assert_not_called()

    @patch("modules.vault_copy_service.shutil")
    def test_prepare_destination_remove_existing(self, mock_shutil, service):
        """Test destination preparation when destination already exists."""
        mock_path = Mock()
        mock_path.parent = Mock()
        mock_path.exists.return_value = True

        service._prepare_destination(mock_path)

        mock_path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_shutil.rmtree.assert_called_once_with(mock_path)
        service.logger.info.assert_any_call(
            f"Removing existing destination: {mock_path}"
        )

    def test_prepare_destination_mkdir_error(self, service):
        """Test destination preparation when mkdir fails."""
        mock_path = Mock()
        mock_path.parent = Mock()
        mock_path.parent.mkdir.side_effect = Exception("Permission denied")

        with pytest.raises(
            Exception,
            match="Failed to prepare destination directory: Permission denied",
        ):
            service._prepare_destination(mock_path)

    @patch("modules.vault_copy_service.shutil")
    def test_prepare_destination_rmtree_error(self, mock_shutil, service):
        """Test destination preparation when rmtree fails."""
        mock_path = Mock()
        mock_path.parent = Mock()
        mock_path.exists.return_value = True
        mock_shutil.rmtree.side_effect = Exception("Cannot remove directory")

        with pytest.raises(
            Exception,
            match="Failed to prepare destination directory: Cannot remove directory",
        ):
            service._prepare_destination(mock_path)

    @patch("modules.vault_copy_service.shutil")
    def test_perform_copy_success(self, mock_shutil, service):
        """Test successful copy operation."""
        mock_source = Mock()
        mock_destination = Mock()

        service._perform_copy(mock_source, mock_destination)

        mock_shutil.copytree.assert_called_once_with(mock_source, mock_destination)
        service.logger.debug.assert_called()

    @patch("modules.vault_copy_service.shutil")
    def test_perform_copy_error(self, mock_shutil, service):
        """Test copy operation failure."""
        mock_source = Mock()
        mock_destination = Mock()
        mock_shutil.copytree.side_effect = Exception("Copy operation failed")

        with pytest.raises(
            Exception, match="Failed during copy operation: Copy operation failed"
        ):
            service._perform_copy(mock_source, mock_destination)

    @patch("modules.vault_copy_service.Path")
    def test_get_vault_info_path_not_exists(self, mock_path_class, service):
        """Test get_vault_info when path doesn't exist."""
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path

        result = service.get_vault_info("/non/existent/path")

        assert isinstance(result, VaultInfo)
        assert result.exists is False
        assert result.is_directory is False
        assert result.file_count == 0
        assert result.md_file_count == 0
        assert result.path is None
        assert result.error is None

    @patch("modules.vault_copy_service.Path")
    def test_get_vault_info_not_directory(self, mock_path_class, service):
        """Test get_vault_info when path exists but is not a directory."""
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = False
        mock_path_class.return_value = mock_path

        result = service.get_vault_info("/path/to/file")

        assert result.exists is True
        assert result.is_directory is False
        assert result.file_count == 0
        assert result.md_file_count == 0
        assert result.path is None

    @patch("modules.vault_copy_service.Path")
    def test_get_vault_info_valid_vault(self, mock_path_class, service):
        """Test get_vault_info with a valid vault directory."""
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True

        # Mock file objects
        mock_file1 = Mock()
        mock_file1.is_file.return_value = True
        mock_file1.suffix = ".md"

        mock_file2 = Mock()
        mock_file2.is_file.return_value = True
        mock_file2.suffix = ".txt"

        mock_file3 = Mock()
        mock_file3.is_file.return_value = True
        mock_file3.suffix = ".MD"  # Test case sensitivity

        mock_dir = Mock()
        mock_dir.is_file.return_value = False

        mock_path.rglob.return_value = [mock_file1, mock_file2, mock_file3, mock_dir]
        mock_path_class.return_value = mock_path

        result = service.get_vault_info("/valid/vault")

        assert result.exists is True
        assert result.is_directory is True
        assert result.file_count == 3  # Only files, not directories
        assert result.md_file_count == 2  # .md and .MD files
        assert result.path == str(mock_path)
        assert result.error is None

    @patch("modules.vault_copy_service.Path")
    def test_get_vault_info_empty_directory(self, mock_path_class, service):
        """Test get_vault_info with empty directory."""
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.rglob.return_value = []
        mock_path_class.return_value = mock_path

        result = service.get_vault_info("/empty/vault")

        assert result.exists is True
        assert result.is_directory is True
        assert result.file_count == 0
        assert result.md_file_count == 0
        assert result.path == str(mock_path)

    @patch("modules.vault_copy_service.Path")
    def test_get_vault_info_exception_handling(self, mock_path_class, service):
        """Test get_vault_info exception handling."""
        mock_path = Mock()
        mock_path.exists.side_effect = Exception("Permission denied")
        mock_path_class.return_value = mock_path

        result = service.get_vault_info("/problematic/vault")

        assert result.exists is False
        assert result.is_directory is False
        assert result.file_count == 0
        assert result.md_file_count == 0
        assert result.path is None
        assert result.error == "Permission denied"
        service.logger.error.assert_called_once()

    @patch("modules.vault_copy_service.Path")
    def test_get_vault_info_with_string_path(self, mock_path_class, service):
        """Test get_vault_info accepts string paths."""
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path

        result = service.get_vault_info("/string/path")

        mock_path_class.assert_called_once_with("/string/path")
        assert isinstance(result, VaultInfo)

    @patch("modules.vault_copy_service.Path")
    def test_get_vault_info_with_path_object(self, mock_path_class, service):
        """Test get_vault_info accepts Path objects."""
        path_obj = Path("/path/object")
        mock_path = Mock()
        mock_path.exists.return_value = False
        mock_path_class.return_value = mock_path

        result = service.get_vault_info(path_obj)

        mock_path_class.assert_called_once_with(path_obj)
        assert isinstance(result, VaultInfo)
