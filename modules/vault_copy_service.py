import shutil
import logging
from pathlib import Path
from typing import Union
from dataclasses import dataclass
from typing_extensions import Optional


@dataclass
class VaultInfo:
    """Model for vault information."""

    exists: bool
    is_directory: bool
    file_count: int
    md_file_count: int
    path: Optional[str] = None
    error: Optional[str] = None

    @staticmethod
    def builder():
        """Create a new VaultInfoBuilder instance."""
        return VaultInfoBuilder()


class VaultInfoBuilder:
    """Builder for creating VaultInfo objects with a fluent interface."""

    def __init__(self):
        self._exists = False
        self._is_directory = False
        self._file_count = 0
        self._md_file_count = 0
        self._path = None
        self._error = None

    def vault_exists(self, exists: bool = True):
        """Set exists status."""
        self._exists = exists
        return self

    def as_directory(self, is_directory: bool = True):
        """Set directory status."""
        self._is_directory = is_directory
        return self

    def with_file_count(self, count: int):
        """Set total file count."""
        self._file_count = count
        return self

    def with_md_file_count(self, count: int):
        """Set markdown file count."""
        self._md_file_count = count
        return self

    def with_path(self, path: str):
        """Set vault path."""
        self._path = path
        return self

    def with_error(self, error: str):
        """Set error message."""
        self._error = error
        return self

    def reset_to_defaults(self):
        """Reset builder to default values."""
        self._exists = False
        self._is_directory = False
        self._file_count = 0
        self._md_file_count = 0
        self._path = None
        self._error = None
        return self

    def build(self) -> VaultInfo:
        """Build the VaultInfo object."""
        return VaultInfo(
            exists=self._exists,
            is_directory=self._is_directory,
            file_count=self._file_count,
            md_file_count=self._md_file_count,
            path=self._path,
            error=self._error,
        )


class VaultCopyService:
    """Handles copying of Obsidian vaults to working directories."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def copy_vault(
        self, source_path: Union[str, Path], destination_path: Union[str, Path]
    ) -> Path:
        """
        Copy Obsidian vault from source to destination.

        Args:
            source_path: Path to the source Obsidian vault
            destination_path: Path where to copy the vault

        Returns:
            Path to the copied vault

        Raises:
            Exception: If copying fails
        """
        source_path = Path(source_path)
        destination_path = Path(destination_path)

        self.logger.info(
            f"Starting vault copy operation: {source_path} -> {destination_path}"
        )

        try:
            self._validate_source_path(source_path)
            self._prepare_destination(destination_path)
            self._perform_copy(source_path, destination_path)

            self.logger.info(f"Successfully copied vault to {destination_path}")
            return destination_path

        except Exception as e:
            self.logger.error(f"Failed to copy vault: {str(e)}")
            raise Exception(
                f"Failed to copy vault from {source_path} to {destination_path}: {str(e)}"
            )

    def _validate_source_path(self, source_path: Path) -> None:
        """Validate that the source path exists and is a directory."""
        self.logger.debug(f"Validating source path: {source_path}")

        if not source_path.exists():
            raise Exception(f"Source vault path does not exist: {source_path}")

        if not source_path.is_dir():
            raise Exception(f"Source vault path is not a directory: {source_path}")

        self.logger.debug("Source path validation passed")

    def _prepare_destination(self, destination_path: Path) -> None:
        """Prepare the destination directory for copying."""
        self.logger.debug(f"Preparing destination: {destination_path}")

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created parent directories for: {destination_path}")

            if destination_path.exists():
                self.logger.info(f"Removing existing destination: {destination_path}")
                shutil.rmtree(destination_path)
                self.logger.debug("Existing destination removed successfully")

        except Exception as e:
            raise Exception(f"Failed to prepare destination directory: {str(e)}")

    def _perform_copy(self, source_path: Path, destination_path: Path) -> None:
        """Perform the actual copying operation."""
        self.logger.debug(
            f"Copying vault files from {source_path} to {destination_path}"
        )

        try:
            shutil.copytree(source_path, destination_path)
            self.logger.debug("Vault copying completed successfully")

        except Exception as e:
            raise Exception(f"Failed during copy operation: {str(e)}")

    def get_vault_info(self, vault_path: Union[str, Path]) -> VaultInfo:
        """
        Get information about a vault directory.

        Args:
            vault_path: Path to the vault directory

        Returns:
            VaultInfo containing vault information
        """
        vault_path = Path(vault_path)
        builder = VaultInfo.builder()

        try:
            if not vault_path.exists():
                return builder.build()

            builder.vault_exists()

            if not vault_path.is_dir():
                return builder.build()

            builder.as_directory().with_path(str(vault_path))

            all_files = list(vault_path.rglob("*"))
            file_count = sum(1 for f in all_files if f.is_file())
            md_file_count = sum(
                1 for f in all_files if f.is_file() and f.suffix.lower() == ".md"
            )

            builder.with_file_count(file_count).with_md_file_count(md_file_count)

        except Exception as e:
            self.logger.error(f"Error getting vault info for {vault_path}: {str(e)}")
            builder.reset_to_defaults().with_error(str(e))

        return builder.build()
