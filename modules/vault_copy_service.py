import shutil
import logging
from pathlib import Path
from typing import Union


class VaultCopyService:
    """Handles copying of Obsidian vaults to working directories."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def copy_vault(self, source_path: Union[str, Path], destination_path: Union[str, Path]) -> Path:
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
        
        self.logger.info(f"Starting vault copy operation: {source_path} -> {destination_path}")
        
        try:
            # Validate source path
            self._validate_source_path(source_path)
            
            # Prepare destination
            self._prepare_destination(destination_path)
            
            # Perform the copy
            self._perform_copy(source_path, destination_path)
            
            self.logger.info(f"Successfully copied vault to {destination_path}")
            return destination_path
            
        except Exception as e:
            self.logger.error(f"Failed to copy vault: {str(e)}")
            raise Exception(f"Failed to copy vault from {source_path} to {destination_path}: {str(e)}")

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
            # Ensure destination parent directory exists
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created parent directories for: {destination_path}")
            
            # If destination exists, remove it first
            if destination_path.exists():
                self.logger.info(f"Removing existing destination: {destination_path}")
                shutil.rmtree(destination_path)
                self.logger.debug("Existing destination removed successfully")
                
        except Exception as e:
            raise Exception(f"Failed to prepare destination directory: {str(e)}")

    def _perform_copy(self, source_path: Path, destination_path: Path) -> None:
        """Perform the actual copying operation."""
        self.logger.debug(f"Copying vault files from {source_path} to {destination_path}")
        
        try:
            shutil.copytree(source_path, destination_path)
            self.logger.debug("Vault copying completed successfully")
            
        except Exception as e:
            raise Exception(f"Failed during copy operation: {str(e)}")

    def get_vault_info(self, vault_path: Union[str, Path]) -> dict:
        """
        Get information about a vault directory.
        
        Args:
            vault_path: Path to the vault directory
            
        Returns:
            Dictionary containing vault information
        """
        vault_path = Path(vault_path)
        
        try:
            if not vault_path.exists():
                return {"exists": False, "is_directory": False, "file_count": 0, "md_file_count": 0}
            
            if not vault_path.is_dir():
                return {"exists": True, "is_directory": False, "file_count": 0, "md_file_count": 0}
            
            # Count files
            all_files = list(vault_path.rglob("*"))
            file_count = sum(1 for f in all_files if f.is_file())
            md_file_count = sum(1 for f in all_files if f.is_file() and f.suffix.lower() == '.md')
            
            return {
                "exists": True,
                "is_directory": True,
                "file_count": file_count,
                "md_file_count": md_file_count,
                "path": str(vault_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting vault info for {vault_path}: {str(e)}")
            return {"exists": False, "is_directory": False, "file_count": 0, "md_file_count": 0, "error": str(e)} 