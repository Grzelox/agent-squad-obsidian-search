import pytest
import os
from unittest.mock import patch, Mock

from modules.config import get_config, CONFIG_KEYS


class TestConfig:
    """Test cases for config module."""

    def test_config_keys_list(self):
        """Test that CONFIG_KEYS contains expected configuration keys."""
        expected_keys = [
            "MODEL_NAME",
            "EMBEDDING_MODEL",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "RETRIEVAL_K",
            "PERSIST_DIRECTORY",
            "COLLECTION_NAME",
        ]
        
        assert CONFIG_KEYS == expected_keys
        assert len(CONFIG_KEYS) == 7

    @patch('modules.config.os.getenv')
    def test_get_config_empty_environment(self, mock_getenv):
        """Test get_config when no environment variables are set."""
        mock_getenv.return_value = None
        
        result = get_config()
        
        assert result == {}
        assert len(mock_getenv.call_args_list) == len(CONFIG_KEYS)
        
        # Verify all config keys were checked
        called_keys = [call[0][0] for call in mock_getenv.call_args_list]
        assert set(called_keys) == set(CONFIG_KEYS)

    @patch('modules.config.os.getenv')
    def test_get_config_all_string_values(self, mock_getenv):
        """Test get_config with all string configuration values."""
        mock_values = {
            "MODEL_NAME": "test-model",
            "EMBEDDING_MODEL": "test-embedding",
            "CHUNK_SIZE": None,
            "CHUNK_OVERLAP": None,
            "RETRIEVAL_K": None,
            "PERSIST_DIRECTORY": "/test/persist",
            "COLLECTION_NAME": "test_collection",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        expected = {
            "MODEL_NAME": "test-model",
            "EMBEDDING_MODEL": "test-embedding",
            "PERSIST_DIRECTORY": "/test/persist",
            "COLLECTION_NAME": "test_collection",
        }
        assert result == expected

    @patch('modules.config.os.getenv')
    def test_get_config_numeric_values(self, mock_getenv):
        """Test get_config with numeric configuration values."""
        mock_values = {
            "MODEL_NAME": None,
            "EMBEDDING_MODEL": None,
            "CHUNK_SIZE": "1000",
            "CHUNK_OVERLAP": "200",
            "RETRIEVAL_K": "5",
            "PERSIST_DIRECTORY": None,
            "COLLECTION_NAME": None,
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        expected = {
            "CHUNK_SIZE": 1000,
            "CHUNK_OVERLAP": 200,
            "RETRIEVAL_K": 5,
        }
        assert result == expected
        
        # Verify types are correct
        assert isinstance(result["CHUNK_SIZE"], int)
        assert isinstance(result["CHUNK_OVERLAP"], int)
        assert isinstance(result["RETRIEVAL_K"], int)

    @patch('modules.config.os.getenv')
    def test_get_config_mixed_values(self, mock_getenv):
        """Test get_config with mixed string and numeric values."""
        mock_values = {
            "MODEL_NAME": "llama3.2",
            "EMBEDDING_MODEL": "nomic-embed-text",
            "CHUNK_SIZE": "1500",
            "CHUNK_OVERLAP": "300",
            "RETRIEVAL_K": "10",
            "PERSIST_DIRECTORY": "/data/chroma",
            "COLLECTION_NAME": "obsidian_docs",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        expected = {
            "MODEL_NAME": "llama3.2",
            "EMBEDDING_MODEL": "nomic-embed-text",
            "CHUNK_SIZE": 1500,
            "CHUNK_OVERLAP": 300,
            "RETRIEVAL_K": 10,
            "PERSIST_DIRECTORY": "/data/chroma",
            "COLLECTION_NAME": "obsidian_docs",
        }
        assert result == expected

    @patch('modules.config.os.getenv')
    def test_get_config_partial_values(self, mock_getenv):
        """Test get_config with only some environment variables set."""
        mock_values = {
            "MODEL_NAME": "test-model",
            "EMBEDDING_MODEL": None,
            "CHUNK_SIZE": "800",
            "CHUNK_OVERLAP": None,
            "RETRIEVAL_K": "3",
            "PERSIST_DIRECTORY": None,
            "COLLECTION_NAME": "partial_collection",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        expected = {
            "MODEL_NAME": "test-model",
            "CHUNK_SIZE": 800,
            "RETRIEVAL_K": 3,
            "COLLECTION_NAME": "partial_collection",
        }
        assert result == expected

    @patch('modules.config.os.getenv')
    def test_get_config_zero_numeric_values(self, mock_getenv):
        """Test get_config handles zero values correctly."""
        mock_values = {
            "MODEL_NAME": None,
            "EMBEDDING_MODEL": None,
            "CHUNK_SIZE": "0",
            "CHUNK_OVERLAP": "0",
            "RETRIEVAL_K": "0",
            "PERSIST_DIRECTORY": None,
            "COLLECTION_NAME": None,
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        expected = {
            "CHUNK_SIZE": 0,
            "CHUNK_OVERLAP": 0,
            "RETRIEVAL_K": 0,
        }
        assert result == expected

    @patch('modules.config.os.getenv')
    def test_get_config_negative_numeric_values(self, mock_getenv):
        """Test get_config handles negative numeric values."""
        mock_values = {
            "MODEL_NAME": None,
            "EMBEDDING_MODEL": None,
            "CHUNK_SIZE": "-100",
            "CHUNK_OVERLAP": "-50",
            "RETRIEVAL_K": "-1",
            "PERSIST_DIRECTORY": None,
            "COLLECTION_NAME": None,
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        expected = {
            "CHUNK_SIZE": -100,
            "CHUNK_OVERLAP": -50,
            "RETRIEVAL_K": -1,
        }
        assert result == expected

    @patch('modules.config.os.getenv')
    def test_get_config_empty_string_values(self, mock_getenv):
        """Test get_config handles empty string values."""
        mock_values = {
            "MODEL_NAME": "",
            "EMBEDDING_MODEL": "",
            "CHUNK_SIZE": "",
            "CHUNK_OVERLAP": "",
            "RETRIEVAL_K": "",
            "PERSIST_DIRECTORY": "",
            "COLLECTION_NAME": "",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        # Empty strings should be included for string values
        expected = {
            "MODEL_NAME": "",
            "EMBEDDING_MODEL": "",
            "PERSIST_DIRECTORY": "",
            "COLLECTION_NAME": "",
        }
        assert result == expected

    @patch('modules.config.os.getenv')
    def test_get_config_invalid_numeric_values(self, mock_getenv):
        """Test get_config with invalid numeric values raises ValueError."""
        mock_values = {
            "MODEL_NAME": None,
            "EMBEDDING_MODEL": None,
            "CHUNK_SIZE": "not_a_number",
            "CHUNK_OVERLAP": None,
            "RETRIEVAL_K": None,
            "PERSIST_DIRECTORY": None,
            "COLLECTION_NAME": None,
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        with pytest.raises(ValueError):
            get_config()

    @patch('modules.config.os.getenv')
    def test_get_config_numeric_with_decimal(self, mock_getenv):
        """Test get_config with decimal numeric values raises ValueError."""
        mock_values = {
            "MODEL_NAME": None,
            "EMBEDDING_MODEL": None,
            "CHUNK_SIZE": "1000.5",
            "CHUNK_OVERLAP": None,
            "RETRIEVAL_K": None,
            "PERSIST_DIRECTORY": None,
            "COLLECTION_NAME": None,
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        with pytest.raises(ValueError):
            get_config()

    @patch('modules.config.os.getenv')
    def test_get_config_preserves_string_types(self, mock_getenv):
        """Test that string configuration values preserve their type."""
        mock_values = {
            "MODEL_NAME": "123",  # Numeric string but should remain string
            "EMBEDDING_MODEL": "456",
            "CHUNK_SIZE": None,
            "CHUNK_OVERLAP": None,
            "RETRIEVAL_K": None,
            "PERSIST_DIRECTORY": "789",
            "COLLECTION_NAME": "000",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)
        
        result = get_config()
        
        expected = {
            "MODEL_NAME": "123",
            "EMBEDDING_MODEL": "456",
            "PERSIST_DIRECTORY": "789",
            "COLLECTION_NAME": "000",
        }
        assert result == expected
        
        # Verify they remain as strings
        assert isinstance(result["MODEL_NAME"], str)
        assert isinstance(result["EMBEDDING_MODEL"], str)
        assert isinstance(result["PERSIST_DIRECTORY"], str)
        assert isinstance(result["COLLECTION_NAME"], str)

    def test_dotenv_functionality(self):
        """Test that load_dotenv functionality is working (by checking module was imported)."""
        # Since the module is already imported and load_dotenv was called at import time,
        # we can't mock it after the fact. This test just confirms the module imported successfully.
        from modules.config import get_config
        assert callable(get_config)

    def test_config_keys_are_strings(self):
        """Test that all CONFIG_KEYS are strings."""
        for key in CONFIG_KEYS:
            assert isinstance(key, str)
            assert len(key) > 0

    def test_numeric_config_keys_subset(self):
        """Test that numeric configuration keys are a subset of all keys."""
        numeric_keys = ["CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_K"]
        
        for key in numeric_keys:
            assert key in CONFIG_KEYS

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_with_actual_environment(self):
        """Test get_config with actual environment (no mocking)."""
        # This test uses actual os.environ but starts with a clean slate
        result = get_config()
        assert result == {}

    @patch.dict(os.environ, {
        "MODEL_NAME": "real-model",
        "CHUNK_SIZE": "2000",
        "UNKNOWN_KEY": "ignored"
    }, clear=True)
    def test_get_config_with_real_environment_subset(self):
        """Test get_config with real environment variables (subset)."""
        result = get_config()
        
        expected = {
            "MODEL_NAME": "real-model",
            "CHUNK_SIZE": 2000,
        }
        assert result == expected
        # Should ignore UNKNOWN_KEY as it's not in CONFIG_KEYS 