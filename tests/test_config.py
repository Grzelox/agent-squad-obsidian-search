import pytest
import os
from unittest.mock import patch, Mock

from modules.config import get_config, AppConfig


class TestAppConfig:
    """Test cases for AppConfig dataclass."""

    def test_app_config_defaults(self):
        """Test that AppConfig has expected default values."""
        config = AppConfig()
        
        assert config.model_name == "llama3.2"
        assert config.embedding_model == "nomic-embed-text"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.retrieval_k == 3
        assert config.persist_directory == "./chroma_db"
        assert config.collection_name == "obsidian_documents"
        assert config.logs_file == "logs/app.log"
        assert config.summarization_enabled == True
        assert config.summarization_min_words == 500
        assert config.summarization_max_length == 150
        assert config.markdown_mode == "single"
        assert config.markdown_strategy == "auto"

    @patch("modules.config.os.getenv")
    def test_from_env_empty_environment(self, mock_getenv):
        """Test AppConfig.from_env when no environment variables are set."""
        mock_getenv.return_value = None

        config = AppConfig.from_env()

        # Should return default values
        assert config.model_name == "llama3.2"
        assert config.embedding_model == "nomic-embed-text"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.retrieval_k == 3
        assert config.persist_directory == "./chroma_db"
        assert config.collection_name == "obsidian_documents"

    @patch("modules.config.os.getenv")
    def test_from_env_string_values(self, mock_getenv):
        """Test AppConfig.from_env with string configuration values."""
        mock_values = {
            "MODEL_NAME": "test-model",
            "EMBEDDING_MODEL": "test-embedding",
            "PERSIST_DIRECTORY": "/test/persist",
            "COLLECTION_NAME": "test_collection",
            "LOGS_FILE": "/test/logs.log",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)

        config = AppConfig.from_env()

        assert config.model_name == "test-model"
        assert config.embedding_model == "test-embedding"
        assert config.persist_directory == "/test/persist"
        assert config.collection_name == "test_collection"
        assert config.logs_file == "/test/logs.log"

    @patch("modules.config.os.getenv")
    def test_from_env_numeric_values(self, mock_getenv):
        """Test AppConfig.from_env with numeric configuration values."""
        mock_values = {
            "CHUNK_SIZE": "1500",
            "CHUNK_OVERLAP": "300",
            "RETRIEVAL_K": "5",
            "SUMMARIZATION_MIN_WORDS": "1000",
            "SUMMARIZATION_MAX_LENGTH": "200",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)

        config = AppConfig.from_env()

        assert config.chunk_size == 1500
        assert config.chunk_overlap == 300
        assert config.retrieval_k == 5
        assert config.summarization_min_words == 1000
        assert config.summarization_max_length == 200

        # Verify types are correct
        assert isinstance(config.chunk_size, int)
        assert isinstance(config.chunk_overlap, int)
        assert isinstance(config.retrieval_k, int)
        assert isinstance(config.summarization_min_words, int)
        assert isinstance(config.summarization_max_length, int)

    @patch("modules.config.os.getenv")
    def test_from_env_boolean_values(self, mock_getenv):
        """Test AppConfig.from_env with boolean configuration values."""
        test_cases = [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("anything_else", False),
        ]

        for env_value, expected in test_cases:
            mock_getenv.side_effect = lambda key: env_value if key == "SUMMARIZATION_ENABLED" else None
            config = AppConfig.from_env()
            assert config.summarization_enabled == expected, f"Failed for env value: {env_value}"

    @patch("modules.config.os.getenv")
    def test_from_env_markdown_mode_values(self, mock_getenv):
        """Test AppConfig.from_env with valid markdown mode values."""
        valid_modes = ["single", "elements"]
        
        for mode in valid_modes:
            mock_getenv.side_effect = lambda key: mode if key == "MARKDOWN_MODE" else None
            config = AppConfig.from_env()
            assert config.markdown_mode == mode

        # Test invalid mode (should keep default)
        mock_getenv.side_effect = lambda key: "invalid" if key == "MARKDOWN_MODE" else None
        config = AppConfig.from_env()
        assert config.markdown_mode == "single"  # default

    @patch("modules.config.os.getenv")
    def test_from_env_markdown_strategy_values(self, mock_getenv):
        """Test AppConfig.from_env with valid markdown strategy values."""
        valid_strategies = ["auto", "hi_res", "fast"]
        
        for strategy in valid_strategies:
            mock_getenv.side_effect = lambda key: strategy if key == "MARKDOWN_STRATEGY" else None
            config = AppConfig.from_env()
            assert config.markdown_strategy == strategy

        # Test invalid strategy (should keep default)
        mock_getenv.side_effect = lambda key: "invalid" if key == "MARKDOWN_STRATEGY" else None
        config = AppConfig.from_env()
        assert config.markdown_strategy == "auto"  # default

    @patch("modules.config.os.getenv")
    def test_from_env_mixed_values(self, mock_getenv):
        """Test AppConfig.from_env with mixed configuration values."""
        mock_values = {
            "MODEL_NAME": "llama3.2-custom",
            "CHUNK_SIZE": "2000",
            "RETRIEVAL_K": "7",
            "PERSIST_DIRECTORY": "/custom/path",
            "SUMMARIZATION_ENABLED": "true",
            "MARKDOWN_MODE": "elements",
            "MARKDOWN_STRATEGY": "hi_res",
        }
        mock_getenv.side_effect = lambda key: mock_values.get(key)

        config = AppConfig.from_env()

        assert config.model_name == "llama3.2-custom"
        assert config.chunk_size == 2000
        assert config.retrieval_k == 7
        assert config.persist_directory == "/custom/path"
        assert config.summarization_enabled == True
        assert config.markdown_mode == "elements"
        assert config.markdown_strategy == "hi_res"

    def test_update_from_cli(self):
        """Test AppConfig.update_from_cli method."""
        config = AppConfig()
        
        config.update_from_cli(
            model_name="new-model",
            chunk_size=1500,
            retrieval_k=5,
            invalid_key="should_be_ignored"
        )

        assert config.model_name == "new-model"
        assert config.chunk_size == 1500
        assert config.retrieval_k == 5
        # Invalid key should be ignored
        assert not hasattr(config, "invalid_key")

    def test_update_from_cli_with_none_values(self):
        """Test AppConfig.update_from_cli ignores None values."""
        config = AppConfig()
        original_model = config.model_name
        
        config.update_from_cli(
            model_name=None,
            chunk_size=1500,
        )

        assert config.model_name == original_model  # Should remain unchanged
        assert config.chunk_size == 1500

    def test_get_method(self):
        """Test AppConfig.get method for backward compatibility."""
        config = AppConfig()
        
        assert config.get("model_name") == "llama3.2"
        assert config.get("chunk_size") == 1000
        assert config.get("nonexistent_key") is None
        assert config.get("nonexistent_key", "default") == "default"

    @patch("modules.config.os.getenv")
    def test_numeric_conversion_errors(self, mock_getenv):
        """Test that invalid numeric values raise ValueError."""
        mock_getenv.side_effect = lambda key: "not_a_number" if key == "CHUNK_SIZE" else None

        with pytest.raises(ValueError):
            AppConfig.from_env()

    @patch("modules.config.os.getenv")
    def test_decimal_numeric_values_raise_error(self, mock_getenv):
        """Test that decimal numeric values raise ValueError."""
        mock_getenv.side_effect = lambda key: "1000.5" if key == "CHUNK_SIZE" else None

        with pytest.raises(ValueError):
            AppConfig.from_env()


class TestGetConfig:
    """Test cases for get_config function."""

    @patch("modules.config.AppConfig.from_env")
    def test_get_config_returns_app_config(self, mock_from_env):
        """Test that get_config returns an AppConfig instance."""
        mock_config = AppConfig()
        mock_from_env.return_value = mock_config

        # Clear the global config to force recreation
        import modules.config
        modules.config._app_config = None

        result = get_config()

        assert isinstance(result, AppConfig)
        assert result is mock_config
        mock_from_env.assert_called_once()

    @patch("modules.config.AppConfig.from_env")
    def test_get_config_singleton_behavior(self, mock_from_env):
        """Test that get_config returns the same instance on multiple calls."""
        mock_config = AppConfig()
        mock_from_env.return_value = mock_config

        # Clear the global config to force recreation
        import modules.config
        modules.config._app_config = None

        result1 = get_config()
        result2 = get_config()

        assert result1 is result2
        mock_from_env.assert_called_once()  # Should only be called once

    def test_dotenv_loaded(self):
        """Test that dotenv functionality is available."""
        # Since load_dotenv is called at module import, we just verify
        # the module imported successfully and the function is available
        from modules.config import get_config
        assert callable(get_config)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_with_clean_environment(self):
        """Test get_config with empty environment."""
        # Clear the global config to force recreation
        import modules.config
        modules.config._app_config = None

        config = get_config()
        
        # Should have default values
        assert config.model_name == "llama3.2"
        assert config.chunk_size == 1000
        assert config.retrieval_k == 3

    @patch.dict(
        os.environ,
        {"MODEL_NAME": "test-model", "CHUNK_SIZE": "2000", "UNKNOWN_KEY": "ignored"},
        clear=True,
    )
    def test_get_config_with_environment_variables(self):
        """Test get_config with actual environment variables."""
        # Clear the global config to force recreation
        import modules.config
        modules.config._app_config = None

        config = get_config()

        assert config.model_name == "test-model"
        assert config.chunk_size == 2000
        # UNKNOWN_KEY should be ignored since it's not a valid config field
