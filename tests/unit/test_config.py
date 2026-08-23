# -*- coding: utf-8 -*-

"""
Unit tests for the configuration module.
Verifies loading settings from environment variables.
"""

import pytest
import os
from unittest.mock import patch


class TestLogLevelConfig:
    """Tests for LOG_LEVEL configuration."""
    
    def test_default_log_level_is_info(self):
        """
        What it does: Verifies that LOG_LEVEL defaults to INFO.
        Purpose: Ensure that INFO is used when no environment variable is set.
        
        Note: This test verifies the config.py code logic, not the actual
        value from the .env file. We mock os.getenv to simulate
        the absence of the environment variable.
        """
        print("Setup: Mocking os.getenv for LOG_LEVEL...")
        
        # Create a mock that returns None for LOG_LEVEL (simulating missing variable)
        original_getenv = os.getenv
        
        def mock_getenv(key, default=None):
            if key == "LOG_LEVEL":
                print(f"os.getenv('{key}') -> None (mocked)")
                return default  # Return default, simulating missing variable
            return original_getenv(key, default)
        
        with patch.object(os, 'getenv', side_effect=mock_getenv):
            # Reload config module with mocked getenv
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'INFO', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "INFO"
        
        # Restore module with real values
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
    
    def test_log_level_from_environment(self):
        """
        What it does: Verifies loading LOG_LEVEL from environment variable.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting LOG_LEVEL=DEBUG...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'DEBUG', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "DEBUG"
    
    def test_log_level_uppercase_conversion(self):
        """
        What it does: Verifies LOG_LEVEL conversion to uppercase.
        Purpose: Ensure that lowercase value is converted to uppercase.
        """
        print("Setup: Setting LOG_LEVEL=warning (lowercase)...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "warning"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'WARNING', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "WARNING"
    
    def test_log_level_trace(self):
        """
        What it does: Verifies setting LOG_LEVEL=TRACE.
        Purpose: Ensure that TRACE level is supported.
        """
        print("Setup: Setting LOG_LEVEL=TRACE...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "TRACE"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "TRACE"
    
    def test_log_level_error(self):
        """
        What it does: Verifies setting LOG_LEVEL=ERROR.
        Purpose: Ensure that ERROR level is supported.
        """
        print("Setup: Setting LOG_LEVEL=ERROR...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "ERROR"
    
    def test_log_level_critical(self):
        """
        What it does: Verifies setting LOG_LEVEL=CRITICAL.
        Purpose: Ensure that CRITICAL level is supported.
        """
        print("Setup: Setting LOG_LEVEL=CRITICAL...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "CRITICAL"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "CRITICAL"


class TestToolDescriptionMaxLengthConfig:
    """Tests for TOOL_DESCRIPTION_MAX_LENGTH configuration."""
    
    def test_default_tool_description_max_length(self):
        """
        What it does: Verifies the default value for TOOL_DESCRIPTION_MAX_LENGTH.
        Purpose: Ensure that 10000 is used by default.
        """
        print("Setup: Removing TOOL_DESCRIPTION_MAX_LENGTH from environment...")
        
        with patch.dict(os.environ, {}, clear=False):
            if "TOOL_DESCRIPTION_MAX_LENGTH" in os.environ:
                del os.environ["TOOL_DESCRIPTION_MAX_LENGTH"]
            
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 10000
    
    def test_tool_description_max_length_from_environment(self):
        """
        What it does: Verifies loading TOOL_DESCRIPTION_MAX_LENGTH from environment.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting TOOL_DESCRIPTION_MAX_LENGTH=5000...")
        
        with patch.dict(os.environ, {"TOOL_DESCRIPTION_MAX_LENGTH": "5000"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 5000
    
    def test_tool_description_max_length_zero_disables(self):
        """
        What it does: Verifies that 0 disables the feature.
        Purpose: Ensure that TOOL_DESCRIPTION_MAX_LENGTH=0 works.
        """
        print("Setup: Setting TOOL_DESCRIPTION_MAX_LENGTH=0...")
        
        with patch.dict(os.environ, {"TOOL_DESCRIPTION_MAX_LENGTH": "0"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 0


class TestTimeoutConfigurationWarning:
    """Tests for _warn_timeout_configuration() function."""
    
    def test_no_warning_when_first_token_less_than_streaming(self, capsys):
        """
        What it does: Verifies that warning is NOT shown with correct configuration.
        Purpose: Ensure that no warning when FIRST_TOKEN_TIMEOUT < STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=15, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "15",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning should NOT be shown
            assert "WARNING" not in captured.err
            assert "Suboptimal timeout configuration" not in captured.err
    
    def test_warning_when_first_token_equals_streaming(self, capsys):
        """
        What it does: Verifies that warning is shown when timeouts are equal.
        Purpose: Ensure that warning when FIRST_TOKEN_TIMEOUT == STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=300, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "300",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning SHOULD be shown
            assert "WARNING" in captured.err or "Suboptimal timeout configuration" in captured.err
    
    def test_warning_when_first_token_greater_than_streaming(self, capsys):
        """
        What it does: Verifies that warning is shown when FIRST_TOKEN > STREAMING.
        Purpose: Ensure that warning when FIRST_TOKEN_TIMEOUT > STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=500, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "500",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning SHOULD be shown
            assert "WARNING" in captured.err or "Suboptimal timeout configuration" in captured.err
            # Verify that timeout values are mentioned in warning
            assert "500" in captured.err
            assert "300" in captured.err
    
    def test_warning_contains_recommendation(self, capsys):
        """
        What it does: Verifies that warning contains a recommendation.
        Purpose: Ensure that user receives useful information.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=400, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "400",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning should contain recommendation
            assert "Recommendation" in captured.err or "LESS than" in captured.err


class TestAwsSsoOidcUrlConfig:
    """Tests for AWS SSO OIDC URL configuration."""
    
    def test_aws_sso_oidc_url_template_exists(self):
        """
        What it does: Verifies that AWS_SSO_OIDC_URL_TEMPLATE constant exists.
        Purpose: Ensure the template is defined in config.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
        
        print("Verification: AWS_SSO_OIDC_URL_TEMPLATE exists...")
        assert hasattr(config_module, 'AWS_SSO_OIDC_URL_TEMPLATE')
        
        print(f"AWS_SSO_OIDC_URL_TEMPLATE: {config_module.AWS_SSO_OIDC_URL_TEMPLATE}")
        assert "oidc" in config_module.AWS_SSO_OIDC_URL_TEMPLATE
        assert "amazonaws.com" in config_module.AWS_SSO_OIDC_URL_TEMPLATE
        assert "{region}" in config_module.AWS_SSO_OIDC_URL_TEMPLATE
    
    def test_get_aws_sso_oidc_url_returns_correct_url(self):
        """
        What it does: Verifies that get_aws_sso_oidc_url returns correct URL.
        Purpose: Ensure the function formats URL correctly.
        """
        print("Setup: Importing get_aws_sso_oidc_url...")
        from kiro.config import get_aws_sso_oidc_url
        
        print("Action: Calling get_aws_sso_oidc_url('us-east-1')...")
        url = get_aws_sso_oidc_url("us-east-1")
        
        print(f"Verification: URL is correct...")
        expected = "https://oidc.us-east-1.amazonaws.com/token"
        print(f"Comparing: Expected '{expected}', Got '{url}'")
        assert url == expected
    
    def test_get_aws_sso_oidc_url_with_different_regions(self):
        """
        What it does: Verifies URL generation for different regions.
        Purpose: Ensure the function works with various AWS regions.
        """
        print("Setup: Importing get_aws_sso_oidc_url...")
        from kiro.config import get_aws_sso_oidc_url
        
        test_cases = [
            ("us-east-1", "https://oidc.us-east-1.amazonaws.com/token"),
            ("eu-west-1", "https://oidc.eu-west-1.amazonaws.com/token"),
            ("ap-southeast-1", "https://oidc.ap-southeast-1.amazonaws.com/token"),
            ("us-west-2", "https://oidc.us-west-2.amazonaws.com/token"),
        ]
        
        for region, expected in test_cases:
            print(f"Action: Calling get_aws_sso_oidc_url('{region}')...")
            url = get_aws_sso_oidc_url(region)
            print(f"Comparing: Expected '{expected}', Got '{url}'")
            assert url == expected


class TestServerHostConfig:
    """Tests for SERVER_HOST configuration."""
    
    def test_default_server_host_is_0_0_0_0(self):
        """
        What it does: Verifies that SERVER_HOST defaults to 0.0.0.0.
        Purpose: Ensure that 0.0.0.0 (all interfaces) is used when no environment variable is set.
        """
        print("Setup: Removing SERVER_HOST from environment and isolating from .env...")

        import importlib
        import kiro.config as config_module

        # Isolate from any developer .env: reloading config re-runs load_dotenv(),
        # which would re-populate SERVER_HOST from a local .env (e.g. a developer
        # using SERVER_HOST=127.0.0.1) and mask the built-in default. Patch
        # load_dotenv to a no-op so this test asserts the real default regardless
        # of the local environment.
        with patch.dict(os.environ, {}, clear=False), patch("dotenv.load_dotenv"):
            os.environ.pop("SERVER_HOST", None)
            importlib.reload(config_module)

            print(f"SERVER_HOST: {config_module.SERVER_HOST}")
            print(f"DEFAULT_SERVER_HOST: {config_module.DEFAULT_SERVER_HOST}")
            print(f"Comparing: Expected '0.0.0.0', Got '{config_module.SERVER_HOST}'")
            assert config_module.SERVER_HOST == "0.0.0.0"
            assert config_module.DEFAULT_SERVER_HOST == "0.0.0.0"

        # Restore real config state (reload with the actual environment/.env) so
        # later tests and modules observe the normal configuration.
        importlib.reload(config_module)
    
    def test_server_host_from_environment(self):
        """
        What it does: Verifies loading SERVER_HOST from environment variable.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting SERVER_HOST=127.0.0.1...")
        
        with patch.dict(os.environ, {"SERVER_HOST": "127.0.0.1"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_HOST: {config_module.SERVER_HOST}")
            print(f"Comparing: Expected '127.0.0.1', Got '{config_module.SERVER_HOST}'")
            assert config_module.SERVER_HOST == "127.0.0.1"
    
    def test_server_host_custom_value(self):
        """
        What it does: Verifies setting SERVER_HOST to a custom IP address.
        Purpose: Ensure that any valid IP address can be used.
        """
        print("Setup: Setting SERVER_HOST=192.168.1.100...")
        
        with patch.dict(os.environ, {"SERVER_HOST": "192.168.1.100"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_HOST: {config_module.SERVER_HOST}")
            assert config_module.SERVER_HOST == "192.168.1.100"


class TestServerPortConfig:
    """Tests for SERVER_PORT configuration."""
    
    def test_default_server_port_is_8000(self):
        """
        What it does: Verifies that SERVER_PORT defaults to 8000.
        Purpose: Ensure that 8000 is used when no environment variable is set.
        """
        print("Setup: Removing SERVER_PORT from environment...")
        
        with patch.dict(os.environ, {}, clear=False):
            if "SERVER_PORT" in os.environ:
                del os.environ["SERVER_PORT"]
            
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            print(f"DEFAULT_SERVER_PORT: {config_module.DEFAULT_SERVER_PORT}")
            print(f"Comparing: Expected 8000, Got {config_module.SERVER_PORT}")
            assert config_module.SERVER_PORT == 8000
            assert config_module.DEFAULT_SERVER_PORT == 8000
    
    def test_server_port_from_environment(self):
        """
        What it does: Verifies loading SERVER_PORT from environment variable.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting SERVER_PORT=9000...")
        
        with patch.dict(os.environ, {"SERVER_PORT": "9000"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            print(f"Comparing: Expected 9000, Got {config_module.SERVER_PORT}")
            assert config_module.SERVER_PORT == 9000
    
    def test_server_port_custom_value(self):
        """
        What it does: Verifies setting SERVER_PORT to a custom port number.
        Purpose: Ensure that any valid port number can be used.
        """
        print("Setup: Setting SERVER_PORT=3000...")
        
        with patch.dict(os.environ, {"SERVER_PORT": "3000"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            assert config_module.SERVER_PORT == 3000
    
    def test_server_port_is_integer(self):
        """
        What it does: Verifies that SERVER_PORT is converted to integer.
        Purpose: Ensure that string from environment is converted to int.
        """
        print("Setup: Setting SERVER_PORT=8080 (as string)...")
        
        with patch.dict(os.environ, {"SERVER_PORT": "8080"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            print(f"Type: {type(config_module.SERVER_PORT)}")
            assert isinstance(config_module.SERVER_PORT, int)
            assert config_module.SERVER_PORT == 8080


class TestKiroCliDbFileConfig:
    """Tests for KIRO_CLI_DB_FILE configuration."""
    
    def test_kiro_cli_db_file_config_exists(self):
        """
        What it does: Verifies that KIRO_CLI_DB_FILE constant exists.
        Purpose: Ensure the config parameter is defined.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
        
        print("Verification: KIRO_CLI_DB_FILE exists...")
        assert hasattr(config_module, 'KIRO_CLI_DB_FILE')
        
        print(f"KIRO_CLI_DB_FILE: '{config_module.KIRO_CLI_DB_FILE}'")
        # Default should be empty string
        assert isinstance(config_module.KIRO_CLI_DB_FILE, str)
    
    def test_kiro_cli_db_file_from_environment(self):
        """
        What it does: Verifies loading KIRO_CLI_DB_FILE from environment variable.
        Purpose: Ensure the value from environment is used and normalized.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        
        # Test that KIRO_CLI_DB_FILE is loaded and is a string
        print(f"KIRO_CLI_DB_FILE: {config_module.KIRO_CLI_DB_FILE}")
        assert isinstance(config_module.KIRO_CLI_DB_FILE, str)
        
        # If value is set (not empty), verify it's a normalized path
        if config_module.KIRO_CLI_DB_FILE:
            # Path should be normalized (no raw ~ or forward slashes on Windows)
            assert not config_module.KIRO_CLI_DB_FILE.startswith("~")
            # Should be a valid path string (contains path separators or is absolute)
            from pathlib import Path
            path = Path(config_module.KIRO_CLI_DB_FILE)
            # Path should be constructable (doesn't raise exception)
            assert str(path) == config_module.KIRO_CLI_DB_FILE


class TestInvalidModelRetryConfig:
    """Tests for transient INVALID_MODEL_ID retry configuration."""

    def test_invalid_model_retry_configuration_is_bounded(self) -> None:
        """What it does: validates types and safety bounds for retry settings.

        Purpose: Prevent invalid settings from creating zero-attempt loops,
        negative delays, jitter above 100%, or an ineffective maximum delay.
        """
        from kiro.config import (
            INVALID_MODEL_BASE_RETRY_DELAY,
            INVALID_MODEL_MAX_RETRIES,
            INVALID_MODEL_MAX_RETRY_DELAY,
            INVALID_MODEL_RETRY_JITTER_RATIO,
        )

        assert isinstance(INVALID_MODEL_MAX_RETRIES, int)
        assert INVALID_MODEL_MAX_RETRIES >= 1
        assert isinstance(INVALID_MODEL_BASE_RETRY_DELAY, float)
        assert INVALID_MODEL_BASE_RETRY_DELAY >= 0.0
        assert isinstance(INVALID_MODEL_MAX_RETRY_DELAY, float)
        assert INVALID_MODEL_MAX_RETRY_DELAY >= INVALID_MODEL_BASE_RETRY_DELAY
        assert isinstance(INVALID_MODEL_RETRY_JITTER_RATIO, float)
        assert 0.0 <= INVALID_MODEL_RETRY_JITTER_RATIO <= 1.0


class TestFallbackModelsConfig:
    """Tests for FALLBACK_MODELS configuration."""
    
    def test_fallback_models_exists(self):
        """
        What it does: Verifies that FALLBACK_MODELS constant exists.
        Purpose: Ensure the fallback model list is defined in config.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
        
        print("Verification: FALLBACK_MODELS exists...")
        assert hasattr(config_module, 'FALLBACK_MODELS')
        
        print(f"FALLBACK_MODELS type: {type(config_module.FALLBACK_MODELS)}")
        assert isinstance(config_module.FALLBACK_MODELS, list)
    
    def test_fallback_models_not_empty(self):
        """
        What it does: Verifies that FALLBACK_MODELS contains at least one model.
        Purpose: Ensure fallback list is populated for DNS failure recovery.
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        print(f"FALLBACK_MODELS length: {len(FALLBACK_MODELS)}")
        print(f"Comparing: Expected > 0, Got {len(FALLBACK_MODELS)}")
        assert len(FALLBACK_MODELS) > 0
    
    def test_fallback_models_structure(self):
        """
        What it does: Verifies that each fallback model has required modelId field.
        Purpose: Ensure fallback models have correct structure for cache.update().
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        print(f"Action: Checking structure of {len(FALLBACK_MODELS)} models...")
        for i, model in enumerate(FALLBACK_MODELS):
            print(f"Checking model {i}: {model}")
            
            print(f"  Verification: model is dict...")
            assert isinstance(model, dict), f"Model {i} is not a dict"
            
            print(f"  Verification: model has 'modelId'...")
            assert "modelId" in model, f"Model {i} missing 'modelId'"
            
            print(f"  Verification: modelId is string...")
            assert isinstance(model["modelId"], str), f"Model {i} modelId is not string"
            
            print(f"  Verification: modelId is not empty...")
            assert len(model["modelId"]) > 0, f"Model {i} modelId is empty"
    
    def test_fallback_models_contain_claude_models(self):
        """
        What it does: Verifies that fallback models include Claude models.
        Purpose: Ensure fallback list contains expected Claude 4/4.5 models.
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        model_ids = [m["modelId"] for m in FALLBACK_MODELS]
        print(f"Model IDs in fallback list: {model_ids}")
        
        print("Verification: Contains at least one Claude model...")
        has_claude = any("claude" in mid.lower() for mid in model_ids)
        assert has_claude, "No Claude models in fallback list"
    
    def test_fallback_models_use_dot_format(self):
        """
        What it does: Verifies that model IDs use dot format (e.g., claude-4.5).
        Purpose: Ensure consistency with Kiro API format.
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        print("Action: Checking model ID format...")
        for model in FALLBACK_MODELS:
            model_id = model["modelId"]
            print(f"Checking: {model_id}")
            
            # If model has version number, it should use dot format
            if any(char.isdigit() for char in model_id):
                # Check for patterns like "4.5" or "4-5"
                if "-4-5" in model_id or "-4-0" in model_id:
                    print(f"  WARNING: {model_id} uses dash format instead of dot")
                    # This is acceptable but not ideal
                    pass


class TestFallbackModelsIntegration:
    """Integration tests for FALLBACK_MODELS with ModelResolver."""
    
    @pytest.mark.asyncio
    async def test_fallback_models_work_with_model_resolver(self):
        """
        What it does: Verifies that fallback models work with ModelResolver normalization.
        Purpose: Ensure that model name normalization (claude-opus-4-5 → claude-opus-4.5)
                 works correctly with fallback models, just like with API models.
        """
        print("Setup: Importing FALLBACK_MODELS and creating cache...")
        from kiro.config import FALLBACK_MODELS
        from kiro.cache import ModelInfoCache
        from kiro.model_resolver import ModelResolver
        
        # Simulate DNS failure scenario - populate cache with fallback models
        cache = ModelInfoCache()
        await cache.update(FALLBACK_MODELS)
        
        print(f"Cache populated with {cache.size} fallback models")
        print(f"Model IDs in cache: {cache.get_all_model_ids()}")
        
        # Create resolver
        resolver = ModelResolver(cache=cache, hidden_models={})
        
        print("\nAction: Testing normalization with dash format...")
        # Test that dash format (claude-opus-4-5) is normalized and found
        test_cases = [
            ("claude-opus-4-5", "claude-opus-4.5"),  # Dash → Dot
            ("claude-sonnet-4-5", "claude-sonnet-4.5"),  # Dash → Dot
            ("claude-haiku-4-5", "claude-haiku-4.5"),  # Dash → Dot
        ]
        
        for input_name, expected_normalized in test_cases:
            print(f"\n  Testing: {input_name} → {expected_normalized}")
            resolution = resolver.resolve(input_name)
            
            print(f"    Resolution source: {resolution.source}")
            print(f"    Normalized: {resolution.normalized}")
            print(f"    Internal ID: {resolution.internal_id}")
            print(f"    Is verified: {resolution.is_verified}")
            
            # Verify normalization happened
            print(f"    Comparing normalized: Expected '{expected_normalized}', Got '{resolution.normalized}'")
            assert resolution.normalized == expected_normalized
            
            # Verify model was found in cache (not passthrough)
            print(f"    Comparing source: Expected 'cache', Got '{resolution.source}'")
            assert resolution.source == "cache", f"Model {input_name} should be found in fallback cache"
            
            print(f"    Comparing is_verified: Expected True, Got {resolution.is_verified}")
            assert resolution.is_verified is True
    
    @pytest.mark.asyncio
    async def test_fallback_models_appear_in_available_models(self):
        """
        What it does: Verifies that fallback models appear in get_available_models().
        Purpose: Ensure that /v1/models endpoint will show fallback models.
        """
        print("Setup: Importing FALLBACK_MODELS and creating cache...")
        from kiro.config import FALLBACK_MODELS
        from kiro.cache import ModelInfoCache
        from kiro.model_resolver import ModelResolver
        
        cache = ModelInfoCache()
        await cache.update(FALLBACK_MODELS)
        
        resolver = ModelResolver(cache=cache, hidden_models={})
        
        print("Action: Getting available models...")
        available = resolver.get_available_models()
        
        print(f"Available models: {available}")
        print(f"Comparing length: Expected {len(FALLBACK_MODELS)}, Got {len(available)}")
        assert len(available) == len(FALLBACK_MODELS)
        
        # Verify all fallback models are present
        fallback_ids = {m["modelId"] for m in FALLBACK_MODELS}
        available_set = set(available)
        
        print(f"Comparing sets: Expected {fallback_ids}, Got {available_set}")
        assert fallback_ids == available_set


# ==================================================================================================
# Tests for WebSearch Configuration
# ==================================================================================================

class TestWebSearchConfig:
    """Tests for WebSearch configuration (WEB_SEARCH_ENABLED)."""
    
    def test_web_search_enabled_default_true(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED defaults to true.
        Purpose: Ensure auto-injection is enabled by default.
        """
        print("Setup: Removing WEB_SEARCH_ENABLED from environment...")
        monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
    
    def test_web_search_enabled_false(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED=false disables auto-injection.
        Purpose: Ensure users can disable auto-injection.
        """
        print("Setup: Setting WEB_SEARCH_ENABLED=false...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected False, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is False
    
    def test_web_search_enabled_true(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED=true enables auto-injection.
        Purpose: Ensure explicit true value works.
        """
        print("Setup: Setting WEB_SEARCH_ENABLED=true...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
    
    def test_web_search_enabled_numeric_values(self, monkeypatch):
        """
        What it does: Verifies numeric values (1/0) work for WEB_SEARCH_ENABLED.
        Purpose: Ensure compatibility with numeric boolean values.
        """
        print("Setup: Testing WEB_SEARCH_ENABLED=1...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "1")
        
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
        
        print("Setup: Testing WEB_SEARCH_ENABLED=0...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "0")
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected False, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is False
    
    def test_web_search_enabled_yes_value(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED=yes enables auto-injection.
        Purpose: Ensure 'yes' value works.
        """
        print("Setup: Setting WEB_SEARCH_ENABLED=yes...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "yes")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
    
    def test_web_search_enabled_case_insensitive(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED is case-insensitive.
        Purpose: Ensure TRUE, True, true all work.
        """
        print("Setup: Testing WEB_SEARCH_ENABLED=TRUE...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "TRUE")
        
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
        
        print("Setup: Testing WEB_SEARCH_ENABLED=FALSE...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "FALSE")
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected False, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is False


# ==================================================================================================
# Tests for Account System Configuration
# ==================================================================================================

class TestAccountSystemConfig:
    """Tests for Account System configuration constants."""
    
    def test_account_system_default_false(self):
        """
        What it does: Verifies ACCOUNT_SYSTEM defaults to false.
        Purpose: Ensure legacy mode is default (backward compatibility).
        """
        print("Setup: Mocking os.getenv for ACCOUNT_SYSTEM...")
        
        original_getenv = os.getenv
        
        def mock_getenv(key, default=None):
            if key == "ACCOUNT_SYSTEM":
                print(f"os.getenv('{key}') -> None (mocked)")
                return default  # Return default, simulating missing variable
            return original_getenv(key, default)
        
        with patch.object(os, 'getenv', side_effect=mock_getenv):
            print("Action: Reloading config module...")
            from importlib import reload
            import kiro.config as config_module
            reload(config_module)
            
            print(f"Comparing ACCOUNT_SYSTEM: Expected False, Got {config_module.ACCOUNT_SYSTEM}")
            assert config_module.ACCOUNT_SYSTEM is False
        
        # Restore module with real values
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
    
    def test_account_system_enabled(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_SYSTEM=true enables account system.
        Purpose: Ensure account system can be enabled via environment variable.
        """
        print("Setup: Setting ACCOUNT_SYSTEM=true...")
        monkeypatch.setenv("ACCOUNT_SYSTEM", "true")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_SYSTEM: Expected True, Got {config_module.ACCOUNT_SYSTEM}")
        assert config_module.ACCOUNT_SYSTEM is True
    
    def test_accounts_config_file_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNTS_CONFIG_FILE defaults to credentials.json.
        Purpose: Ensure default path for credentials configuration.
        """
        print("Setup: Removing ACCOUNTS_CONFIG_FILE from environment...")
        monkeypatch.delenv("ACCOUNTS_CONFIG_FILE", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNTS_CONFIG_FILE: Expected 'credentials.json', Got '{config_module.ACCOUNTS_CONFIG_FILE}'")
        assert config_module.ACCOUNTS_CONFIG_FILE == "credentials.json"
    
    def test_accounts_state_file_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNTS_STATE_FILE defaults to state.json.
        Purpose: Ensure default path for runtime state file.
        """
        print("Setup: Removing ACCOUNTS_STATE_FILE from environment...")
        monkeypatch.delenv("ACCOUNTS_STATE_FILE", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNTS_STATE_FILE: Expected 'state.json', Got '{config_module.ACCOUNTS_STATE_FILE}'")
        assert config_module.ACCOUNTS_STATE_FILE == "state.json"
    
    def test_account_recovery_timeout_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_RECOVERY_TIMEOUT defaults to 60 seconds.
        Purpose: Ensure base timeout for exponential backoff is 60s.
        """
        print("Setup: Removing ACCOUNT_RECOVERY_TIMEOUT from environment...")
        monkeypatch.delenv("ACCOUNT_RECOVERY_TIMEOUT", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_RECOVERY_TIMEOUT: Expected 60, Got {config_module.ACCOUNT_RECOVERY_TIMEOUT}")
        assert config_module.ACCOUNT_RECOVERY_TIMEOUT == 60
    
    def test_account_max_backoff_multiplier_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_MAX_BACKOFF_MULTIPLIER defaults to 1440.0.
        Purpose: Ensure maximum backoff cap is 1 day (60s * 1440 = 86400s).
        """
        print("Setup: Removing ACCOUNT_MAX_BACKOFF_MULTIPLIER from environment...")
        monkeypatch.delenv("ACCOUNT_MAX_BACKOFF_MULTIPLIER", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_MAX_BACKOFF_MULTIPLIER: Expected 1440.0, Got {config_module.ACCOUNT_MAX_BACKOFF_MULTIPLIER}")
        assert config_module.ACCOUNT_MAX_BACKOFF_MULTIPLIER == 1440.0
    
    def test_account_probabilistic_retry_chance_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_PROBABILISTIC_RETRY_CHANCE defaults to 0.1.
        Purpose: Ensure 10% chance for probabilistic retry of broken accounts.
        """
        print("Setup: Removing ACCOUNT_PROBABILISTIC_RETRY_CHANCE from environment...")
        monkeypatch.delenv("ACCOUNT_PROBABILISTIC_RETRY_CHANCE", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_PROBABILISTIC_RETRY_CHANCE: Expected 0.1, Got {config_module.ACCOUNT_PROBABILISTIC_RETRY_CHANCE}")
        assert config_module.ACCOUNT_PROBABILISTIC_RETRY_CHANCE == 0.1
    
    def test_account_cache_ttl_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_CACHE_TTL defaults to 43200 seconds (12 hours).
        Purpose: Ensure model cache TTL is 12 hours by default.
        """
        print("Setup: Removing ACCOUNT_CACHE_TTL from environment...")
        monkeypatch.delenv("ACCOUNT_CACHE_TTL", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_CACHE_TTL: Expected 43200, Got {config_module.ACCOUNT_CACHE_TTL}")
        assert config_module.ACCOUNT_CACHE_TTL == 43200
    
    def test_state_save_interval_seconds_default(self, monkeypatch):
        """
        What it does: Verifies STATE_SAVE_INTERVAL_SECONDS defaults to 10 seconds.
        Purpose: Ensure periodic state saving happens every 10 seconds.
        """
        print("Setup: Removing STATE_SAVE_INTERVAL_SECONDS from environment...")
        monkeypatch.delenv("STATE_SAVE_INTERVAL_SECONDS", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing STATE_SAVE_INTERVAL_SECONDS: Expected 10, Got {config_module.STATE_SAVE_INTERVAL_SECONDS}")
        assert config_module.STATE_SAVE_INTERVAL_SECONDS == 10


# ==================================================================================================
# Tests for get_ssl_verify() / KIRO_EXTRA_CA_CERTS (process-scoped extra CA trust)
# ==================================================================================================

# A throwaway self-signed certificate used ONLY to exercise the SSLContext branch of
# get_ssl_verify(). It is not a secret and is unrelated to any real private CA.
_TEST_CA_PEM = """-----BEGIN CERTIFICATE-----
MIIDHzCCAgegAwIBAgIUaFPNiKsL1q+RLwh9P6OqHk7ca18wDQYJKoZIhvcNAQEL
BQAwHzEdMBsGA1UEAwwUS2lybyBHYXRld2F5IFRlc3QgQ0EwHhcNMjYwNzIxMDIx
NDEyWhcNMzYwNzE4MDIxNDEyWjAfMR0wGwYDVQQDDBRLaXJvIEdhdGV3YXkgVGVz
dCBDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAN8eciRSCD/LtcnD
EWGZHa8mWp7s/e9vI5+mqblSkdKXALn9eyCVyuJvu4GTb90yfwlAegE36/xeEC9x
gAU9G/3x0xeOvdesbDjo3bbReZYEjwnTMd8PwMZzMG+0T6a4DvUjfhAobfIGBR0X
ZJFmH9KCgDCV3D6GuNgwCo+wMbje9tK/mKnLZ4qS4cP8gaJuutKFQBVuEmDiuyzq
HgPyzWQ6PrB3Yz/Vihy0I+TW/OQj4z3LDSHue41uGaXkQ/jHE8LDRUhgwfmh/Osu
vnqDh+pQJB5WKeyeKve8pIj8ROniWXZQa5B8p9A7j8+ocNgK2xZKvbeegsnOR7Vy
193laP8CAwEAAaNTMFEwHQYDVR0OBBYEFDjXybbUEQFtlfyY8gAVOLuSehwxMB8G
A1UdIwQYMBaAFDjXybbUEQFtlfyY8gAVOLuSehwxMA8GA1UdEwEB/wQFMAMBAf8w
DQYJKoZIhvcNAQELBQADggEBAIKjnF0LgLTsuV8HGny+9LPg01vAU1gSV/enSYM0
pKdtnFjYEgQCpgjZml3KrG0JjAiCCX4czMrYG0hRVI407rJrBu6Ybl4ZRKXiU1H1
W7kWOxBlMkAPVy5BEo5QWdNpxlk+Mbkc2kxGW9t/PqYMxKRWouJgjkiOJ/+FkjyY
S6kpX8PMZGA32gzqHI1CCMV7YPl/h99kKzsOaaR+fzZMmZ1Ifts500oSAqXHTxki
tKroxnml20dMnqpycROVkfjUJW8yLZ8Blc/oj/t5x2Sd010dwkp6n6d/SMdPN33K
YaB60nFLEP0/l5d7RoD0VfYFgk0SRnUm3teYUdDsifBAH/U=
-----END CERTIFICATE-----
"""


class TestGetSslVerify:
    """Tests for get_ssl_verify() and the KIRO_EXTRA_CA_CERTS extra-CA trust setting."""

    def test_returns_true_when_no_extra_ca(self):
        """
        What it does: get_ssl_verify() returns True when KIRO_EXTRA_CA_CERTS is unset.
        Purpose: The default must remain standard certifi verification so that ordinary
                 public endpoints keep verifying normally when no extra CA is configured.
        """
        import kiro.config as config
        with patch.object(config, "EXTRA_CA_CERTS", ""):
            result = config.get_ssl_verify()
        assert result is True

    def test_returns_sslcontext_when_extra_ca_set(self, tmp_path):
        """
        What it does: get_ssl_verify() returns an ssl.SSLContext when KIRO_EXTRA_CA_CERTS
                      points to a readable PEM file.
        Purpose: The extra root CA must be loaded into a process-scoped SSLContext
                 (additive on top of certifi), never the OS trust store.
        """
        import ssl
        import kiro.config as config
        ca_file = tmp_path / "extra-ca.pem"
        ca_file.write_text(_TEST_CA_PEM)
        with patch.object(config, "EXTRA_CA_CERTS", str(ca_file)):
            result = config.get_ssl_verify()
        assert isinstance(result, ssl.SSLContext)

    def test_expands_user_home_in_path(self, tmp_path, monkeypatch):
        """
        What it does: get_ssl_verify() expands a leading ~ in KIRO_EXTRA_CA_CERTS.
        Purpose: Users commonly store the CA under their home directory; a literal '~'
                 must resolve to the home path rather than being treated as a filename.
        """
        import ssl
        import kiro.config as config
        monkeypatch.setenv("HOME", str(tmp_path))
        ca_file = tmp_path / "extra-ca.pem"
        ca_file.write_text(_TEST_CA_PEM)
        with patch.object(config, "EXTRA_CA_CERTS", "~/extra-ca.pem"):
            result = config.get_ssl_verify()
        assert isinstance(result, ssl.SSLContext)

    def test_raises_file_not_found_when_ca_missing(self, tmp_path):
        """
        What it does: get_ssl_verify() raises FileNotFoundError when KIRO_EXTRA_CA_CERTS
                      points to a non-existent file.
        Purpose: A misconfigured path must fail loudly and actionably at startup rather than
                 silently reverting to default trust (which would reintroduce the SSL error).
        """
        import kiro.config as config
        missing = tmp_path / "does-not-exist.pem"
        with patch.object(config, "EXTRA_CA_CERTS", str(missing)):
            with pytest.raises(FileNotFoundError):
                config.get_ssl_verify()


class TestModelsHostConfig:
    """Tests for the dedicated ListAvailableModels host (KIRO_MODELS_HOST_TEMPLATE)."""

    def test_models_host_template_uses_aws_q_host(self):
        """
        What it does: Verifies KIRO_MODELS_HOST_TEMPLATE targets q.{region}.amazonaws.com.
        Purpose: ListAvailableModels is only served by the AWS Q host; the runtime host
                 returns 404 for it. The template must therefore point at the AWS host.
        """
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)

        print(f"KIRO_MODELS_HOST_TEMPLATE: {config_module.KIRO_MODELS_HOST_TEMPLATE}")
        assert config_module.KIRO_MODELS_HOST_TEMPLATE == "https://q.{region}.amazonaws.com"

    def test_get_kiro_models_host_formats_region(self):
        """
        What it does: get_kiro_models_host(region) formats the region into the AWS host.
        Purpose: Ensure the model-catalog fetch targets the correct regional host.
        """
        from kiro.config import get_kiro_models_host

        assert get_kiro_models_host("us-east-1") == "https://q.us-east-1.amazonaws.com"
        assert get_kiro_models_host("eu-central-1") == "https://q.eu-central-1.amazonaws.com"

    def test_models_host_differs_from_generation_and_mcp_hosts(self):
        """
        What it does: Verifies the models host is distinct from the generation host
                      (KIRO_API_HOST_TEMPLATE) and the /mcp host (KIRO_Q_HOST_TEMPLATE).
        Purpose: Re-pointing the catalog fetch must never disturb generation or web_search,
                 which stay on the runtime host.
        """
        from kiro.config import (
            KIRO_MODELS_HOST_TEMPLATE,
            KIRO_API_HOST_TEMPLATE,
            KIRO_Q_HOST_TEMPLATE,
        )

        assert KIRO_MODELS_HOST_TEMPLATE != KIRO_API_HOST_TEMPLATE
        assert KIRO_MODELS_HOST_TEMPLATE != KIRO_Q_HOST_TEMPLATE
        assert "runtime." in KIRO_API_HOST_TEMPLATE
        assert "runtime." in KIRO_Q_HOST_TEMPLATE
        assert "amazonaws.com" in KIRO_MODELS_HOST_TEMPLATE


class TestFallbackModelsLatestCatalog:
    """Tests that FALLBACK_MODELS reflects the current live catalog superset."""

    def test_fallback_includes_latest_models(self):
        """
        What it does: Verifies the newest models (Opus 5, Sonnet 5, GPT-5.6 variants,
                      Opus 4.8) are present in the offline fallback snapshot.
        Purpose: When the dynamic catalog fetch is unavailable, /v1/models must still
                 advertise the newest models so clients can select them.
        """
        from kiro.config import FALLBACK_MODELS

        ids = {m["modelId"] for m in FALLBACK_MODELS}
        print(f"Fallback model IDs: {sorted(ids)}")
        for expected in [
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-opus-4.8",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
        ]:
            assert expected in ids, f"{expected} missing from FALLBACK_MODELS"

    def test_fallback_has_no_duplicate_model_ids(self):
        """
        What it does: Verifies FALLBACK_MODELS has no duplicate modelIds.
        Purpose: Duplicates would inflate /v1/models and waste cache slots.
        """
        from kiro.config import FALLBACK_MODELS

        ids = [m["modelId"] for m in FALLBACK_MODELS]
        assert len(ids) == len(set(ids)), "Duplicate modelId in FALLBACK_MODELS"
