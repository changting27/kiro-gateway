# -*- coding: utf-8 -*-

"""
Tests for kiro/account_manager.py - Unified Account System.

Tests the AccountManager class that manages multiple Kiro accounts with:
- Lazy initialization
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- TTL-based model cache refresh
- State persistence
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from kiro.account_manager import (
    Account,
    AccountStats,
    ModelAccountList,
    AccountManager,
    _format_duration
)
from kiro.account_errors import ErrorType
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver


class TestAccountDataclass:
    """
    Tests for Account and AccountStats dataclasses.
    """
    
    def test_account_creation_with_defaults(self):
        """
        Test Account creation with default values.
        
        What it does: Verifies Account dataclass initialization
        Purpose: Ensure default values are set correctly
        """
        print("\n=== Test: Account creation with defaults ===")
        
        # Act
        account = Account(id="/test/path.json")
        
        # Assert
        print(f"Account ID: {account.id}")
        print(f"Auth manager: {account.auth_manager}")
        print(f"Failures: {account.failures}")
        print(f"Last failure time: {account.last_failure_time}")
        
        assert account.id == "/test/path.json"
        assert account.auth_manager is None
        assert account.model_cache is None
        assert account.model_resolver is None
        assert account.failures == 0
        assert account.last_failure_time == 0.0
        assert account.models_cached_at == 0.0
        assert isinstance(account.stats, AccountStats)
    
    def test_account_stats_initialization(self):
        """
        Test AccountStats initialization with zeros.
        
        What it does: Verifies AccountStats default values
        Purpose: Ensure statistics start at zero
        """
        print("\n=== Test: AccountStats initialization ===")
        
        # Act
        stats = AccountStats()
        
        # Assert
        print(f"Total requests: {stats.total_requests}")
        print(f"Successful requests: {stats.successful_requests}")
        print(f"Failed requests: {stats.failed_requests}")
        
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0


class TestAccountManagerLoadCredentials:
    """
    Tests for AccountManager.load_credentials() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_credentials_json_type(self, tmp_path):
        """
        Test loading credentials with type=json.
        
        What it does: Loads single JSON credential file
        Purpose: Verify JSON type credential loading
        """
        print("\n=== Test: load_credentials with type=json ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        assert str(test_json.resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_sqlite_type(self, tmp_path, temp_sqlite_db):
        """
        Test loading credentials with type=sqlite.
        
        What it does: Loads SQLite database credential
        Purpose: Verify SQLite type credential loading
        """
        print("\n=== Test: load_credentials with type=sqlite ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "sqlite",
                "path": temp_sqlite_db,
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1
        assert str(Path(temp_sqlite_db).resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_refresh_token_type(self, tmp_path):
        """
        Test loading credentials with type=refresh_token.
        
        What it does: Loads refresh token credential
        Purpose: Verify refresh_token type credential loading
        """
        print("\n=== Test: load_credentials with type=refresh_token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "refresh_token": "test_refresh_token_abc123",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "region": "us-east-1",
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        # Create state file to avoid errors
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"current_account_index": 0, "model_to_accounts": {}, "accounts": {}}))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        # refresh_token type uses deterministic hash as ID
        account_id = list(manager._accounts.keys())[0]
        assert account_id.startswith("refresh_token_")
    
    @pytest.mark.asyncio
    async def test_load_credentials_folder_scanning(self, tmp_path):
        """
        Test folder scanning for credential files.
        
        What it does: Scans folder and loads all valid credential files
        Purpose: Verify folder scanning functionality
        """
        print("\n=== Test: load_credentials with folder scanning ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Create valid files
        file1 = folder / "account1.json"
        file1.write_text(json.dumps({
            "refreshToken": "token1",
            "accessToken": "access1",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        file2 = folder / "account2.json"
        file2.write_text(json.dumps({
            "refreshToken": "token2",
            "accessToken": "access2",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_invalid_files(self, tmp_path):
        """
        Test that invalid files are skipped with WARNING.
        
        What it does: Loads folder with invalid files
        Purpose: Verify invalid files are skipped gracefully
        """
        print("\n=== Test: load_credentials skips invalid files ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Valid file
        valid_file = folder / "valid.json"
        valid_file.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        # Invalid JSON
        invalid_file = folder / "invalid.json"
        invalid_file.write_text("not a valid json {{{")
        
        # Non-JSON file
        text_file = folder / "readme.txt"
        text_file.write_text("This is not a credential file")
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1  # Only valid file loaded
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_disabled(self, tmp_path):
        """
        Test that entries with enabled=false are skipped.
        
        What it does: Loads credentials with disabled entry
        Purpose: Verify enabled flag is respected
        """
        print("\n=== Test: load_credentials skips disabled entries ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": False  # Disabled
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_type(self, tmp_path):
        """
        Test that entries without type are skipped.
        
        What it does: Loads credentials with missing type field
        Purpose: Verify type validation
        """
        print("\n=== Test: load_credentials skips entries without type ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "path": "/some/path.json",
                "enabled": True
                # Missing "type" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_path(self, tmp_path):
        """
        Test that json/sqlite entries without path are skipped.
        
        What it does: Loads credentials with missing path field
        Purpose: Verify path validation for json/sqlite types
        """
        print("\n=== Test: load_credentials skips json/sqlite without path ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "enabled": True
                # Missing "path" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_refresh_token(self, tmp_path):
        """
        Test that refresh_token entries without refresh_token field are skipped.
        
        What it does: Loads credentials with missing refresh_token field
        Purpose: Verify refresh_token validation
        """
        print("\n=== Test: load_credentials skips refresh_token without token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "enabled": True
                # Missing "refresh_token" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_file_not_found(self, tmp_path):
        """
        Test handling of non-existent credentials.json.
        
        What it does: Attempts to load non-existent file
        Purpose: Verify graceful handling of missing file
        """
        print("\n=== Test: load_credentials with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "nonexistent.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0


class TestAccountManagerLoadState:
    """
    Tests for AccountManager.load_state() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, tmp_path, sample_state_with_data):
        """
        Test loading existing state.json.
        
        What it does: Loads state from file
        Purpose: Verify state restoration
        """
        print("\n=== Test: load_state success ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(sample_state_with_data))
        
        # Create accounts first
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) > 0
    
    @pytest.mark.asyncio
    async def test_load_state_restore_current_account_index(self, tmp_path):
        """
        Test restoration of global current_account_index.
        
        What it does: Restores sticky index from state
        Purpose: Verify global sticky behavior persistence
        """
        print("\n=== Test: load_state restores current_account_index ===")
        
        # Arrange
        state_data = {
            "current_account_index": 2,
            "model_to_accounts": {},
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Current account index: {manager._current_account_index}")
        
        assert manager._current_account_index == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_model_to_accounts(self, tmp_path):
        """
        Test restoration of model_to_accounts mapping.
        
        What it does: Restores model mappings from state
        Purpose: Verify model-to-account mapping persistence
        """
        print("\n=== Test: load_state restores model_to_accounts ===")
        
        # Arrange
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {
                "claude-opus-4.5": {
                    "accounts": ["/test/account1.json", "/test/account2.json"]
                }
            },
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {manager._model_to_accounts}")
        
        assert "claude-opus-4.5" in manager._model_to_accounts
        assert len(manager._model_to_accounts["claude-opus-4.5"].accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_account_runtime_state(self, tmp_path):
        """
        Test restoration of account runtime state (failures, stats, etc).
        
        What it does: Restores account state from file
        Purpose: Verify runtime state persistence
        """
        print("\n=== Test: load_state restores account runtime state ===")
        
        # Arrange
        # Create account first to get correct resolved path
        test_json = tmp_path / "account.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        account_id = str(test_json.resolve())
        
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {},
            "accounts": {
                account_id: {
                    "failures": 3,
                    "last_failure_time": 1704110400.0,
                    "models_cached_at": 1704106800.0,
                    "stats": {
                        "total_requests": 100,
                        "successful_requests": 97,
                        "failed_requests": 3
                    }
                }
            }
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        account = manager._accounts[account_id]
        print(f"Account failures: {account.failures}")
        print(f"Account stats: {account.stats}")
        
        assert account.failures == 3
        assert account.last_failure_time == 1704110400.0
        assert account.models_cached_at == 1704106800.0
        assert account.stats.total_requests == 100
    
    @pytest.mark.asyncio
    async def test_load_state_file_not_found(self, tmp_path):
        """
        Test handling of non-existent state.json (empty state).
        
        What it does: Attempts to load non-existent state file
        Purpose: Verify graceful handling with empty state
        """
        print("\n=== Test: load_state with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "nonexistent.json")
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) == 0
        assert manager._current_account_index == 0
    
    @pytest.mark.asyncio
    async def test_load_state_corrupted_json(self, tmp_path):
        """
        Test handling of corrupted state.json.
        
        What it does: Attempts to load invalid JSON
        Purpose: Verify error handling for corrupted state
        """
        print("\n=== Test: load_state with corrupted JSON ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text("not a valid json {{{")
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert - should handle gracefully
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        
        assert len(manager._model_to_accounts) == 0



class TestAccountManagerInitializeAccount:
    """
    Tests for AccountManager._initialize_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_initialize_account_json_success(self, tmp_path, mock_list_models_response):
        """
        Test successful account initialization with type=json.
        
        What it does: Initializes account with JSON credentials
        Purpose: Verify complete initialization flow
        """
        print("\n=== Test: initialize_account with JSON ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
            "region": "us-east-1"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client for ListAvailableModels
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True
        assert manager._accounts[account_id].auth_manager is not None
        assert manager._accounts[account_id].model_cache is not None
        assert manager._accounts[account_id].model_resolver is not None
    
    @pytest.mark.asyncio
    async def test_initialize_account_fetch_models_fallback(self, tmp_path):
        """
        Test fallback to FALLBACK_MODELS when API fails.
        
        What it does: Initializes account when ListAvailableModels fails
        Purpose: Verify fallback mechanism
        """
        print("\n=== Test: initialize_account with fallback models ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client to fail
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_client.request_with_retry = AsyncMock(side_effect=Exception("Network error"))
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True  # Should succeed with fallback
        assert manager._accounts[account_id].model_cache is not None


class TestAccountManagerGetNextAccount:
    """
    Tests for AccountManager.get_next_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_next_account_single_bypass_circuit_breaker(self, tmp_path, mock_list_models_response):
        """
        Test that single account bypasses Circuit Breaker.
        
        What it does: Gets account when only one exists
        Purpose: Verify single account always returns (no cooldown)
        """
        print("\n=== Test: get_next_account single account bypasses Circuit Breaker ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures (should be ignored for single account)
        manager._accounts[account_id].failures = 10
        manager._accounts[account_id].last_failure_time = time.time()
        
        # Act
        account = await manager.get_next_account("claude-opus-4.5")
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None  # Single account always returns


class TestAccountManagerReportSuccess:
    """
    Tests for AccountManager.report_success() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_success_reset_failures(self, tmp_path, mock_list_models_response):
        """
        Test that report_success resets failures to 0.
        
        What it does: Reports success after failures
        Purpose: Verify failure counter reset
        """
        print("\n=== Test: report_success resets failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures
        manager._accounts[account_id].failures = 5
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        print(f"Failures after success: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0
    
    @pytest.mark.asyncio
    async def test_report_success_update_stats(self, tmp_path, mock_list_models_response):
        """
        Test that report_success updates statistics.
        
        What it does: Reports success and checks stats
        Purpose: Verify statistics tracking
        """
        print("\n=== Test: report_success updates stats ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        stats = manager._accounts[account_id].stats
        print(f"Stats: total={stats.total_requests}, successful={stats.successful_requests}")
        assert stats.total_requests == 1
        assert stats.successful_requests == 1


class TestAccountManagerReportFailure:
    """
    Tests for AccountManager.report_failure() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_failure_recoverable_increment_failures(self, tmp_path, mock_list_models_response):
        """
        Test that RECOVERABLE errors increment failures.
        
        What it does: Reports RECOVERABLE failure
        Purpose: Verify failure counter increment
        """
        print("\n=== Test: report_failure RECOVERABLE increments failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.RECOVERABLE, 429, None
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 1
    
    @pytest.mark.asyncio
    async def test_report_failure_fatal_no_increment(self, tmp_path, mock_list_models_response):
        """
        Test that FATAL errors do NOT increment failures.
        
        What it does: Reports FATAL failure
        Purpose: Verify failures not incremented for request errors
        """
        print("\n=== Test: report_failure FATAL does not increment failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.FATAL, 400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0  # Not incremented


class TestAccountManagerSaveState:
    """
    Tests for AccountManager._save_state() and save_state_periodically().
    """
    
    @pytest.mark.asyncio
    async def test_save_state_atomic_write(self, tmp_path):
        """
        Test atomic state saving via tmp file.
        
        What it does: Saves state and checks tmp file usage
        Purpose: Verify atomic write pattern
        """
        print("\n=== Test: save_state atomic write ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager._save_state()
        
        # Assert
        print(f"State file exists: {state_file.exists()}")
        assert state_file.exists()
        
        # Verify tmp file was cleaned up
        tmp_file = tmp_path / "state.json.tmp"
        print(f"Tmp file exists: {tmp_file.exists()}")
        assert not tmp_file.exists()


class TestAccountManagerGetFirstAccount:
    """
    Tests for AccountManager.get_first_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_first_account_success(self, tmp_path, mock_list_models_response):
        """
        Test getting first initialized account.
        
        What it does: Gets first account for legacy mode
        Purpose: Verify legacy mode support
        """
        print("\n=== Test: get_first_account success ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        account = manager.get_first_account()
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None
        assert account.auth_manager is not None
    
    def test_get_first_account_no_initialized(self, tmp_path):
        """
        Test RuntimeError when no initialized accounts.
        
        What it does: Attempts to get account when none initialized
        Purpose: Verify error handling
        """
        print("\n=== Test: get_first_account with no initialized accounts ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="No initialized accounts available"):
            manager.get_first_account()


class TestAccountManagerGetAllAvailableModels:
    """
    Tests for AccountManager.get_all_available_models() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_all_available_models_collect_from_all(self, tmp_path, mock_list_models_response):
        """
        Test collecting unique models from all accounts.
        
        What it does: Gets models from multiple accounts
        Purpose: Verify model aggregation for /v1/models endpoint
        """
        print("\n=== Test: get_all_available_models collects from all ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        models = manager.get_all_available_models()
        
        # Assert
        print(f"Available models: {len(models)}")
        assert len(models) > 0
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


class TestAccountManagerGetModelMaxInputTokens:
    """
    Tests for AccountManager.get_model_max_input_tokens() method.
    
    This aggregator backs the /v1/models endpoint's context-window advertisement.
    Accounts are injected directly with populated ModelInfoCache instances so the
    aggregation logic is exercised in isolation (no network, no lazy init).
    """
    
    def _make_manager(self, tmp_path) -> AccountManager:
        """Build a bare AccountManager whose _accounts is populated by the test."""
        return AccountManager(
            credentials_file=str(tmp_path / "credentials.json"),
            state_file=str(tmp_path / "state.json"),
        )
    
    @pytest.mark.asyncio
    async def test_returns_limit_for_model_present_single_account(self, tmp_path):
        """
        What it does: Returns the model's real limit from a single account's cache.
        Purpose: Baseline - the advertised window equals Kiro's maxInputTokens.
        """
        print("\n=== Test: single account, model present ===")
        cache = ModelInfoCache()
        await cache.update([
            {"modelId": "claude-opus-4.5", "tokenLimits": {"maxInputTokens": 200000}}
        ])
        manager = self._make_manager(tmp_path)
        manager._accounts = {"acc1": Account(id="acc1", model_cache=cache)}
        
        result = manager.get_model_max_input_tokens("claude-opus-4.5")
        
        print(f"Comparing: Expected 200000, Got {result}")
        assert result == 200000
    
    @pytest.mark.asyncio
    async def test_returns_default_when_model_absent_everywhere(self, tmp_path):
        """
        What it does: Falls back to DEFAULT_MAX_INPUT_TOKENS for an unknown model.
        Purpose: Ensure the endpoint never advertises a null/zero window for a model
                 that no initialized account knows about.
        """
        print("\n=== Test: model absent in all caches -> default ===")
        from kiro.config import DEFAULT_MAX_INPUT_TOKENS
        
        cache = ModelInfoCache()
        await cache.update([
            {"modelId": "claude-opus-4.5", "tokenLimits": {"maxInputTokens": 200000}}
        ])
        manager = self._make_manager(tmp_path)
        manager._accounts = {"acc1": Account(id="acc1", model_cache=cache)}
        
        result = manager.get_model_max_input_tokens("model-that-does-not-exist")
        
        print(f"Comparing: Expected {DEFAULT_MAX_INPUT_TOKENS}, Got {result}")
        assert result == DEFAULT_MAX_INPUT_TOKENS
    
    @pytest.mark.asyncio
    async def test_returns_max_across_accounts(self, tmp_path):
        """
        What it does: Returns the LARGEST limit when accounts disagree for a model.
        Purpose: A model available on a higher-tier account must not be advertised
                 with a smaller window than that account can actually serve.
        """
        print("\n=== Test: multiple accounts -> max limit wins ===")
        cache_small = ModelInfoCache()
        await cache_small.update([
            {"modelId": "claude-opus-4.5", "tokenLimits": {"maxInputTokens": 200000}}
        ])
        cache_large = ModelInfoCache()
        await cache_large.update([
            {"modelId": "claude-opus-4.5", "tokenLimits": {"maxInputTokens": 1000000}}
        ])
        manager = self._make_manager(tmp_path)
        manager._accounts = {
            "acc_small": Account(id="acc_small", model_cache=cache_small),
            "acc_large": Account(id="acc_large", model_cache=cache_large),
        }
        
        result = manager.get_model_max_input_tokens("claude-opus-4.5")
        
        print(f"Comparing: Expected 1000000 (max), Got {result}")
        assert result == 1000000
    
    @pytest.mark.asyncio
    async def test_skips_uninitialized_account_with_none_cache(self, tmp_path):
        """
        What it does: Ignores accounts whose model_cache is None (not yet lazily
                      initialized) and still returns the limit from a ready account.
        Purpose: Lazy init means some accounts have no cache yet - the aggregator
                 must not crash on None and must still find the real limit.
        """
        print("\n=== Test: uninitialized (None cache) account is skipped ===")
        cache = ModelInfoCache()
        await cache.update([
            {"modelId": "claude-haiku-4.5", "tokenLimits": {"maxInputTokens": 100000}}
        ])
        manager = self._make_manager(tmp_path)
        manager._accounts = {
            "uninit": Account(id="uninit", model_cache=None),
            "ready": Account(id="ready", model_cache=cache),
        }
        
        result = manager.get_model_max_input_tokens("claude-haiku-4.5")
        
        print(f"Comparing: Expected 100000, Got {result}")
        assert result == 100000
    
    @pytest.mark.asyncio
    async def test_ignores_cache_that_lacks_the_model(self, tmp_path):
        """
        What it does: A cache that does NOT contain the model is skipped entirely,
                      so its default-floor never inflates the result.
        Purpose: Critical correctness guard - get_max_input_tokens() returns the
                 default for unknown models, so the aggregator must gate on
                 is_valid_model() to avoid reporting a default that is LARGER than
                 the real limit from the account that actually has the model.
        """
        print("\n=== Test: cache lacking the model is gated out ===")
        # Account A genuinely has the model with a SMALL window (below the default).
        cache_has = ModelInfoCache()
        await cache_has.update([
            {"modelId": "tiny-model", "tokenLimits": {"maxInputTokens": 150000}}
        ])
        # Account B does NOT have the model; its get_max_input_tokens would return the
        # 200000 default, which must NOT leak into the result.
        cache_lacks = ModelInfoCache()
        await cache_lacks.update([
            {"modelId": "other-model", "tokenLimits": {"maxInputTokens": 200000}}
        ])
        manager = self._make_manager(tmp_path)
        manager._accounts = {
            "has": Account(id="has", model_cache=cache_has),
            "lacks": Account(id="lacks", model_cache=cache_lacks),
        }
        
        result = manager.get_model_max_input_tokens("tiny-model")
        
        print(f"Comparing: Expected 150000 (default-floor of 'lacks' must be ignored), "
              f"Got {result}")
        assert result == 150000
    
    def test_returns_default_when_no_accounts(self, tmp_path):
        """
        What it does: Returns DEFAULT_MAX_INPUT_TOKENS when there are no accounts.
        Purpose: Defensive - an empty account map must not raise.
        """
        print("\n=== Test: no accounts -> default ===")
        from kiro.config import DEFAULT_MAX_INPUT_TOKENS
        
        manager = self._make_manager(tmp_path)
        manager._accounts = {}
        
        result = manager.get_model_max_input_tokens("claude-opus-4.5")
        
        print(f"Comparing: Expected {DEFAULT_MAX_INPUT_TOKENS}, Got {result}")
        assert result == DEFAULT_MAX_INPUT_TOKENS


class TestFormatDuration:
    """
    Tests for _format_duration() helper function.
    """
    
    def test_format_duration_seconds(self):
        """Test formatting seconds."""
        assert _format_duration(30) == "30s"
        assert _format_duration(59) == "59s"
    
    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        assert _format_duration(60) == "1m"
        assert _format_duration(300) == "5m"
        assert _format_duration(3599) == "59m"
    
    def test_format_duration_hours(self):
        """Test formatting hours."""
        assert _format_duration(3600) == "1h"
        assert _format_duration(7200) == "2h"
        assert _format_duration(86399) == "23h"
    
    def test_format_duration_days(self):
        """Test formatting days."""
        assert _format_duration(86400) == "1d"
        assert _format_duration(172800) == "2d"


class TestAccountManagerModelsHostFetch:
    """
    Tests that account initialization fetches the live model catalog from the
    dedicated AWS models host (q.{region}.amazonaws.com/ListAvailableModels).
    """

    @pytest.mark.asyncio
    async def test_initialize_fetches_from_models_host_with_profile_arn(
        self, tmp_path, mock_list_models_response
    ):
        """
        What it does: Verifies _initialize_account calls ListAvailableModels on the AWS
                      models host and includes origin + profileArn params, then populates
                      the dynamic catalog into the account's resolver.
        Purpose: This is the core fix - the gateway must discover the real, live catalog
                 (Opus 5, Sonnet 5, GPT-5.6, ...) instead of skipping the fetch and using
                 only the static fallback.
        """
        print("\n=== Test: initialize fetches catalog from models_host ===")

        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
            "region": "us-east-1",
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json"),
        )
        await manager.load_credentials()
        account_id = str(test_json.resolve())

        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            success = await manager._initialize_account(account_id)

        assert success is True

        # The request must target the AWS models host, not the runtime host.
        await_kwargs = mock_client.request_with_retry.await_args.kwargs
        print(f"request_with_retry kwargs: {await_kwargs}")
        assert await_kwargs["url"] == "https://q.us-east-1.amazonaws.com/ListAvailableModels"
        assert await_kwargs["params"]["origin"] == "AI_EDITOR"
        assert await_kwargs["params"]["profileArn"] == \
            "arn:aws:codewhisperer:us-east-1:123456789:profile/test"

        # The dynamic catalog from the mocked response must be visible via the resolver.
        available = manager._accounts[account_id].model_resolver.get_available_models()
        print(f"Available models after init: {available}")
        assert "claude-opus-4.5" in available
        assert "claude-sonnet-4.5" in available

    @pytest.mark.asyncio
    async def test_initialize_uses_models_host_even_on_runtime_generation(
        self, tmp_path, mock_list_models_response
    ):
        """
        What it does: Confirms the catalog fetch host is the AWS models host regardless of
                      the generation host being runtime.{region}.kiro.dev.
        Purpose: Regression guard for the old behavior that skipped ListAvailableModels
                 whenever the generation host was the runtime host.
        """
        print("\n=== Test: models_host used even though generation host is runtime ===")

        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "region": "us-east-1",
        }))

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))

        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json"),
        )
        await manager.load_credentials()
        account_id = str(test_json.resolve())

        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client

            await manager._initialize_account(account_id)

        auth_manager = manager._accounts[account_id].auth_manager
        # Generation host stays on runtime, catalog fetch went to the AWS host.
        assert "runtime." in auth_manager.api_host
        assert mock_client.request_with_retry.await_args.kwargs["url"].startswith(
            "https://q.us-east-1.amazonaws.com"
        )
