# Copyright 2023 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
"""Unit testing for SessionMap class — BUG-2 regression suite"""
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from .setup_utils import get_mock_config, wrap_model_config
from connectchain.utils import SessionMap


class TestSessionMap(unittest.TestCase):
    """Unit testing the SessionMap"""

    def setUp(self):
        """Reset singleton between tests."""
        SessionMap._instance = None

    # ── Existing tests (unchanged) ──────────────────────────────────────────

    def test_uuid_from_config(self):
        """UUID generation from config produces stable, predictable strings."""
        test_config = get_mock_config()
        test_uuid = SessionMap.uuid_from_config(test_config, test_config.models["1"])
        test_uuid2 = SessionMap.uuid_from_config(test_config, test_config.models["2"])
        self.assertEqual(test_uuid, "id_key_secret_key_openai_chat_engine_test_model_api_version")
        self.assertEqual(
            test_uuid2, "id_key_secret_key_openai_azure_engine_test_model_other_api_version_other"
        )

    def test_model_config_override(self):
        """Per-model EAS config overrides the global config values."""
        test_config = get_mock_config()
        test_model_config = wrap_model_config(
            {
                "eas": {"id_key": "mod_id", "secret_key": "mod_sec"},  # EARLYBIRD-IGNORE
                "provider": "oss_provider",
                "model_name": "some_model",
                "type": "some_model_type",
                "engine": "oss_engine",
                "api_version": "latest",
            }
        )
        test_uuid = SessionMap.uuid_from_config(test_config, test_model_config)
        self.assertEqual(
            test_uuid, "mod_id_mod_sec_oss_provider_some_model_type_oss_engine_some_model_latest"
        )

    # ── BUG-2 FIX: regression tests ─────────────────────────────────────────

    def test_is_expired_unknown_session_returns_true(self):
        """BUG-2: is_expired() must return True (not raise KeyError) for unknown session IDs."""
        sm = SessionMap(expires_in=900)
        # Must not raise KeyError; must return True to trigger token refresh
        result = sm.is_expired("session-that-was-never-registered")
        self.assertTrue(result)

    def test_is_expired_active_session_returns_false(self):
        """A freshly created session must not be considered expired."""
        sm = SessionMap(expires_in=900)
        mock_llm = MagicMock()
        sm.new_session("active-session", mock_llm)
        self.assertFalse(sm.is_expired("active-session"))

    def test_is_expired_stale_session_returns_true(self):
        """A session whose timestamp is older than expires_in must be expired."""
        sm = SessionMap(expires_in=900)
        mock_llm = MagicMock()
        sm.new_session("stale-session", mock_llm)
        # Backdate the timestamp
        sm.session_map["stale-session"] = (
            datetime.now() - timedelta(seconds=1000),
            mock_llm,
        )
        self.assertTrue(sm.is_expired("stale-session"))

    def test_thread_safety_no_race_condition(self):
        """Concurrent reads on is_expired() must never raise or corrupt state."""
        sm = SessionMap(expires_in=900)
        errors: list = []

        def worker():
            for _ in range(200):
                try:
                    sm.is_expired("concurrent-key")
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Race condition detected: {errors}")
