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
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from .setup_utils import get_mock_config, wrap_model_config
from connectchain.utils import SessionMap


class TestSessionMap(unittest.TestCase):
    """Unit testing the SessionMap"""

    def setUp(self):
        """Reset the singleton and its shared cache so tests don't leak state."""
        SessionMap._instance = None  # pylint: disable=protected-access
        SessionMap.session_map.clear()

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
        """A session whose timestamp is older than its expires_in must be expired."""
        sm = SessionMap(expires_in=900)
        mock_llm = MagicMock()
        sm.new_session("stale-session", mock_llm)
        # Backdate the timestamp (entries are (cached_at, expires_in, llm) tuples)
        sm.session_map["stale-session"] = (
            datetime.now() - timedelta(seconds=1000),
            900,
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

    def test_concurrent_first_construction_yields_single_consistent_instance(self):
        """CODE-REVIEW FOLLOWUP regression: __new__'s singleton construction was
        an unguarded check-then-act race -- many threads calling SessionMap(...)
        for the very first time simultaneously could each construct their own
        instance/lock and race to assign cls._instance, potentially returning
        different instances (or an instance whose expires_in came from a
        different thread's call) to different callers. All threads here race
        to construct the singleton for the first time with distinct expires_in
        values; every caller must get back the exact same instance."""
        results: list = []
        errors: list = []

        def worker(expires_in: int) -> None:
            try:
                results.append(SessionMap(expires_in=expires_in))
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(100, 100 + 20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Errors during construction: {errors}")
        self.assertEqual(len(results), 20)
        self.assertTrue(
            all(instance is results[0] for instance in results),
            "Concurrent first construction returned more than one distinct instance",
        )

    # ── Per-session expiry regression tests ─────────────────────────────────

    def test_expires_in_is_captured_per_session_not_shared(self):
        """A session cached under one expires_in must keep its own expiry even after
        the singleton is later reconstructed with a different interval for another model"""
        session_map = SessionMap(900)
        session_map.new_session("long-lived", "llm-1")
        # Simulate 5 seconds having elapsed since the session was cached.
        cached_at, expires_in, llm = session_map.session_map["long-lived"]
        session_map.session_map["long-lived"] = (cached_at - timedelta(seconds=5), expires_in, llm)

        # A later caller resolves a different model configured with a much shorter interval.
        reconfigured = SessionMap(1)
        self.assertIs(session_map, reconfigured)

        # The long-lived session was cached under expires_in=900 and must still honor
        # that, not the most-recently-configured value of 1.
        self.assertFalse(reconfigured.is_expired("long-lived"))

        # A session cached after the reconfiguration correctly uses the new interval.
        reconfigured.new_session("short-lived", "llm-2")
        cached_at, expires_in, llm = reconfigured.session_map["short-lived"]
        reconfigured.session_map["short-lived"] = (
            cached_at - timedelta(seconds=5),
            expires_in,
            llm,
        )
        self.assertTrue(reconfigured.is_expired("short-lived"))

    def test_new_session_uses_explicit_expires_in_not_singleton_value_at_write_time(self):
        """Regression test for a write-time race: a caller must be able to capture
        expires_in immediately (e.g. before a slow network call) and have new_session()
        honor that captured value, even if another caller reconfigures the singleton's
        shared `expires_in` field in between -- reading self.expires_in only inside
        new_session() would silently apply the wrong, most-recently-configured interval
        instead of the one this session was actually meant to use."""
        session_map = SessionMap(900)
        captured_expires_in = session_map.expires_in  # captured "before the network call"

        # Another request reconfigures the singleton for a different model in between.
        SessionMap(1)

        # Passing the captured value explicitly must win over the singleton's current value.
        session_map.new_session("my-session", "llm-1", captured_expires_in)
        self.assertEqual(session_map.session_map["my-session"][1], 900)

    def test_bare_construction_does_not_reset_configured_interval(self):
        """CODE-REVIEW regression: SessionMap() with no argument (e.g. constructed just
        to read the cache) must NOT silently reset a previously configured interval
        back to the default."""
        SessionMap(3600)
        sm = SessionMap()  # bare read-only construction
        self.assertEqual(sm.expires_in, 3600)
        sm.new_session("k", "llm")
        self.assertEqual(sm.session_map["k"][1], 3600)
        # An explicit reconfiguration still works.
        SessionMap(60)
        self.assertEqual(sm.expires_in, 60)

    def test_missing_token_refresh_interval_uses_default_not_none(self):
        """CODE-REVIEW regression: a config without eas.token_refresh_interval reaches
        SessionMap as None (ConfigWrapper returns None for missing keys). That must
        fall back to the default -- previously the entry cached expires_in=None and
        every later cache-hit expiry check crashed with TypeError ('>' vs NoneType)."""
        sm = SessionMap(None)
        self.assertEqual(sm.expires_in, SessionMap.DEFAULT_EXPIRES_IN)
        sm.new_session("k", "llm", None)
        self.assertEqual(sm.session_map["k"][1], SessionMap.DEFAULT_EXPIRES_IN)
        self.assertFalse(sm.is_expired("k"))  # must not raise TypeError
