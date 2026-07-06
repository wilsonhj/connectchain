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
"""Unit testing for SessionMap class"""
import unittest
from datetime import timedelta

from .setup_utils import get_mock_config, wrap_model_config
from connectchain.utils import SessionMap


class TestSessionMap(unittest.TestCase):
    """Unit testing the SessionMap"""

    def tearDown(self):
        """Reset the singleton and its shared cache so tests don't leak state"""
        SessionMap._instance = None  # pylint: disable=protected-access
        SessionMap.session_map.clear()

    def test_uuid_from_config(self):
        """Test that a PortableOrchestrator instance can be built with the default LLM"""
        test_config = get_mock_config()
        test_uuid = SessionMap.uuid_from_config(test_config, test_config.models["1"])
        test_uuid2 = SessionMap.uuid_from_config(test_config, test_config.models["2"])
        self.assertEqual(test_uuid, "id_key_secret_key_openai_chat_engine_test_model_api_version")
        self.assertEqual(
            test_uuid2, "id_key_secret_key_openai_azure_engine_test_model_other_api_version_other"
        )

    def test_model_config_override(self):
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
