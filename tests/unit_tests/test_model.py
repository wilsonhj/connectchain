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
"""Unit testing for PortableOrchestrator class"""

import os
import unittest
from unittest.mock import Mock, patch

from langchain_openai import AzureOpenAI, ChatOpenAI

from connectchain.lcel import LCELModelException, model
from connectchain.lcel.model import _get_direct_model_

from .setup_utils import get_mock_config, wrap_model_config


class TestModel(unittest.TestCase):
    """Unit testing the model LCEL method"""

    def setUpWithConfig(self, mock_config):
        """Set up the test with a mock config"""
        patcher_env = patch.dict(
            os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"}
        )
        patcher_config = patch("connectchain.utils.Config.from_env", return_value=mock_config)
        patcher_token = patch(
            "connectchain.lcel.model.get_token_from_env", return_value="test_token"
        )
        patcher_uuid = patch(
            "connectchain.lcel.model.SessionMap.uuid_from_config", return_value="TEST_MODEL_ENV"
        )

        return {
            "env": patcher_env.start(),
            "config": patcher_config.start(),
            "token": patcher_token.start(),
            "uuid": patcher_uuid.start(),
        }

    def tearDown(self):
        """Tear down the test"""
        patch.stopall()

    @patch("connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI))
    # pylint: disable=unused-argument
    def test_model_with_default_llm(self, *args):
        self.setUpWithConfig(get_mock_config())
        test_model = model()
        self.assertIsInstance(test_model, ChatOpenAI)
        test_token = os.getenv("TEST_MODEL_ENV")
        self.assertEqual(test_token, "test_token")

    @patch("connectchain.lcel.model.AzureOpenAI", return_value=Mock(AzureOpenAI))
    # pylint: disable=unused-argument
    def test_model_with_defined_llm(self, *args):
        self.setUpWithConfig(get_mock_config())
        test_model = model("2")
        self.assertIsInstance(test_model, AzureOpenAI)
        test_token = os.getenv("TEST_MODEL_ENV")
        self.assertEqual(test_token, "test_token")

    def test_model_with_no_models_configured(self):
        test_config = get_mock_config()
        del test_config.data["models"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "No models defined in config") as _:
            test_model = model()

    def test_model_with_undefined_llm(self):
        self.setUpWithConfig(get_mock_config())
        with self.assertRaisesRegex(
            LCELModelException, 'Model config at index "gpt5" is not defined'
        ) as _:
            test_model = model("gpt5")

    def test_model_with_unsupported_provider(self):
        """SYSTEMATIC-DEBUGGING FIX: unsupported provider must fail fast with the
        'not supported' message, not a misleading 'API key not found' error from
        an unrelated env var that the caller was never going to set. This also
        fixes a stale assertion: the actual message has always said 'not
        supported', never 'Not implemented' — the regex could not have matched
        even if the ordering bug were absent."""
        test_config = get_mock_config()
        # required to not modify dict instance
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["provider"] = "meta"
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "not supported") as _:
            test_model = model()

    @patch("langchain.chat_models.init_chat_model")
    def test_get_direct_model_reraises_unexpected_exception(self, mock_init_chat_model):
        """BUG-4 regression: an unexpected exception from init_chat_model() (anything
        other than ImportError/ValueError) must be re-raised as LCELModelException with
        the original exception preserved via `from e`, not silently swallowed by a bare
        `except: pass` and left to fall through to manual init."""
        mock_init_chat_model.side_effect = RuntimeError("boom: unexpected failure")
        model_config = wrap_model_config(get_mock_config().data["models"]["1"])
        with self.assertRaisesRegex(LCELModelException, "Unexpected error initialising model"):
            _get_direct_model_(model_config)
        # confirm the original exception is chained, not lost
        try:
            _get_direct_model_(model_config)
        except LCELModelException as exc:
            self.assertIsInstance(exc.__cause__, RuntimeError)

    @patch("langchain.chat_models.init_chat_model")
    def test_get_direct_model_falls_through_on_expected_exceptions(self, mock_init_chat_model):
        """BUG-4 regression: expected ImportError/ValueError from init_chat_model() must
        still fall through to manual provider init (not be re-raised)."""
        mock_init_chat_model.side_effect = ValueError("Unable to infer model provider")
        model_config = wrap_model_config(get_mock_config().data["models"]["1"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, ChatOpenAI)

    def test_use_of_session_map(self):
        self.setUpWithConfig(get_mock_config())
        test_model = model()
        test_model2 = model()
        self.assertIs(test_model, test_model2)

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_no_proxy(self, mock_wrap_with_proxy):
        self.setUpWithConfig(get_mock_config())
        model()
        mock_wrap_with_proxy.assert_not_called()

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_global_proxy(self, mock_wrap_with_proxy: Mock):
        test_config = get_mock_config()
        test_proxy_config = {"host": "localhost", "port": 8080}
        test_config.data["proxy"] = test_proxy_config
        self.setUpWithConfig(test_config)
        model_instance = model()
        mock_wrap_with_proxy.assert_called_once()
        self.assertIs(model_instance, mock_wrap_with_proxy.call_args[0][0])
        used_proxy_config = mock_wrap_with_proxy.call_args[0][1]
        self.assertEqual(used_proxy_config["host"], test_proxy_config["host"])
        self.assertEqual(used_proxy_config["port"], test_proxy_config["port"])

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_model_only_proxy(self, mock_wrap_with_proxy: Mock):
        test_config = get_mock_config()
        test_proxy_config = {"host": "localhost", "port": 8080}
        # required to not modify dict instance
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["proxy"] = test_proxy_config
        self.setUpWithConfig(test_config)
        model_instance = model()
        mock_wrap_with_proxy.assert_called_once()
        self.assertIs(model_instance, mock_wrap_with_proxy.call_args[0][0])
        used_proxy_config = mock_wrap_with_proxy.call_args[0][1]
        self.assertEqual(used_proxy_config["host"], test_proxy_config["host"])
        self.assertEqual(used_proxy_config["port"], test_proxy_config["port"])

    @patch("connectchain.lcel.model.wrap_llm_with_proxy")
    def test_model_configured_with_model_override_proxy(self, mock_wrap_with_proxy: Mock):
        test_config = get_mock_config()
        test_global_proxy_config = {"host": "localhost", "port": 8080}
        test_model_proxy_config = {"host": "localhost", "port": 8080}
        test_config.data["proxy"] = test_global_proxy_config
        # required to not modify dict instance
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["proxy"] = test_model_proxy_config
        self.setUpWithConfig(test_config)
        model_instance = model()
        mock_wrap_with_proxy.assert_called_once()
        self.assertIs(model_instance, mock_wrap_with_proxy.call_args[0][0])
        used_proxy_config = mock_wrap_with_proxy.call_args[0][1]
        self.assertEqual(used_proxy_config["host"], test_model_proxy_config["host"])
        self.assertEqual(used_proxy_config["port"], test_model_proxy_config["port"])
