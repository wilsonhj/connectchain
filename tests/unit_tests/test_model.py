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
        with self.assertRaisesRegex(
            LCELModelException, "Unexpected error initialising model"
        ) as cm:
            _get_direct_model_(model_config)
        # confirm the original exception is chained, not lost
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)

    @patch("langchain.chat_models.init_chat_model")
    def test_get_direct_model_falls_through_on_expected_exceptions(self, mock_init_chat_model):
        """BUG-4 regression: expected ImportError/ValueError from init_chat_model() must
        still fall through to manual provider init (not be re-raised)."""
        mock_init_chat_model.side_effect = ValueError("Unable to infer model provider")
        model_config = wrap_model_config(get_mock_config().data["models"]["1"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, ChatOpenAI)

    def test_get_direct_model_safe_message_when_model_name_raises(self):
        """If the original failure inside _get_direct_model_'s try block was itself
        model_config.model_name raising AttributeError (a malformed config missing
        that attribute), the except-Exception handler's error message must not
        re-access that same attribute -- doing so would raise a second, uncaught
        AttributeError instead of the intended clean LCELModelException."""

        class _ModelNameRaises:
            provider = "openai"

            @property
            def model_name(self):
                raise AttributeError("model_name is not available")

        with self.assertRaisesRegex(
            LCELModelException, "Unexpected error initialising model"
        ) as cm:
            _get_direct_model_(_ModelNameRaises())
        self.assertIsInstance(cm.exception.__cause__, AttributeError)

    @patch("connectchain.lcel.model.logger")
    @patch("langchain.chat_models.init_chat_model")
    def test_unsupported_provider_does_not_log_fallback_warning(
        self, mock_init_chat_model, mock_logger
    ):
        """An unsupported provider must be rejected BEFORE init_chat_model() is ever
        attempted, so the misleading 'falling back to manual provider init' warning
        never fires ahead of the correct 'not supported' error."""
        model_config = wrap_model_config(
            {**get_mock_config().data["models"]["1"], "provider": "meta"}
        )
        with self.assertRaisesRegex(LCELModelException, "not supported"):
            _get_direct_model_(model_config)
        mock_init_chat_model.assert_not_called()
        mock_logger.warning.assert_not_called()

    def test_model_azure_endpoint_without_api_version_raises(self):
        """An Azure-shaped api_base without api_version must fail loudly instead of
        silently falling back to a non-Azure ChatOpenAI client pointed at Azure."""
        test_config = get_mock_config()
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["bypass_eas"] = True
        test_config.data["models"]["1"]["api_base"] = "https://my-resource.openai.azure.com/"
        del test_config.data["models"]["1"]["api_version"]
        self.setUpWithConfig(test_config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
                test_model = model()

    def test_model_azure_endpoint_realistic_model_name_without_api_version_raises(self):
        """Regression test: a realistic model_name (e.g. "gpt-4") makes
        langchain.chat_models.init_chat_model() succeed and return early on the fast
        path, so the api_version guard must fire BEFORE that path is even attempted --
        not only in the manual fallback branch reached when the fast path fails (which
        is all test_model_azure_endpoint_without_api_version_raises above exercised,
        via the fixture's non-standard model_name="test_model")."""
        test_config = get_mock_config()
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        test_config.data["models"]["1"]["bypass_eas"] = True
        test_config.data["models"]["1"]["model_name"] = "gpt-4"
        test_config.data["models"]["1"]["api_base"] = "https://my-resource.openai.azure.com/"
        del test_config.data["models"]["1"]["api_version"]
        self.setUpWithConfig(test_config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
                test_model = model()

    @patch("connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI))
    # pylint: disable=unused-argument
    def test_model_eas_chat_missing_api_version_raises_clear_error(self, *args):
        """Regression test: the EAS/_get_chat_model_ path must raise a clear
        LCELModelException for a missing api_version instead of an opaque pydantic
        ValidationError from deep inside ChatOpenAI's constructor."""
        test_config = get_mock_config()
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        del test_config.data["models"]["1"]["api_version"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
            test_model = model()

    @patch("connectchain.lcel.model.AzureOpenAI", return_value=Mock(AzureOpenAI))
    # pylint: disable=unused-argument
    def test_model_eas_azure_missing_api_version_raises_clear_error(self, *args):
        """Regression test: the EAS/_get_azure_model_ path must raise a clear
        LCELModelException for a missing api_version instead of an opaque pydantic
        ValidationError from deep inside AzureOpenAI's constructor."""
        test_config = get_mock_config()
        test_config.data["models"]["2"] = {**test_config.data["models"]["2"]}
        del test_config.data["models"]["2"]["api_version"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
            test_model = model("2")

    def test_use_of_session_map(self):
        self.setUpWithConfig(get_mock_config())
        test_model = model()
        test_model2 = model()
        self.assertIs(test_model, test_model2)

    def test_direct_azure_model_does_not_supply_api_version_twice(self):
        """Regression test: _get_direct_model_'s Azure branch must construct a real
        (non-mocked) AzureOpenAI without pydantic rejecting api_version as supplied
        twice (once as a direct kwarg, once inside model_kwargs)."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://my-resource.openai.azure.com/",
                "api_version": "2024-02-01",
                "engine": "gpt-4",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, AzureOpenAI)

    def test_model_azure_endpoint_realistic_model_name_without_api_version_raises(self):
        """Regression test: a realistic model_name (e.g. "gpt-4") makes
        langchain.chat_models.init_chat_model() succeed and return early on the fast
        path, so the api_version guard must fire BEFORE that path is even attempted --
        not only in the manual fallback branch reached when the fast path fails."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://my-resource.openai.azure.com/",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
                _get_direct_model_(model_config)

    def test_sovereign_cloud_azure_endpoint_detected(self):
        """CODE-REVIEW regression: Azure US Government / China endpoints
        (openai.azure.us, openai.azure.cn) must route to the Azure builder like the
        public cloud, not silently fall through to a plain ChatOpenAI."""
        for host in ("openai.azure.us", "openai.azure.cn"):
            with self.subTest(host=host):
                model_config = wrap_model_config(
                    {
                        "provider": "openai",
                        "model_name": "gpt-4",
                        "api_base": f"https://my-resource.{host}/",
                    }
                )
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    # Reaching the api_version guard proves Azure routing happened.
                    with self.assertRaisesRegex(LCELModelException, "api_version is required"):
                        _get_direct_model_(model_config)

    def test_explicit_azure_flag_forces_azure_routing(self):
        """CODE-REVIEW regression: APIM/custom domains can't be detected by hostname;
        `azure: true` on the model config must force Azure routing explicitly."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://llm-gw.mycorp.example/",
                "azure": True,
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with self.assertRaisesRegex(LCELModelException, "api_version is required"):
                _get_direct_model_(model_config)

    def test_api_key_env_set_but_unset_var_fails_clearly_on_fast_path(self):
        """CODE-REVIEW regression: a config that explicitly names api_key_env whose
        variable is unset must fail with a clear error on the init_chat_model fast
        path too -- previously the fast path silently proceeded without a key and
        failed opaquely inside the provider client at call time."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key_env": "MY_CUSTOM_KEY_VAR_THAT_IS_UNSET",
            }
        )
        env = {k: v for k, v in os.environ.items() if k != "MY_CUSTOM_KEY_VAR_THAT_IS_UNSET"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(LCELModelException, "MY_CUSTOM_KEY_VAR_THAT_IS_UNSET"):
                _get_direct_model_(model_config)

    def test_temperature_honored_on_manual_fallback_path(self):
        """CODE-REVIEW regression: configured temperature was only honored on the
        init_chat_model fast path; the manual fallback ChatOpenAI dropped it."""
        model_config = wrap_model_config(
            {
                "provider": "openai",
                # Non-inferable name forces the manual fallback branch.
                "model_name": "test_model",
                "temperature": 0.25,
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, ChatOpenAI)
        self.assertEqual(result.temperature, 0.25)

    def test_unquoted_yaml_date_api_version_is_coerced_to_string(self):
        """VERIFY finding: YAML parses an unquoted `api_version: 2024-02-01` as a
        datetime.date, which AzureOpenAI's pydantic validation rejects opaquely.
        The api_version guard must coerce it so the natural config spelling works."""
        import datetime  # pylint: disable=import-outside-toplevel

        model_config = wrap_model_config(
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "api_base": "https://my-resource.openai.azure.com/",
                "api_version": datetime.date(2024, 2, 1),  # what yaml.safe_load produces
                "engine": "gpt-4",
            }
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = _get_direct_model_(model_config)
        self.assertIsInstance(result, AzureOpenAI)

    @patch("connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI))
    # pylint: disable=unused-argument
    def test_model_eas_chat_missing_api_version_raises_clear_error(self, *args):
        """Regression test: the EAS/_get_chat_model_ path must raise a clear
        LCELModelException for a missing api_version instead of an opaque pydantic
        ValidationError from deep inside ChatOpenAI's constructor."""
        test_config = get_mock_config()
        test_config.data["models"]["1"] = {**test_config.data["models"]["1"]}
        del test_config.data["models"]["1"]["api_version"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
            test_model = model()

    @patch("connectchain.lcel.model.AzureOpenAI", return_value=Mock(AzureOpenAI))
    # pylint: disable=unused-argument
    def test_model_eas_azure_missing_api_version_raises_clear_error(self, *args):
        """Regression test: the EAS/_get_azure_model_ path must raise a clear
        LCELModelException for a missing api_version instead of an opaque pydantic
        ValidationError from deep inside AzureOpenAI's constructor."""
        test_config = get_mock_config()
        test_config.data["models"]["2"] = {**test_config.data["models"]["2"]}
        del test_config.data["models"]["2"]["api_version"]
        self.setUpWithConfig(test_config)
        with self.assertRaisesRegex(LCELModelException, "api_version is required") as _:
            test_model = model("2")

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
