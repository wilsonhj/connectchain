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
"""Unit testing for PortableOrchestrator class — BUG-3 regression suite"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from langchain_openai import ChatOpenAI

from connectchain.chains import ValidLLMChain
from connectchain.orchestrators import PortableOrchestrator
from connectchain.prompts import ValidPromptTemplate

from .setup_utils import get_mock_config


class TestPortableOrchestrator(unittest.TestCase):
    """Unit testing for PortableOrchestrator class"""

    def setUp(self) -> None:
        self.from_env_patcher = patch(
            "connectchain.utils.Config.from_env", return_value=get_mock_config()
        )
        self.mock_from_env = self.from_env_patcher.start()
        self.chat_openai_patcher = patch(
            "connectchain.lcel.model.ChatOpenAI", return_value=Mock(ChatOpenAI)
        )
        self.mock_chat_openai = self.chat_openai_patcher.start()

    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch("connectchain.lcel.model.SessionMap.uuid_from_config", return_value="TEST_MODEL_ENV")
    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    def test_build_and_model_with_default_llm(
        self, mock_get_token, *args
    ):  # pylint: disable=unused-argument
        """PortableOrchestrator can be built with the default LLM."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        self.assertEqual(orchestrator._is_lcel, False)  # pylint: disable=protected-access
        self.mock_from_env.assert_called_once()
        mock_get_token.assert_called_once()
        self.mock_chat_openai.assert_called_once()
        PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        self.mock_chat_openai.assert_called_once()  # second call uses cache

    @patch.dict(os.environ, {"CONFIG_PATH": "test_path"})
    def test_build_with_missing_config(self, *args):  # pylint: disable=unused-argument
        """Missing model index raises a descriptive exception."""
        with self.assertRaisesRegex(BaseException, 'Model config at index "gpt5" is not defined'):
            PortableOrchestrator.from_prompt_template("test_template", ["var1"], index="gpt5")

    @patch.dict(os.environ, {"CONFIG_PATH": "any_path"})
    def test_build_with_unsupported_llm(self, *args):  # pylint: disable=unused-argument
        """Unsupported model index raises a descriptive exception."""
        with self.assertRaisesRegex(BaseException, 'Model config at index "gpt5" is not defined'):
            PortableOrchestrator.from_prompt_template("test_template", ["var1"], index="gpt5")

    # ── BUG-3 FIX: run_sync must call .invoke(), not deprecated .run() ──────

    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    def test_run_sync_uses_invoke(self, *args):  # pylint: disable=unused-argument
        """run_sync() must call .invoke() with a dict input, not deprecated .run()."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        orchestrator._chain.invoke = Mock(
            return_value={"text": "invoke_response"}
        )  # pylint: disable=protected-access
        response = orchestrator.run_sync("test_query")
        orchestrator._chain.invoke.assert_called_once_with(
            {"input": "test_query"}
        )  # pylint: disable=protected-access
        self.assertEqual(response, "invoke_response")

    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    def test_run_sync_output_key_fallback(self, *args):  # pylint: disable=unused-argument
        """run_sync() handles chains that return 'output' key instead of 'text'."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        orchestrator._chain.invoke = Mock(
            return_value={"output": "output_key_response"}
        )  # pylint: disable=protected-access
        response = orchestrator.run_sync("test_query")
        self.assertEqual(response, "output_key_response")

    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    def test_run_sync_returns_empty_string_text(self, *args):  # pylint: disable=unused-argument
        """PR-7-FOLLOWUP regression: run_sync() used `result.get("text") or
        result.get("output") or str(result)`. `or` treats a legitimate but falsy
        value (an empty-string completion) as absent, so it fell through to
        `str(result)` and returned the stringified dict instead of the real
        (empty) response. The chain returning "" is valid output, not a missing
        key, and must be returned as-is."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        orchestrator._chain.invoke = Mock(return_value={"text": ""})  # pylint: disable=protected-access
        response = orchestrator.run_sync("test_query")
        self.assertEqual(response, "")

    @patch("connectchain.lcel.model.get_token_from_env", return_value="test_token")
    @patch("connectchain.prompts.ValidPromptTemplate", return_value=Mock(ValidPromptTemplate))
    @patch("connectchain.chains.ValidLLMChain", return_value=Mock(ValidLLMChain))
    @patch.dict(os.environ, {"CONFIG_PATH": "any_path", "id_key": "any", "secret_key": "any"})
    def test_run_async_uses_ainvoke(self, *args):  # pylint: disable=unused-argument
        """async run() must call .ainvoke() with a dict input, not deprecated .arun()."""
        orchestrator = PortableOrchestrator.from_prompt_template("test_template", ["var1"])
        orchestrator._chain.ainvoke = AsyncMock(
            return_value={"text": "async_response"}
        )  # pylint: disable=protected-access
        response = asyncio.run(orchestrator.run("async_query"))
        orchestrator._chain.ainvoke.assert_called_once_with(
            {"input": "async_query"}
        )  # pylint: disable=protected-access
        self.assertEqual(response, "async_response")
