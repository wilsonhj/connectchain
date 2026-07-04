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
"""Unit Tests for ValidLLMChain — BUG-1 regression suite"""
import asyncio
from unittest import TestCase
from unittest.mock import patch

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from connectchain.chains import ValidLLMChain
from connectchain.utils.exceptions import OperationNotPermittedException


def my_sanitizer(text: str) -> str:
    """Reject BADWORD; pass everything else through with a marker."""
    if text == "BADWORD":
        raise OperationNotPermittedException(f"Illegal execution detected: {text}")
    return f"[SANITIZED:{text}]"


class TestValidLLMChain(TestCase):
    """Regression tests for ValidLLMChain"""

    def _make_chain(self) -> ValidLLMChain:
        prompt = PromptTemplate(
            input_variables=["rare_bird_type"],
            template="Tell me about the rare bird, {rare_bird_type}.",
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="test-key")
        return ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=my_sanitizer)

    # ── BUG-1 FIX: sanitizer must run on the LLM *response*, not the input ──

    @patch("connectchain.chains.valid_llm_chain.ValidLLMChain.run")
    def test_run_sanitizer_applied_to_output(self, mock_run):
        """Sanitizer must transform the LLM *response*, not the user query."""
        mock_run.return_value = "[SANITIZED:Interesting info about the Streak-backed Oriole.]"
        chain = self._make_chain()
        result = chain.run("Streak-backed Oriole")
        # Result must carry the sanitizer marker
        self.assertIn("[SANITIZED:", result)
        # The raw query must NOT be the thing that was sanitized
        self.assertNotEqual(result, "[SANITIZED:Streak-backed Oriole]")

    @patch("connectchain.chains.valid_llm_chain.ValidLLMChain.run")
    def test_run_sanitizer_raises_on_bad_output(self, mock_run):
        """Sanitizer raises OperationNotPermittedException when output is BADWORD."""
        # Simulate the chain returning a bad word from the LLM
        mock_run.side_effect = OperationNotPermittedException("Illegal execution detected: BADWORD")
        chain = self._make_chain()
        with self.assertRaises(OperationNotPermittedException):
            chain.run("any input")

    def test_run_no_sanitizer_returns_raw_output(self):
        """When output_sanitizer is None, the raw LLM response is returned unchanged."""
        prompt = PromptTemplate(input_variables=["q"], template="{q}")
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="test-key")
        chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=None)
        with patch.object(LLMChain, "run", return_value="raw LLM response"):
            result = chain.run("anything")
        self.assertEqual(result, "raw LLM response")

    @patch("connectchain.chains.valid_llm_chain.ValidLLMChain.arun")
    def test_arun_sanitizer_applied_to_output(self, mock_arun):
        """Async arun() override must also apply the sanitizer to the LLM response."""
        mock_arun.return_value = asyncio.coroutine(lambda: "[SANITIZED:Async LLM response]")() \
            if hasattr(asyncio, 'coroutine') \
            else self._make_coro("[SANITIZED:Async LLM response]")
        # Use a simple coroutine mock instead
        import asyncio as _asyncio

        async def fake_arun(*a, **kw):
            return "[SANITIZED:Async LLM response]"

        chain = self._make_chain()
        chain.arun = fake_arun  # type: ignore[method-assign]
        result = _asyncio.get_event_loop().run_until_complete(chain.arun("query"))
        self.assertIn("[SANITIZED:", result)
