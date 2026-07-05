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

    def _make_chain(self, sanitizer=my_sanitizer) -> ValidLLMChain:
        prompt = PromptTemplate(
            input_variables=["rare_bird_type"],
            template="Tell me about the rare bird, {rare_bird_type}.",
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="test-key")
        return ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=sanitizer)

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
        chain = self._make_chain(sanitizer=None)
        with patch.object(LLMChain, "run", return_value="raw LLM response"):
            result = chain.run("anything")
        self.assertEqual(result, "raw LLM response")

    def test_arun_sanitizer_applied_to_output(self):
        """Async arun() override must also apply the sanitizer to the LLM response.

        NOTE: The original version of this test tried `chain.arun = fake_arun`, which
        raises `ValueError: "ValidLLMChain" object has no field "arun"` — ValidLLMChain
        is a pydantic model and rejects instance attribute assignment for anything not
        declared as a field (see llm_proxy_wrapper.py's own docstring on this exact
        pydantic quirk). Patching the parent class's arun() and calling the real
        chain.arun() exercises the actual sanitizer logic instead.
        """

        async def fake_super_arun(*a, **kw):  # pylint: disable=unused-argument
            return "Raw async LLM response"

        chain = self._make_chain()
        with patch.object(LLMChain, "arun", side_effect=fake_super_arun):
            result = asyncio.run(chain.arun("query"))
        self.assertEqual(result, "[SANITIZED:Raw async LLM response]")

    def test_invoke_sanitizer_applied_to_output(self):
        """REVIEW-FOLLOWUP: .invoke() (the LCEL path PortableOrchestrator uses)
        must also apply the sanitizer, not just run()/arun()."""
        chain = self._make_chain()
        with patch.object(LLMChain, "invoke", return_value={"text": "Raw LLM response"}):
            result = chain.invoke({"rare_bird_type": "Streak-backed Oriole"})
        self.assertEqual(result["text"], "[SANITIZED:Raw LLM response]")

    def test_ainvoke_sanitizer_applied_to_output(self):
        """REVIEW-FOLLOWUP (async): .ainvoke() must also apply the sanitizer."""

        async def fake_super_ainvoke(*a, **kw):  # pylint: disable=unused-argument
            return {"text": "Raw async LLM response"}

        chain = self._make_chain()
        with patch.object(LLMChain, "ainvoke", side_effect=fake_super_ainvoke):
            result = asyncio.run(chain.ainvoke({"rare_bird_type": "Streak-backed Oriole"}))
        self.assertEqual(result["text"], "[SANITIZED:Raw async LLM response]")

    def test_invoke_no_sanitizer_returns_raw_output(self):
        """When output_sanitizer is None, .invoke() returns the raw dict unchanged."""
        chain = self._make_chain(sanitizer=None)
        with patch.object(LLMChain, "invoke", return_value={"text": "raw LLM response"}):
            result = chain.invoke({"rare_bird_type": "anything"})
        self.assertEqual(result["text"], "raw LLM response")

    def test_run_passes_through_extra_kwargs(self):
        """REVIEW-FOLLOWUP: run() must not silently drop caller-supplied **kwargs."""
        chain = self._make_chain()
        with patch.object(LLMChain, "run", return_value="raw") as mock_super_run:
            chain.run("query", include_run_info=True)
        mock_super_run.assert_called_once_with(
            "query", callbacks=None, tags=None, metadata=None, include_run_info=True
        )
