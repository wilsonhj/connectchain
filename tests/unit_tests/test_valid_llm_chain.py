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
    #
    # IMPORTANT: run()/arun() must be tested by mocking the deepest real hook
    # (LLMChain._call/_acall), NOT run()/arun() (or ValidLLMChain.run itself) or
    # even LLMChain.run/arun directly. Chain.run() internally calls self(...) ->
    # Chain.__call__ -> self.invoke(...), which polymorphically dispatches to
    # ValidLLMChain.invoke() (this class's own override) -- so invoke() already
    # sanitizes before run() ever gets the result back. Mocking run()/arun() at
    # a shallow layer bypasses that real dispatch chain and hides bugs there
    # (a double-sanitization regression here escaped 3 review rounds and 82
    # passing tests for exactly this reason).

    def test_run_sanitizer_applied_to_output(self):
        """Sanitizer must transform the LLM *response*, not the user query, and must
        be applied exactly once (not zero times, not twice)."""
        chain = self._make_chain()
        with patch.object(LLMChain, "_call", return_value={"text": "Interesting info."}):
            result = chain.run("Streak-backed Oriole")
        self.assertEqual(result, "[SANITIZED:Interesting info.]")

    def test_run_sanitizer_raises_on_bad_output(self):
        """Sanitizer raises OperationNotPermittedException when output is BADWORD."""
        chain = self._make_chain()
        with patch.object(LLMChain, "_call", return_value={"text": "BADWORD"}):
            with self.assertRaises(OperationNotPermittedException):
                chain.run("any input")

    def test_run_no_sanitizer_returns_raw_output(self):
        """When output_sanitizer is None, the raw LLM response is returned unchanged."""
        chain = self._make_chain(sanitizer=None)
        with patch.object(LLMChain, "_call", return_value={"text": "raw LLM response"}):
            result = chain.run("anything")
        self.assertEqual(result, "raw LLM response")

    def test_arun_sanitizer_applied_to_output(self):
        """Async arun() must also apply the sanitizer to the LLM response, exactly once."""

        async def fake_acall(*a, **kw):  # pylint: disable=unused-argument
            return {"text": "Raw async LLM response"}

        chain = self._make_chain()
        with patch.object(LLMChain, "_acall", side_effect=fake_acall):
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

    def test_run_forwards_args_and_kwargs_variants(self):
        """run() must forward *args/**kwargs to Chain.run() unmodified across calling
        conventions: extra positional args must not be silently dropped, and the
        kwargs-only multi-input convention (chain.run(key1=val1, key2=val2) -- the
        ONLY way Chain.run() supports multi-input, since it rejects multiple
        positional args itself) must not crash with IndexError, as it did when
        run() hardcoded super().run(args[0], ...)."""
        cases = [
            (("query",), {"include_run_info": True}),
            (("query", "extra_positional"), {}),
            ((), {"rare_bird_type": "Streak-backed Oriole"}),
        ]
        chain = self._make_chain(sanitizer=None)
        for call_args, call_kwargs in cases:
            with self.subTest(call_args=call_args, call_kwargs=call_kwargs):
                with patch.object(LLMChain, "run", return_value="raw") as mock_super_run:
                    result = chain.run(*call_args, **call_kwargs)
                mock_super_run.assert_called_once_with(
                    *call_args, callbacks=None, tags=None, metadata=None, **call_kwargs
                )
                self.assertEqual(result, "raw")

    def test_sanitize_dict_logs_warning_when_output_key_missing(self):
        """When output_sanitizer is set but output_key isn't in the result dict,
        _sanitize_dict must log a warning rather than silently no-op -- an
        unlogged skip here would reintroduce an unsanitized-output bypass with
        no visible trace."""
        chain = self._make_chain()
        with self.assertLogs("connectchain.chains.valid_llm_chain", level="WARNING") as cm:
            result = chain._sanitize_dict(  # pylint: disable=protected-access
                {"unexpected_key": "value"}
            )
        self.assertEqual(result, {"unexpected_key": "value"})
        self.assertTrue(any("output_sanitizer is set but output_key" in msg for msg in cm.output))
