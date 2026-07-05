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
"""
This module contains the PortableOrchestrator class.
"""
from typing import Any, List

import connectchain.chains
import connectchain.prompts
import connectchain.utils
from connectchain.lcel import model


class PortableOrchestrator:
    """
    This class is a portable orchestrator that can be used to run a query
    against any chain.  It is portable as it can wrap any third-party LLM
    framework.

    BUG-3 FIX: run_sync() and run() previously called the deprecated
    LLMChain.run() / LLMChain.arun() methods which are scheduled for removal
    in LangChain 0.4.x.  Both methods now use the LCEL .invoke() / .ainvoke()
    API instead.

    CODE-REVIEW FOLLOWUP FIX: The BUG-3 fix above passed a hardcoded
    {"input": query} dict to .invoke()/.ainvoke(). Unlike the old .run(query),
    which auto-maps a single positional value onto the chain's real (sole)
    input key via Chain.prep_inputs(), a dict passed to .invoke() is used
    as-is -- so "input" had to exactly match the prompt's declared
    input_variables, which it essentially never does. This broke every real
    (non-mocked) chain, including the README's own usage example. query is
    now passed through unwrapped so Chain.prep_inputs() maps it exactly as
    .run() used to: onto the chain's own input key when query is a bare
    value, or used as-is when the caller passes a pre-built dict for a
    multi-input chain.
    """

    def __init__(self, chain: connectchain.chains.ValidLLMChain, **kvargs: Any) -> None:
        """Constructor for the PortableOrchestrator class"""
        self._chain = chain
        is_lcel = kvargs.get("lcel")
        self._is_lcel = is_lcel if is_lcel is not None and is_lcel is True else False

    @staticmethod
    def from_prompt_template(
        prompt_template: str, input_variables: List[str], **kwargs: Any
    ) -> "PortableOrchestrator":
        """Method to build a PortableOrchestrator instance"""
        index = kwargs.get("index")
        llm = model(index or "1")

        prompt_sanitizer = (
            kwargs["prompt_sanitizer"] if kwargs.get("prompt_sanitizer") else lambda x: x
        )

        prompt = connectchain.prompts.ValidPromptTemplate(
            output_sanitizer=prompt_sanitizer,
            input_variables=input_variables,
            template=prompt_template,
        )

        chain = connectchain.chains.ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=None)
        return PortableOrchestrator(chain)

    def run_sync(self, query: Any) -> Any:
        """Run the chain synchronously via LCEL .invoke()."""
        result = self._chain.invoke(query)
        return self._extract_output(result)

    async def run(self, query: Any) -> Any:
        """Run the chain asynchronously via LCEL .ainvoke()."""
        result = await self._chain.ainvoke(query)
        return self._extract_output(result)

    def _extract_output(self, result: Any) -> Any:
        """Pull the response text out of a chain's output dict.

        CODE-REVIEW FOLLOWUP FIX: This used to hardcode a "text" then "output"
        key fallback, which silently mishandled any chain built with a custom
        output_key (a first-class LLMChain constructor arg) -- the real key
        would never match either guess, so the response fell through to
        str(result). Reading self._chain.output_key (default "text") targets
        the chain's actual output key instead of guessing.
        """
        if isinstance(result, dict):
            output_key = getattr(self._chain, "output_key", "text")
            missing = object()
            value = result.get(output_key, missing)
            if value is not missing:
                return value
            return str(result)
        return result
