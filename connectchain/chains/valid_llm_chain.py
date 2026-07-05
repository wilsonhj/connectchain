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
This module contains the ValidLLMChain class, which is a subclass of LLMChain.
In addition, it has a callback for sanitizing the output.
"""

from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.base import Callbacks
from langchain.chains.llm import LLMChain
from langchain_core.runnables import RunnableConfig


class ValidLLMChain(LLMChain):
    # pylint: disable=too-few-public-methods
    """
    Extension to LLMChain that sanitizes the **output** if provided with a
    sanitizer function.  The sanitizer is intentionally applied *after* the
    LLM call so that it can inspect or transform the model response — not
    the user's raw input.

    REVIEW-FOLLOWUP FIX: The BUG-1 fix in PR #7 only overrode the legacy
    run()/arun() methods. PortableOrchestrator's BUG-3 fix in the same PR
    switched to calling .invoke()/.ainvoke() directly, which does NOT route
    through run()/arun() — so the sanitizer was silently skipped for every
    call made through PortableOrchestrator (the intended entry point). This
    adds invoke()/ainvoke() overrides so the sanitizer applies on both the
    legacy and LCEL call paths.
    """

    output_sanitizer: Optional[Callable[[str], str]]

    def run(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the chain and sanitize the LLM *response* before returning.

        BUG-1 FIX: Previously the sanitizer was applied to args[0] (the user
        query) before calling super().run(), meaning the actual model output
        was never sanitized.  The fix calls super().run() first and passes the
        result through the sanitizer.
        """
        result = super().run(args[0], callbacks=callbacks, tags=tags, metadata=metadata, **kwargs)
        return self._sanitize(result)

    async def arun(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Async variant — sanitize the LLM *response* before returning.

        BUG-1 FIX (async): Without this override the output_sanitizer is
        silently skipped on all async invocations.
        """
        result = await super().arun(
            args[0], callbacks=callbacks, tags=tags, metadata=metadata, **kwargs
        )
        return self._sanitize(result)

    def invoke(
        self,
        input: Any,  # pylint: disable=redefined-builtin
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """LCEL entry point — sanitize the LLM *response* before returning.

        REVIEW-FOLLOWUP FIX: Callers that use the modern .invoke() API (e.g.
        PortableOrchestrator after its BUG-3 fix) bypassed run()'s sanitizer
        entirely, since Chain.invoke() calls self._call() directly and does
        not route through run(). This override closes that gap.

        `input` is typed Any (not Dict) because Chain.invoke() accepts a bare
        value for single-input chains too -- Chain.prep_inputs() maps it onto
        the chain's own input key, same as run()/arun() do via args[0].
        """
        result = super().invoke(input, config=config, **kwargs)
        return self._sanitize_dict(result)

    async def ainvoke(
        self,
        input: Any,  # pylint: disable=redefined-builtin
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async LCEL entry point — sanitize the LLM *response* before returning.

        REVIEW-FOLLOWUP FIX (async): Without this override the sanitizer is
        silently skipped for every async call made through .ainvoke().
        """
        result = await super().ainvoke(input, config=config, **kwargs)
        return self._sanitize_dict(result)

    def _sanitize(self, result: Any) -> Any:
        """Apply output_sanitizer to a plain string result, if configured."""
        return self.output_sanitizer(result) if self.output_sanitizer else result

    def _sanitize_dict(self, result: Any) -> Any:
        """Apply output_sanitizer to result[self.output_key] in place, if configured."""
        if self.output_sanitizer and isinstance(result, dict) and self.output_key in result:
            result[self.output_key] = self.output_sanitizer(result[self.output_key])
        return result
