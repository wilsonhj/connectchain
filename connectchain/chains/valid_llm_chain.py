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

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain.callbacks.base import Callbacks
from langchain.chains.llm import LLMChain
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class ValidLLMChain(LLMChain):
    # pylint: disable=too-few-public-methods
    """
    Extension to LLMChain that sanitizes the model response if an
    output_sanitizer callable is provided.

    The sanitizer is applied *after* the LLM call, on all four dispatch
    paths: run(), arun(), invoke(), and ainvoke(). It is intentionally NOT
    applied to the user's input.
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
        """Run the chain; the LLM *response* is already sanitized by the time this returns.

        Chain.run() internally calls self(...), i.e. Chain.__call__, which calls
        self.invoke(...) -- and since self is a ValidLLMChain, that's a polymorphic
        dispatch to THIS class's invoke() override below, which already sanitizes.
        Do not add a second self._sanitize(...) call here: doing so double-applies
        the sanitizer (verified: a sanitizer wrapping its input in "[S:...]" produces
        "[S:[S:...]]" instead of "[S:...]").

        *args and **kwargs are forwarded to Chain.run() unmodified so its own input
        validation and the kwargs-only calling convention for multi-input chains
        (chain.run(key1=val1, key2=val2)) behave exactly as they do on the base class.
        """
        return super().run(*args, callbacks=callbacks, tags=tags, metadata=metadata, **kwargs)

    async def arun(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Async variant of run() -- see run()'s docstring for why no additional
        sanitization happens here (arun() dispatches through ainvoke() the same way).
        """
        return await super().arun(*args, callbacks=callbacks, tags=tags, metadata=metadata, **kwargs)

    def invoke(
        self,
        input: Any,  # pylint: disable=redefined-builtin
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """LCEL entry point — sanitize the LLM *response* before returning.

        `input` is typed Any (not Dict) because Chain.invoke() accepts a bare
        value for single-input chains too -- Chain.prep_inputs() maps it onto
        the chain's own input key, same as run() does via *args.
        """
        result = super().invoke(input, config=config, **kwargs)
        return self._sanitize_dict(result)

    async def ainvoke(
        self,
        input: Any,  # pylint: disable=redefined-builtin
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async variant of invoke() — sanitize the LLM *response* before returning."""
        result = await super().ainvoke(input, config=config, **kwargs)
        return self._sanitize_dict(result)

    def _sanitize_dict(self, result: Any) -> Any:
        """Apply output_sanitizer to result[self.output_key] in place, if configured.

        Logs a warning rather than silently no-op'ing when output_sanitizer
        is set but output_key isn't present in result (a downstream chain
        override or key mismatch), since an unlogged skip here would be an
        unsanitized-output bypass with no visible trace.
        """
        if not self.output_sanitizer:
            return result
        if isinstance(result, dict) and self.output_key in result:
            result[self.output_key] = self.output_sanitizer(result[self.output_key])
        else:
            logger.warning(
                "output_sanitizer is set but output_key '%s' not found in result: %s",
                self.output_key,
                list(result.keys()) if isinstance(result, dict) else type(result),
            )
        return result
