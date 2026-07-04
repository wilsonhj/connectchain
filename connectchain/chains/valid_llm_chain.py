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


class ValidLLMChain(LLMChain):
    # pylint: disable=too-few-public-methods
    """
    Extension to LLMChain that sanitizes the **output** if provided with a
    sanitizer function.  The sanitizer is intentionally applied *after* the
    LLM call so that it can inspect or transform the model response — not
    the user's raw input.
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
        # pylint: disable=unused-argument
        """Run the chain and sanitize the LLM *response* before returning.

        BUG-1 FIX: Previously the sanitizer was applied to args[0] (the user
        query) before calling super().run(), meaning the actual model output
        was never sanitized.  The fix calls super().run() first and passes the
        result through the sanitizer.
        """
        result = super().run(args[0], callbacks=callbacks, tags=tags, metadata=metadata)
        return self.output_sanitizer(result) if self.output_sanitizer else result

    async def arun(
        self,
        *args: Any,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        # pylint: disable=unused-argument
        """Async variant — sanitize the LLM *response* before returning.

        BUG-1 FIX (async): Without this override the output_sanitizer is
        silently skipped on all async invocations.
        """
        result = await super().arun(args[0], callbacks=callbacks, tags=tags, metadata=metadata)
        return self.output_sanitizer(result) if self.output_sanitizer else result
