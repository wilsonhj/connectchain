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
This module contains unit tests for the ValidPromptTemplate class.
"""
import re
from unittest import TestCase

from connectchain.prompts import ValidPromptTemplate
from connectchain.utils.exceptions import OperationNotPermittedException


class TestValidPromptTemplate(TestCase):
    """Test Class for ValidPromptTemplate"""

    def test_format_prompt(self):
        """method to test valid_prompt_template"""

        def output_sanitizer(query: str) -> str:
            pattern = r"BADWORD"

            if re.search(pattern, query):
                print("BADWORD found!")
                raise OperationNotPermittedException(f"Illegal execution detected: {query}")

        prompt_template = "Tell me something interesting about {topic}."
        prompt = ValidPromptTemplate(
            output_sanitizer=output_sanitizer,
            input_variables=["topic"],
            template=prompt_template,
        )

        # This will throw an exception as we are trying to pass pattern defined in output_sanitizer
        with self.assertRaises(OperationNotPermittedException):
            prompt.format_prompt(topic="BADWORD")

    def test_format_prompt_sanitizes_rendered_output_not_raw_fields(self):
        """Sanitizer must see the fully rendered output, not each raw field in isolation.

        Splitting "BADWORD" across two template fields means neither field alone
        contains the disallowed pattern -- only the rendered result does. A sanitizer
        that runs on raw per-field inputs before rendering (the old, buggy behavior)
        would miss this; the fix runs the sanitizer on the rendered output instead.
        """

        def output_sanitizer(query: str) -> str:
            pattern = r"BADWORD"

            if re.search(pattern, query):
                print("BADWORD found!")
                raise OperationNotPermittedException(f"Illegal execution detected: {query}")

        prompt_template = "Tell me something interesting about {a}{b}."
        prompt = ValidPromptTemplate(
            output_sanitizer=output_sanitizer,
            input_variables=["a", "b"],
            template=prompt_template,
        )

        # Neither "BAD" nor "WORD" alone matches the pattern; only the rendered
        # concatenation "BADWORD" does.
        with self.assertRaises(OperationNotPermittedException):
            prompt.format_prompt(a="BAD", b="WORD")

    def test_no_sanitizer(self):
        """method to test no_sanitizer"""
        prompt_template = "Tell me something interesting about {topic}."
        prompt = ValidPromptTemplate(
            input_variables=["topic"],
            template=prompt_template,
            output_sanitizer=None,
        )
        output = prompt.format_prompt(topic="BADWORD").text
        self.assertEqual("Tell me something interesting about BADWORD.", output)
