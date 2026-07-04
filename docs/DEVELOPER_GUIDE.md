# ConnectChain Fork — Developer Guide

> **Audience:** Engineers contributing to `wilsonhj/connectchain`
> **Prerequisites:** Python 3.11+, `uv`, basic LangChain familiarity

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Configuration Schema](#2-configuration-schema)
3. [Adding a Sanitizer](#3-adding-a-sanitizer)
4. [Writing Tests](#4-writing-tests)
5. [Debugging Tips](#5-debugging-tips)
6. [Code Style & Linting](#6-code-style--linting)
7. [Branch & PR Conventions](#7-branch--pr-conventions)

---

## 1. Environment Setup

```bash
# 1. Clone your fork
git clone https://github.com/wilsonhj/connectchain.git
cd connectchain

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install all dependencies including dev extras
uv sync --dev

# 4. Copy and configure environment files
cp example.env .env
cp connectchain/config/example.config.yml config.yml
# Edit .env and config.yml with your API keys / EAS credentials

# 5. Run the test suite to verify setup
make test
# Expected: 36 tests pass
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CONNECTCHAIN_CONFIG_PATH` | ✅ | Absolute path to your `config.yml` |
| `OPENAI_API_KEY` | For direct access | OpenAI API key |
| `EAS_ID_KEY` | For EAS auth | EAS client ID env key name |
| `EAS_SECRET_KEY` | For EAS auth | EAS client secret env key name |
| `HTTPS_PROXY` | Optional | Outbound proxy URL |

---

## 2. Configuration Schema

```yaml
# config.yml
models:
  '1':                          # Model index (string key)
    provider: openai            # openai | azure | anthropic
    type: chat                  # chat | completion
    model_name: gpt-4o-mini
    bypass_eas: false           # true = direct API, skip EAS
    # Optional overrides:
    eas:
      id_key: MY_EAS_ID
      secret_key: MY_EAS_SECRET
      scope: ["api://your-scope"]
    proxy:
      host: proxy.corp.com
      port: 8080
    cert:
      cert_path: /etc/ssl/certs
      cert_name: corp.crt
      cert_size: 2048           # Default; BUG-4 fix makes this optional
```

### Direct Access (No EAS)

Omit `eas`, `proxy`, and `cert` blocks entirely, or set `bypass_eas: true` per model.

---

## 3. Adding a Sanitizer

Sanitizers are plain Python callables: `(str) -> str`. They raise `OperationNotPermittedException` to block execution.

### Input Sanitizer (ValidPromptTemplate)

```python
from connectchain.prompts import ValidPromptTemplate
from connectchain.utils.exceptions import OperationNotPermittedException
import re

def block_pii(query: str) -> str:
    """Block prompts containing SSN patterns."""
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', query):
        raise OperationNotPermittedException(f"PII detected in prompt: {query[:50]}...")
    return query

prompt = ValidPromptTemplate(
    input_variables=["question"],
    template="Answer this: {question}",
    output_sanitizer=block_pii,
)
```

> ⚠️ **BUG-2 Note:** `ValidLLMChain.output_sanitizer` currently runs on the *input*, not the output.
> Until BUG-2 is fixed, use `ValidPromptTemplate` for input sanitization and implement output
> sanitization manually in your chain callback.

### Output Sanitizer (Post-BUG-2 Fix)

```python
from connectchain.chains import ValidLLMChain

def redact_output(response: str) -> str:
    """Redact credit card numbers from LLM output."""
    return re.sub(r'\b(?:\d[ -]?){13,16}\b', '[REDACTED]', response)

chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=redact_output)
```

---

## 4. Writing Tests

Tests live in `tests/unit_tests/` and `tests/integration_tests/`. Follow existing patterns:

```python
# tests/unit_tests/test_my_sanitizer.py
import pytest
from connectchain.utils.exceptions import OperationNotPermittedException
from mymodule import block_pii

class TestBlockPII:
    def test_clean_input_passes(self):
        result = block_pii("What is the weather?")
        assert result == "What is the weather?"

    def test_ssn_raises(self):
        with pytest.raises(OperationNotPermittedException):
            block_pii("My SSN is 123-45-6789")

    def test_partial_ssn_passes(self):
        result = block_pii("Call 555-1234")
        assert result == "Call 555-1234"
```

```bash
# Run just your new test
uv run pytest tests/unit_tests/test_my_sanitizer.py -v

# Run with coverage
make test-unit-cov
```

---

## 5. Debugging Tips

### BUG-1: KeyError in SessionMap

```python
# Reproduce:
from connectchain.utils import SessionMap
sm = SessionMap()
sm.is_expired('1')  # ← KeyError here before fix

# Workaround until fix is merged:
try:
    expired = sm.is_expired('1')
except KeyError:
    expired = True  # treat missing session as expired → triggers refresh
```

### BUG-3: PortableOrchestrator async deprecation warning

```
LangChainDeprecationWarning: The method `LLMChain.arun` was deprecated in
langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
```

This is a warning today but will become an error in LangChain 0.4.x. Track: [ROADMAP.md §Fix F-3](ROADMAP.md).

### Verbose LangChain Logging

```python
import langchain
langchain.debug = True  # Prints full chain inputs/outputs
```

### Config Not Loading

```bash
# Check env var is set
echo $CONNECTCHAIN_CONFIG_PATH
# Should print absolute path to config.yml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yml'))"
```

---

## 6. Code Style & Linting

```bash
# Lint (pylint, configured via .pylintrc)
make lint

# Type checking
uv run mypy connectchain/

# Format (black)
uv run black connectchain/ tests/
```

- Max line length: **120** characters (see `.pylintrc`)
- Docstrings: Google style
- Type hints: required for all public functions

---

## 7. Branch & PR Conventions

| Type | Branch Pattern | Example |
|------|---------------|------|
| Bug fix | `fix/<issue-id>-short-description` | `fix/bug-1-session-map-keyerror` |
| Feature | `feat/<ticket>-short-description` | `feat/f1-mcp-langgraph-memory` |
| Docs | `docs/<description>` | `docs/connectchain-architecture-guide` |
| Refactor | `refactor/<scope>` | `refactor/lcel-migration` |
| Chore | `chore/<scope>` | `chore/bump-langchain-0.3` |

### Commit Message Format

```
<type>(<scope>): <short summary>

<body: what changed and why>

Fixes #<issue-number>
```

PR titles must match the commit convention. Fill out `.github/PULL_REQUEST_TEMPLATE.md` completely before requesting review.
