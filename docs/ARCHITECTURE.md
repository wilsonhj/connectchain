# ConnectChain Fork — Architecture Guide

> **Maintainer:** `wilsonhj` | **Base:** `americanexpress/connectchain` | **Last updated:** July 2026

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Module Map](#2-module-map)
3. [LangChain Dependency Chain](#3-langchain-dependency-chain)
4. [Session & Auth Lifecycle](#4-session--auth-lifecycle)
5. [MCP Integration Layer](#5-mcp-integration-layer)
6. [Upstream Risk Surface](#6-upstream-risk-surface)
7. [Bug History](#7-bug-history)

---

## 1. High-Level Overview

ConnectChain is an **enterprise adapter layer** that sits between application code and LangChain, adding:

- **Enterprise Auth Service (EAS)** JWT injection at the model level
- **Outbound proxy** support per-model via config
- **Prompt sanitization hooks** (`ValidPromptTemplate`, `ValidLLMChain`)
- **Portable orchestration** (model-provider-agnostic chain execution)
- **MCP tool integration** (Model Context Protocol via `connectchain.tools.mcp`)

```
┌────────────────────────────────────────────────────────┐
│                   Application Code                      │
└────────────────────────────┬───────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────┐
│                    ConnectChain                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │    lcel/    │  │ orchestrators│  │    prompts/   │  │
│  │  model()   │  │ Portable     │  │ ValidPrompt   │  │
│  └──────┬──────┘  │ Orchestrator │  │ Template      │  │
│         │         └──────┬───────┘  └───────┬───────┘  │
│  ┌──────▼──────────────────────────────────▼───────┐  │
│  │              chains/ValidLLMChain                │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼───────────────────────────┐  │
│  │         utils/ (EAS token, proxy, config)        │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼───────────────────────────┐  │
│  │           tools/mcp (MCPToolAgent)               │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│                      LangChain                        │
│   LLMChain · PromptTemplate · ChatOpenAI · Callbacks  │
└───────────────────────────────────────────────────────┘
```

---

## 2. Module Map

| Module | Key Classes / Functions | Responsibility |
|--------|------------------------|----------------|
| `connectchain.lcel` | `model()`, `LCELLogger`/`PrintLogger`, `LCELRetry` | LCEL-compatible model factory; logging and retry hooks |
| `connectchain.orchestrators` | `PortableOrchestrator` | Provider-agnostic chain runner |
| `connectchain.prompts` | `ValidPromptTemplate` | Pre-send prompt sanitization |
| `connectchain.chains` | `ValidLLMChain` | Post-response output sanitization, applied on `run()`/`arun()`/`invoke()`/`ainvoke()` |
| `connectchain.utils` | `Config`, `get_token_from_env()`, `SessionMap`, `TokenUtil` | Config loading, EAS JWT retrieval, session expiry cache |
| `connectchain.tools.mcp` | `MCPToolAgent`, `MCPToolLoader` | MCP server integration (fully async) |

`connectchain/config/` contains only `example.config.yml` — there is no `connectchain.config` Python module; config parsing lives in `connectchain.utils.config` (`Config`, `ConfigWrapper`).

---

## 3. LangChain Dependency Chain

ConnectChain currently imports from LangChain in one critical path:

```
connectchain.chains.ValidLLMChain
  └── inherits langchain.chains.llm.LLMChain      ← deprecated since 0.1.0, removed in LangChain 1.0
                                                     (per LLMChain's own LangChainDeprecationWarning)
```

`pyproject.toml` pins `langchain<0.4.0` specifically because LangChain's 1.x line removes
`langchain.chains`/`langchain.schema`/`langchain.llms` entirely, which this codebase imports
throughout — an unbounded dependency range breaks the package outright on a fresh install.

`PortableOrchestrator` already uses the LCEL `.invoke()`/`.ainvoke()` API (not the deprecated
`.run()`/`.arun()`), and token injection is a plain constructor argument
(`ChatOpenAI(api_key=SecretStr(auth_token), ...)` in `connectchain/lcel/model.py`) — there is no
monkey-patching involved in getting the token into the model client. (`connectchain.utils.proxy_manager`
does monkey-patch `requests.Session.__init__`, but that's for outbound proxy support, unrelated to
token injection.)

### Migration Target (LCEL Pipes), if `ValidLLMChain` is ever rebuilt on a bare Runnable pipe

```python
# Current (LLMChain-based, still supported through LangChain 0.3.x)
chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=my_sanitizer)
result = chain.invoke({"topic": "..."})

# Hypothetical LCEL-native replacement, needed before/if LangChain 1.0 is adopted
from langchain_core.runnables import RunnableLambda
chain = prompt | llm | RunnableLambda(my_sanitizer)
result = chain.invoke({"topic": "..."})
```

---

## 4. Session & Auth Lifecycle

```
Application calls model() (or PortableOrchestrator built on top of it)
        │
        ▼
SessionMap(config.eas.token_refresh_interval)   ← singleton, per-process
        │
        ▼
get_valid_llm(session_key)   ← atomic: existence + expiry check under one lock
  ├── hit, not expired  → return cached LLM instance
  └── miss or expired   → get_token_from_env() refreshes the JWT via EAS,
                           builds a new ChatOpenAI/AzureOpenAI with that token
                           as a constructor arg, caches it via new_session()
                │
                ▼
        LangChain model call (sync via .invoke(), async via .ainvoke())
                │
                ▼
        [If output_sanitizer is set] applied to the response before returning
```

---

## 5. MCP Integration Layer

The `connectchain.tools.mcp` module wraps the Model Context Protocol (MCP) to expose external tool servers as LangChain-compatible tools, built on `langchain-mcp-adapters`:

```
MCP Server (stdio)
        │
        ▼
MCPToolLoader.load_tools()   ← async, returns LangChain BaseTool objects
        │
        ▼
MCPToolAgent (a langchain_core Runnable)
  ├── ainvoke()   ← async
  └── abatch()    ← async
        │
        ▼
Application code / LangChain agent framework
```

### Current Limitation
`MCPToolAgent` does not support multi-turn memory or LangGraph checkpointing. Each call is stateless.

---

## 6. Upstream Risk Surface

| Risk | Severity | Affected Files | Notes |
|------|----------|---------------|-----|
| `LLMChain` removed in LangChain 1.0 | 🔴 Critical | `chains/valid_llm_chain.py` (inherits it) | `pyproject.toml` pins `langchain<0.4.0` to stay ahead of this; an unbounded range breaks the install today, since `pip`/`uv` will happily resolve to 1.x |
| `langchain-mcp-adapters` version drift | 🟠 High | `tools/mcp/` | Pinned `<0.2.0` — 0.2.0 imports `langchain_core.messages.content`, which needs a newer `langchain-core` than the pinned `langchain<0.4.0` line provides |

Track upstream changes: [langchain-ai/langchain releases](https://github.com/langchain-ai/langchain/releases)

---

## 7. Bug History

The five bugs below were found in a code review. **They are not fixed on this branch or on `main`** — the fixes exist only in a separate, still-open pull request. This table documents the findings for context; it is not a changelog of this branch's own code.

| Area | File | Root Cause (as found) | Fix (in the separate PR, not yet merged) |
|------|------|-----------|-------------|
| Output sanitizer bypass | `chains/valid_llm_chain.py` | `output_sanitizer` was applied to the user's input, not the LLM's response | `run()`/`arun()`/`invoke()`/`ainvoke()` sanitize the response after calling the underlying chain |
| SessionMap KeyError | `utils/session_map.py` | `is_expired()` indexed the session dict directly, raising `KeyError` for an unregistered session | Existence + expiry are checked together (`get_valid_llm()`), returning `None`/`True` instead of raising |
| Deprecated LangChain API | `orchestrators/portable_orchestrator.py` | Called `LLMChain.run()`/`.arun()`, deprecated since LangChain 0.1.0 | Uses `.invoke()`/`.ainvoke()` instead |
| Silent exception swallowing | `lcel/model.py` | A bare `except (ImportError, ValueError, Exception)` discarded all errors during model init | Expected fallback exceptions are logged; anything else is re-raised as `LCELModelException` with the original traceback |
| Wrong base class | `utils/llm_proxy_wrapper.py` | Imported `langchain.llms.BaseLLM`, which only covers legacy completion models | Uses `langchain_core.language_models.BaseLanguageModel`, which covers `BaseChatModel` too |

A follow-up review round on that same PR also found and fixed: `PortableOrchestrator` was passing a hardcoded `{"input": query}` to `.invoke()`/`.ainvoke()`, which broke every real prompt template (the input key has to match the prompt's actual variable name); `run()`/`arun()` crashed with `IndexError` on the kwargs-only multi-input calling convention; and a `SessionMap` singleton-construction race. None of this is present in this branch's code — see that PR for details and current status before relying on this table as a description of the code you're reading.
