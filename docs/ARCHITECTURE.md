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
7. [Known Bugs & Root Causes](#7-known-bugs--root-causes)

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
| `connectchain.lcel` | `model()`, `Logger` | LCEL-compatible model factory; logging hooks |
| `connectchain.orchestrators` | `PortableOrchestrator` | Provider-agnostic chain runner |
| `connectchain.prompts` | `ValidPromptTemplate` | Pre-send prompt sanitization |
| `connectchain.chains` | `ValidLLMChain` | Post-response output sanitization (**BUG-2: currently applies to input**) |
| `connectchain.utils` | `get_token_from_env()`, `SessionMap` | EAS JWT, session expiry (**BUG-1: KeyError on first call**) |
| `connectchain.tools.mcp` | `MCPToolAgent` | MCP server integration |
| `connectchain.config` | `Config`, `ModelConfig` | YAML + env config loader |

---

## 3. LangChain Dependency Chain

ConnectChain currently imports from LangChain in three critical paths:

```
connectchain.chains.ValidLLMChain
  └── inherits langchain.chains.LLMChain          ← DEPRECATED in 0.3.x, REMOVED in 0.4.x

connectchain.orchestrators.PortableOrchestrator
  └── calls LLMChain.arun()                       ← DEPRECATED, use .ainvoke()

connectchain.utils (token injection)
  └── monkey-patches ChatOpenAI.__init__           ← breaks on langchain-openai ≥ 0.2.0
```

### Migration Target (LCEL Pipes)

```python
# Before (deprecated)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input)

# After (LCEL)
chain = prompt | llm
result = chain.invoke({"input": input})

# With output sanitizer (LCEL + custom Runnable)
from langchain_core.runnables import RunnableLambda
chain = prompt | llm | RunnableLambda(output_sanitizer)
```

---

## 4. Session & Auth Lifecycle

```
Application calls model() or PortableOrchestrator.run()
        │
        ▼
SessionMap.get(model_index)
  ├── [FIRST CALL — BUG-1] KeyError: session not yet set
  │   Fix: guard with `if model_index not in self._sessions`
  └── [SUBSEQUENT] check is_expired()
          ├── expired → refresh JWT via EAS
          └── valid   → return cached token
                │
                ▼
        Inject token into model headers
                │
                ▼
        LangChain model call (sync or async)
                │
                ▼
        [Optional] output_sanitizer(response)   ← BUG-2: currently runs on input
```

---

## 5. MCP Integration Layer

The `connectchain.tools.mcp` module wraps the Model Context Protocol (MCP) to expose external tool servers as LangChain-compatible tools:

```
MCP Server (stdio / SSE)
        │
        ▼
MCPToolAdapter
  ├── list_tools()    → LangChain Tool objects
  ├── call_tool()     → StructuredTool.run()
  └── stream_tool()   → AsyncGenerator (not yet implemented — roadmap Q3 2026)
        │
        ▼
LangChain AgentExecutor / LangGraph StateGraph
```

### Current Limitation
MCPToolAgent does not support **multi-turn memory** or **LangGraph checkpointing**. Each call is stateless. See ROADMAP.md §Feature F-1.

---

## 6. Upstream Risk Surface

| Risk | Severity | Affected Files | ETA |
|------|----------|---------------|-----|
| `LLMChain` removed in LangChain 0.4.x | 🔴 Critical | `chains/valid_llm_chain.py`, `orchestrators/portable_orchestrator.py`, `utils/` | LangChain 0.4 releasing Q3 2026 |
| `langchain-openai` ≥ 0.2.0 breaks token injection | 🟠 High | `utils/token_utils.py` | Already landed upstream |
| `arun()` deprecated → `ainvoke()` | 🟡 Medium | `orchestrators/portable_orchestrator.py` | LangChain 0.3.x |
| Callback API signature changed | 🟡 Medium | `lcel/logger.py` | LangChain 0.2.x |

Track upstream changes: [langchain-ai/langchain releases](https://github.com/langchain-ai/langchain/releases)

---

## 7. Known Bugs & Root Causes

| ID | File | Root Cause | Fix Summary |
|----|------|-----------|-------------|
| BUG-1 | `utils/session_map.py` | `is_expired()` reads `_sessions[key]` before key exists on first call | Add `if key not in self._sessions: return True` guard |
| BUG-2 | `chains/valid_llm_chain.py` | `output_sanitizer` applied to `inputs` dict, not `outputs` | Move sanitizer call from `prep_inputs()` to `prep_outputs()` |
| BUG-3 | `orchestrators/portable_orchestrator.py` | Calls deprecated `LLMChain.arun()` | Replace with LCEL `chain.ainvoke()` |
| BUG-4 | `config/loader.py` | Missing `cert_size` default — raises `KeyError` on configs without cert block | Add `cert_size: int = 2048` to `CertConfig` dataclass |
| BUG-5 | `tools/mcp/adapter.py` | Synchronous `call_tool()` blocks the event loop when called from async context | Wrap with `asyncio.run_in_executor()` or make native `async def` |
