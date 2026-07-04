# ConnectChain Fork — Product Roadmap

> **Fork owner:** `wilsonhj` | **Horizon:** Q3 2026 – Q1 2027
> **Priority scale:** 🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Nice-to-have

---

## Summary

| ID | Type | Title | Priority | Target |
|----|------|-------|----------|--------|
| BUG-1 | Bug Fix | `SessionMap.is_expired()` KeyError on first call | 🔴 Critical | Q3 2026 W1 |
| BUG-2 | Bug Fix | `ValidLLMChain` output_sanitizer runs on input, not output | 🔴 Critical | Q3 2026 W1 |
| BUG-3 | Bug Fix | `PortableOrchestrator` calls deprecated `arun()` | 🟠 High | Q3 2026 W2 |
| BUG-4 | Bug Fix | Missing `cert_size` default causes `KeyError` | 🟠 High | Q3 2026 W2 |
| BUG-5 | Bug Fix | `MCPToolAdapter.call_tool()` blocks event loop | 🟠 High | Q3 2026 W2 |
| REF-1 | Refactor | Migrate `LLMChain` → LCEL pipes across all modules | 🔴 Critical | Q3 2026 W3–4 |
| F-1 | Feature | `MCPToolAgent` with LangGraph checkpointing (multi-turn memory) | 🟠 High | Q3 2026 |
| F-2 | Feature | Async parallel tool dispatch via `asyncio.gather` | 🟠 High | Q3 2026 |
| F-3 | Feature | `astream()` / streaming response support | 🟠 High | Q3 2026 |
| F-4 | Feature | Structured output enforcement (`with_structured_output`) | 🟡 Medium | Q4 2026 |
| F-5 | Feature | Per-request retry + fallback model routing | 🟡 Medium | Q4 2026 |
| F-6 | Feature | Prompt injection detection sanitizer (built-in) | 🟡 Medium | Q4 2026 |
| F-7 | Feature | Agent team orchestration (supervisor + worker pattern) | 🟢 Explore | Q4 2026 |
| OPT-1 | Optimization | Session token caching with `asyncio.Lock` (thread safety) | 🟡 Medium | Q4 2026 |
| OPT-2 | Optimization | Config hot-reload without restart | 🟢 Nice-to-have | Q1 2027 |

---

## Bug Fixes

### BUG-1 — `SessionMap.is_expired()` KeyError 🔴

**File:** `connectchain/utils/session_map.py`
**Symptom:** `KeyError` thrown on the very first call to `is_expired()` because `_sessions` dict is empty.
**Fix:**
```python
def is_expired(self, key: str) -> bool:
    if key not in self._sessions:   # ← ADD THIS GUARD
        return True
    return time.time() > self._sessions[key]['expires_at']
```
**Test:** `tests/unit_tests/test_session_map.py::test_is_expired_first_call`

---

### BUG-2 — `ValidLLMChain` sanitizer on wrong side 🔴

**File:** `connectchain/chains/valid_llm_chain.py`
**Symptom:** `output_sanitizer` is invoked inside `prep_inputs()`, meaning it runs on the user's *prompt*, not the LLM's *response*. Any output-level PII scrubbing or code validation is silently skipped.
**Fix:** Move the sanitizer call from `prep_inputs()` → `prep_outputs()` (or equivalent post-invoke hook).
**Impact:** Teams relying on this for output safety have zero protection today.

---

### BUG-3 — Deprecated `arun()` in PortableOrchestrator 🟠

**File:** `connectchain/orchestrators/portable_orchestrator.py`
**Fix:** Replace `LLMChain.arun(input)` with LCEL `chain.ainvoke({"input": input})`
**Upstream removal:** LangChain 0.4.x (est. Q3 2026)

---

### BUG-4 — Missing `cert_size` default 🟠

**File:** `connectchain/config/loader.py`
**Fix:** Add `cert_size: int = 2048` to `CertConfig` dataclass so configs without an explicit cert block don't `KeyError`.

---

### BUG-5 — Sync `call_tool()` blocks event loop 🟠

**File:** `connectchain/tools/mcp/adapter.py`
**Fix:**
```python
async def call_tool(self, tool_name: str, args: dict) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self._sync_call_tool, tool_name, args)
```

---

## Refactoring

### REF-1 — LCEL Migration (LangChain 0.4.x Compatibility) 🔴

**Scope:** All files that subclass or directly instantiate `LLMChain`
**Why:** `LangChain 0.4.x` removes `LLMChain` entirely. ConnectChain will break at import time.

**Migration plan:**

1. **`ValidLLMChain`** → Custom `Runnable` subclass wrapping LCEL pipe + sanitizer
2. **`PortableOrchestrator`** → Rebuild as `prompt | llm | RunnableLambda(sanitizer)`
3. **Token injection** → Move from `__init__` monkey-patch to `RunnableConfig` headers

```python
# Target API (backward compatible)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

class ValidChain:
    def __init__(self, llm, prompt, output_sanitizer=None):
        pipe = prompt | llm
        if output_sanitizer:
            pipe = pipe | RunnableLambda(output_sanitizer)
        self._chain = pipe

    def invoke(self, inputs: dict) -> str:
        return self._chain.invoke(inputs)

    async def ainvoke(self, inputs: dict) -> str:
        return await self._chain.ainvoke(inputs)
```

---

## New Features

### F-1 — MCPToolAgent + LangGraph Checkpointing 🟠

**Problem:** Current `MCPToolAgent` is stateless — no memory between turns, no interrupt/resume.
**Solution:** Integrate LangGraph `StateGraph` with `MemorySaver` checkpointer.

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

class StatefulMCPAgent:
    def __init__(self, tools, llm):
        self.graph = self._build_graph(tools, llm)
        self.checkpointer = MemorySaver()

    def run(self, query: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke({"messages": [query]}, config=config)
```

**Real-world use case:** Multi-turn financial Q&A agent that remembers account context across a session without re-fetching.

---

### F-2 — Parallel Tool Dispatch 🟠

**Problem:** Multi-tool agents call tools sequentially, creating latency proportional to tool count.
**Solution:** `asyncio.gather()` for independent tool calls.

```python
async def parallel_dispatch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
    tasks = [self.call_tool(tc.name, tc.args) for tc in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Expected speedup:** 3–5× for agents calling 3+ independent tools.

---

### F-3 — Streaming Response (`astream()`) 🟠

**Problem:** `PortableOrchestrator` only supports blocking `.run()`.
**Solution:** Expose `astream()` for token-by-token streaming to the application layer.

```python
async def astream(self, input: str):
    async for chunk in self._chain.astream({"input": input}):
        yield chunk
```

---

### F-4 — Structured Output Enforcement 🟡

```python
from pydantic import BaseModel

class TransactionSummary(BaseModel):
    amount: float
    currency: str
    merchant: str
    risk_score: float

chain = prompt | llm.with_structured_output(TransactionSummary)
result: TransactionSummary = chain.invoke({"transaction": raw_data})
```

---

### F-5 — Retry + Fallback Model Routing 🟡

```python
from langchain_core.runnables import RunnableWithFallbacks

primary_chain = prompt | primary_llm
fallback_chain = prompt | fallback_llm

resilient_chain = primary_chain.with_fallbacks([fallback_chain])
```

---

### F-6 — Built-in Prompt Injection Detection 🟡

```python
INJECTION_PATTERNS = [
    r'ignore previous instructions',
    r'disregard (all|your) (prior|previous|system)',
    r'you are now (a |an )?(?!assistant)',
    r'<\s*script\s*>',
    r'system:\s*you are',
]

def detect_prompt_injection(query: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            raise OperationNotPermittedException("Prompt injection detected")
    return query
```

---

### F-7 — Agent Team Orchestration (Supervisor + Worker) 🟢

```python
def supervisor_node(state):
    decision = supervisor_llm.invoke(state['messages'])
    return Command(goto=decision.next_worker)

graph.add_node("research_agent", research_subgraph)
graph.add_node("code_agent", code_subgraph)
graph.add_conditional_edges("supervisor", route_to_worker)
```

---

## Optimizations

### OPT-1 — Thread-Safe Session Token Cache 🟡

```python
import asyncio

class AsyncSessionMap:
    def __init__(self):
        self._sessions = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_token(self, key: str) -> str:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        async with self._locks[key]:
            if self._is_expired(key):
                self._sessions[key] = await self._refresh_token(key)
            return self._sessions[key]['token']
```

### OPT-2 — Config Hot-Reload 🟢

```python
from watchfiles import awatch

async def watch_config(config_path: str, on_reload: callable):
    async for _ in awatch(config_path):
        on_reload(Config.from_file(config_path))
```
