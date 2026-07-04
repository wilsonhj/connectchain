## Summary

<!-- What does this PR do? One paragraph max. -->

## Changes

<!-- Bullet list of specific changes made -->

- 
- 

## Type of Change

- [ ] 🐛 Bug fix (BUG-X: ___)
- [ ] ✨ New feature (F-X: ___)
- [ ] ♻️ Refactor (REF-X: ___)
- [ ] 📝 Documentation
- [ ] ⚡ Optimization
- [ ] 🔧 Chore / dependency bump

## Related Issues / Roadmap Items

Fixes # <!-- issue number -->
Roadmap: <!-- e.g. ROADMAP.md §BUG-1 -->

---

## Reviewer Checklist

### A — Problem Decomposition
- [ ] The PR solves one clearly scoped problem (not a grab-bag of unrelated changes)
- [ ] The approach is explained in the summary — not just *what* changed but *why*
- [ ] Edge cases are identified and either handled or documented as follow-up

### B — Code Abstraction & Design
- [ ] No duplication — new logic reuses or extends existing abstractions
- [ ] Public API surface is minimal (no unnecessary exposure of internals)
- [ ] LCEL migration path followed (no new `LLMChain` instantiations)
- [ ] Sanitizers are pure functions `(str) -> str` with no side effects beyond raising

### C — Debugging & Correctness
- [ ] Unit tests added/updated for all changed logic
- [ ] Tests cover the happy path, an error path, and at least one edge case
- [ ] No new `KeyError` / `AttributeError` silent failure modes introduced
- [ ] Async code uses `await` correctly — no sync calls blocking the event loop
- [ ] `make test` passes locally (`uv run pytest`)

### D — Technical Communication
- [ ] Commit message follows `<type>(<scope>): <summary>` convention
- [ ] Inline comments explain *why*, not *what* (the code explains what)
- [ ] Public functions have docstrings with Args / Returns / Raises
- [ ] ARCHITECTURE.md or ROADMAP.md updated if this changes system design

### E — Domain Knowledge (LangChain / AI)
- [ ] No use of deprecated LangChain APIs (`LLMChain`, `arun()`, old callback signatures)
- [ ] LangChain version compatibility verified (target: 0.3.x, forward-compatible with 0.4.x)
- [ ] If MCP tools changed: MCP server contract still satisfied (tool names, input schema)
- [ ] If sanitizers changed: both input (`ValidPromptTemplate`) and output paths tested

---

## Screenshots / Logs

<!-- Paste relevant test output, logs, or before/after examples -->

```

```

---

## Post-Merge

- [ ] ROADMAP.md item marked as complete
- [ ] Downstream teams notified if public API changed
- [ ] Version bumped if releasing (`python version_bump.py`)
