---
name: 19labs-agent
description: Agent operating guide for the 19Labs AutoML project.
---

# Agent

## Purpose
Define how an automated coding agent should operate in this repository.

## Core Rules
- Keep changes small, targeted, and easy to review.
- Preserve existing API behavior unless a task explicitly asks for a change.
- Favor existing patterns in `engine.py`, `server.py`, and `mcp/`.
- Avoid broad refactors while fixing localized issues.
- Never ask clarifying questions when data context (CSV, profile, domain analysis) is already available.

## Workflow
1. Gather context from relevant files before editing.
2. Implement the smallest viable change.
3. Validate with `python -c "import engine; import server"` for syntax.
4. Summarize what changed and any remaining risks.

## ML / Data Conventions
- Experiment 1 is always AutoGluon `best_quality` — never replace it with a weaker baseline.
- All LLM code generation for exp 2+ must end with `joblib.dump(model, 'model.pkl')` for the FULL pipeline.
- Always parse raw signal strings to human language before passing to `narrate()` or the frontend.
- Use `_domain_analysis_text(da)` everywhere a dict might be passed as a string.

## Validation
- Prefer targeted checks first, then broader checks if needed.
- If validation cannot run, clearly state what was skipped and why.

## Notes
- Update this file when project conventions change.
