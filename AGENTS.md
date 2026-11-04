# Codex Working Guidelines

## Scope
- Inspect only files relevant to the current task.
- Avoid repository-wide searches unless necessary.
- Prefer minimal, targeted changes over broad refactoring.
- Do not modify unrelated files.
- Ask before expanding the task scope.

## Implementation
- Preserve existing behavior unless explicitly asked to change it.
- Prefer existing project patterns and dependencies.
- For behavior changes, write focused tests first when practical.
- Do not commit or push unless explicitly requested.

## Verification
- Run focused tests first.
- Run Ruff and relevant checks after code changes.
- Avoid repeated full test runs without a reason.
- Do not repeat searches or commands unnecessarily.

## Communication
- Keep explanations concise.
- Explain important design decisions and trade-offs.
- Clearly distinguish verified results from assumptions.
- When asked to analyze or review, do not modify files.