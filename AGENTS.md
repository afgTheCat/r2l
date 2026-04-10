# Agent Guidance for `r2l`

## Scope

This repository is a Rust workspace for reinforcement learning libraries and examples. Prefer small, targeted changes that fit the existing crate boundaries instead of broad cross-workspace refactors.

## Repo Map

- `crates/r2l-core`: core RL abstractions and on-policy algorithms
- `crates/r2l-gym`: wrappers around Gym-style environments
- `crates/r2l-agents`: agent-related abstractions and implementations
- `crates/r2l-macros`: procedural macros
- `crates/r2l-examples`: runnable examples and integration demos
- `crates/r2l-candle`: Candle-based LM integrations
- `crates/r2l-burn`: Burn-based LM integrations
- `crates/r2l-api`: shared API-facing types and interfaces

## Working Style

- Prefer the smallest change that solves the task.
- Make the smallest effective diff and avoid touching unrelated code.
- Prefer simple, clear code over clever abstractions.
- Prefer iterator-based solutions when they improve clarity and do not make the code harder to read.
- Preserve public APIs unless the task explicitly calls for API changes.
- Avoid incidental renames, formatting churn, or workspace-wide rewrites.
- Keep crate responsibilities clear; do not move logic across crates without a strong reason.
- Add brief comments only where the intent would otherwise be hard to infer.

## Validation

Start with the narrowest useful validation for the affected crate, then widen only if needed.

- `cargo check -p <crate>`
- `cargo test -p <crate>`
- `cargo fmt --check`
- `cargo clippy -p <crate> --all-targets -- -D warnings`

For cross-cutting changes, use:

- `cargo check --workspace`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

## Notes

- The project is pre-alpha. Favor clarity and maintainability over premature abstraction.
- Examples and demos should remain runnable and easy to inspect.
- If a task touches RL algorithm behavior, call out any assumptions about training loops, hooks, or environment interaction explicitly.
