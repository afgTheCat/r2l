# Agent Guidance for `r2l`

## Scope

This repository is a pre-alpha Rust workspace for reinforcement learning libraries, backend integrations, and runnable examples. Prefer small, targeted changes that respect the existing crate boundaries instead of broad cross-workspace refactors. Public APIs are still evolving, but preserve them unless the task explicitly requires a change.

## Workspace Architecture

- `crates/r2l-core`: shared contracts and backend-neutral training orchestration. It owns environment and tensor traits, model traits, trajectory types, the `Agent` and `Sampler` contracts, adapters, and the outer on-policy training loop. It does not own concrete PPO/A2C loss loops or sampler workers.
- `crates/r2l-sampler`: concrete rollout collection. It owns direct and observation-normalized samplers, vector and threaded worker pools, rollout bounds, sampler hooks, and per-environment trajectory buffers.
- `crates/r2l-agents`: concrete on-policy learning logic. It owns PPO, A2C, and VPG implementations, their algorithm-specific hooks and parameters, generalized advantage estimation, returns, log-probability capture, and minibatch sampling.
- `crates/r2l-candle`: Candle policies, action distributions, networks, optimizers, and on-policy learning-module implementations.
- `crates/r2l-burn`: Burn policies, action distributions, networks, optimizers, and on-policy learning-module implementations.
- `crates/r2l-gym`: the Python Gymnasium adapter. It converts Gym spaces, observations, and actions to and from `r2l-core` types.
- `crates/r2l-api`: the high-level composition layer and primary user-facing facade. It combines environments, samplers, agents, backends, default hooks, evaluators, and builders for common PPO/A2C setups.
- `crates/r2l-examples`: runnable integration examples and TUI/GUI demos. Keep examples easy to inspect and favor the public API where possible.
- `crates/r2l-macros`: experimental hook-related procedural macros. It is currently a standalone workspace member and is not used by the main training stack.
- `book`: user and architecture documentation. Treat it as helpful context, but verify architectural claims against the code because parts of the book may lag behind the pre-alpha APIs.

The usual dependency direction is toward `r2l-core`: sampler, agent, Gym, and backend crates implement its contracts; `r2l-api` composes those implementations; examples consume the public crates. Avoid making lower-level crates depend on `r2l-api` or examples.

## Core Concepts

- `R2lTensor` is the tensor interchange contract shared by environments, actors, buffers, and algorithms. `TensorData` is the simple owned flat-data implementation used by the Gym adapter. Candle and Burn tensor implementations are feature-gated in `r2l-core`.
- `Env` produces a `Snapshot` containing the next state, reward, and separate `terminated` and `truncated` flags. `EnvBuilder` constructs fresh environments for sampler workers.
- `Actor` is the inference-time action surface passed to samplers. `Policy` extends it with train-time log-probability, entropy, standard-deviation, and noise behavior.
- `Agent` owns learning behavior and produces an actor snapshot. Concrete PPO/A2C/VPG agents live in `r2l-agents` and learn from `TrajectoryBatch` values.
- `OnPolicyLearningModule` joins a policy, value function, optimizer update, and tensor conversion. Its `InferenceTensor` is used for rollouts; its `LearningTensor` is used for differentiable backend computations. Preserve this boundary when adding backend support.
- `Sampler` owns rollout collection and exposes borrowed `TrajectoryView` values. The standard samplers keep one aligned trajectory buffer per environment.
- `OnPolicyAdapters` bridge actor and trajectory tensor types between an agent and sampler. `DefaultAdapter` performs the standard `R2lTensor` conversion when their tensor types differ.

## On-Policy Training Lifecycle

The outer loop is implemented in `crates/r2l-core/src/on_policy/algorithm.rs`:

1. The algorithm runs its initialization hook.
2. The runtime asks the agent for an actor snapshot, adapts it for the sampler, and collects fresh rollouts.
3. The post-rollout hook updates outer-loop state such as sampled-step or rollout counts.
4. The sampler exposes trajectory views, the adapter converts them if needed, and the agent learns from them.
5. The post-training hook may evaluate the actor and decide whether the outer loop stops.
6. The shutdown hook releases evaluator, agent, sampler, and worker resources.

There are three separate hook layers; put behavior at the layer whose lifecycle it affects:

- Sampler hooks in `r2l-sampler` and `r2l-api/src/hooks/sampler.rs` control rollout collection bounds.
- `OnPolicyAlgorithmHooks` control the outer collect/learn lifecycle, training schedules, evaluation, and shutdown.
- PPO/A2C hooks in `r2l-agents` customize algorithm learning; their default implementations in `r2l-api` add behavior such as advantage normalization, entropy/value coefficients, gradient clipping, statistics, and PPO epoch control.

## Behavioral Invariants

- Keep `terminated` and `truncated` separate throughout environments and trajectory storage. Code may use their disjunction to detect an episode boundary, but callers can need the original distinction.
- Keep every field in a trajectory buffer aligned: state, next state, action, reward, termination, and truncation entries describe the same transition at each index.
- Rollout collection clears the sampler's buffers before installing the current actor and collecting a new batch. Do not assume buffers accumulate across outer-loop iterations.
- Sampler workers reset an environment after either termination or truncation and continue collecting according to the configured step or episode bound.
- `Space::size` and Gym structured-space conversion use flattened sizes. Preserve explicit tensor shapes at `R2lTensor` boundaries and document any new batching convention.
- Actors used by the standard samplers must be cloneable so each environment worker can receive a policy snapshot; actors sent to threaded workers must also satisfy the `Actor` sendability contract.
- The default GAE path currently treats both termination and truncation as non-bootstrapping episode boundaries. Call out and test any change to that behavior.
- When observation normalization is enabled, evaluation must reuse training observation statistics in read-only mode; evaluation must not update the shared running statistics.

## Where Changes Belong

- Add or change shared traits, transition types, adapters, or outer-loop orchestration in `r2l-core`.
- Add rollout execution, worker coordination, buffer-filling, or normalization behavior in `r2l-sampler`.
- Add algorithm math, advantage/return processing, minibatching, or algorithm-specific learning hooks in `r2l-agents`.
- Add framework-specific distributions, networks, tensor operations, serialization, or optimizer behavior in `r2l-candle` or `r2l-burn`.
- Add ergonomic builders, defaults, reporting hooks, evaluators, and cross-crate assembly in `r2l-api`.
- Add Gymnasium parsing or Python interop in `r2l-gym`.
- Demonstrate end-to-end usage in `r2l-examples`; do not move reusable library logic into an example.

Useful starting points are:

- Contracts and lifecycle: `crates/r2l-core/src/lib.rs`, `env.rs`, `models.rs`, and `on_policy/algorithm.rs`
- Trajectories: `crates/r2l-core/src/buffers/`
- Algorithm math: `crates/r2l-agents/src/on_policy_algorithms/`
- Collection: `crates/r2l-sampler/src/direct/` and `normalized/`
- High-level assembly: `crates/r2l-api/src/builders/on_policy.rs` and algorithm-specific builder modules
- Backend implementations: each backend crate's `distributions/` and `learning_module.rs`

## Working Style

- Make the smallest effective diff and avoid touching unrelated code.
- Prefer simple, clear code over clever abstractions.
- Prefer iterator-based solutions when they improve clarity without creating dense chains.
- Prefer associated functions over free-standing functions when there is a natural type association.
- Avoid incidental renames, formatting churn, or workspace-wide rewrites.
- Keep crate responsibilities clear; do not move logic across crates without a strong reason.
- Add brief comments only where the intent would otherwise be hard to infer.
- If a task changes RL behavior, state assumptions about rollout boundaries, bootstrapping, hooks, tensor shapes, or environment interaction and add focused tests where practical.

## Validation

Start with the narrowest useful validation for the affected crate, then widen only when the change crosses crate boundaries:

- `cargo check -p <crate>`
- `cargo test -p <crate>`
- `cargo fmt --check`
- `cargo clippy -p <crate> --all-targets -- -D warnings`

For cross-cutting changes, use:

- `cargo check --workspace`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

Gym-backed tests require a compatible Python and Gymnasium installation, and full learning tests can be long-running. Backend or CUDA changes may require feature-specific checks in addition to the default workspace configuration.
