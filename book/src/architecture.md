# Architecture

R2l currently only supports on-policy algorithms. Off-policy algorithms is something that we are going to support before v0.1.0.

## On policy algorithms

The on `OnPolicyAlgorithm` is intentionally kept minimal. It consists of the following components:

- The `sampler`
- The `agent`
- The `init`, `post_rollout`, `post_training` and `shutdown` hooks

![On policy algo overview](../images/on_policy_algo.png)

Supported algorithms and capabilities are defined by implementations of these components. The training loop implementation is
```rust
{{#include ../../../crates/r2l/src/on_policy.rs:train_loop}}
```

## Off policy algorithms
TODO
