# Algorithms

The algorithm examples show complete PPO and A2C configurations through
`r2l`. They are useful when you want a runnable training loop before
customizing lower-level hooks or samplers.

## PPO

The PPO example trains on `Pendulum-v1`, writes training artifacts, reloads the
best policy through `InferenceRunner`, and runs rendered inference episodes.

Run it from the workspace root:

```sh
cargo run -p r2l-examples --example ppo
```

```rust,ignore
{{#include ../../../crates/r2l-examples/examples/ppo/main.rs:ppo}}
```

The important pieces are `with_training_artifacts`, which writes
`actor.safetensors`, `inference.yaml`, and metrics files, and
`InferenceRunner::load_from_env`, which rebuilds the inference runner from those
saved files.

## A2C

The A2C example selects the Candle backend, configures rollout collection, and
uses a reporter channel to observe training statistics.

Run it from the workspace root:

```sh
cargo run -p r2l-examples --example a2c
```

```rust,ignore
{{#include ../../../crates/r2l-examples/examples/a2c/main.rs:a2c}}
```

Both PPO and A2C builders expose the same broad setup concepts: choose an
environment builder, select a backend, set rollout bounds, configure a learning
schedule, then call `build()` and `train()`.
