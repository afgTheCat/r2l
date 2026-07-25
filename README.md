# r2l - a Rust Reinforcement Learning Library

> [!WARNING]  
> **Pre-alpha:** This library is under active development. APIs may change
> between releases.

## Why **r2l**

The goal of **r2l** is to be a customizable, ergonomic and easily embeddable
library. To be more exact:

- **Customizable**: users have fine-grained control over _how_ agents are
  trained. **r2l** defines how the components interact while exposing training
  lifecycle hooks for application-specific behavior.
- **Ergonomic**: most users are not necessarily concerned with implementation
  details. High-level builders provide common configurations without requiring
  a complete algorithm implementation.
- **Embeddable**: **r2l** uses traits instead of requiring one deep-learning
  framework. Candle and Burn backends are currently supported.

The scope of **r2l** is what Stable Baselines3 covers (by version 0.1.0) and
Tianshou (by version 1.0.0). On top of core algorithms, a hyperparameter tuning
library is to be included in the future.

## Getting started

You can get started if you have `gymnasium` like so:

```rust,no_run
use r2l_api::{LearningSchedule, PPOAlgorithmBuilder, StepHookBound};

fn main() {
    let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
        .with_burn()
        .with_clip_range(0.2)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_total_epochs(10)
        .with_rollout_bound(StepHookBound::new(1024))
        .with_learning_schedule(LearningSchedule::total_step_bound(100_000));
    let mut algo = builder.build().unwrap();
    algo.train().unwrap();
}
```

For more information, read the [book](https://afgthecat.github.io/r2l/).

## v0.0.2 capabilities

- On-policy PPO and A2C builders, plus lower-level PPO, A2C, and VPG
  implementations
- Candle and Burn policy/value backends
- Inline and threaded rollout workers
- Step- and episode-bounded rollout hooks
- Observation normalization, discounted-reward normalization, and linear
  learning-rate schedules
- Native `Env` implementations and a Gymnasium adapter for Discrete, Box,
  MultiDiscrete, MultiBinary, Tuple, and Dict spaces
- Best-actor evaluation and SafeTensors persistence for backend-specific
  policies

Off-policy algorithms, a stable public API, and claimed benchmark parity with
Stable Baselines3 are outside the v0.0.2 release. The configurations in
`envs_to_test.txt` are a benchmark plan, not a record of completed or passing
training runs.

## Roadmap

**Current version: `v0.0.2`.** The project is in an early experimental phase.
Expect missing features, frequent breaking changes, bugs, and everything in
between.

### `v0.1.0` – Core Algorithm Coverage (SB3 parity)

- Implement all algorithms available in
  [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Add benchmarks for simple environments (e.g. CartPole, Pendulum)
- Introduce a high-level builder API for setting up agents with established
  hooks for logging, observability, and training control
- Expect significant API changes

### `v1.0.0` – Extended Algorithm Set (Tianshou parity)

- Implement remaining algorithms from
  [Tianshou](https://github.com/thu-ml/tianshou)
- Finalize the hook and training APIs
- Provide stable interfaces for embedding, visualization, and training control
- Improve documentation, examples, and possibly add multi-agent support

**Future directions may include:**

- Snapshotting via SafeTensors or ONNX
- TensorBoard integration for monitoring

# Contributing

Any and all contributions are welcome. If you have a feature request, let me
know by opening an issue about it, but please understand that while the project
is ambitious, there are no corporate backers and I work on it in my spare time.
