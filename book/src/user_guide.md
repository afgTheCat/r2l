## Getting started

For most applications, `r2l-api` is the main dependency. It provides complete
PPO and A2C builders while the lower-level workspace crates define
environments, samplers, agents, and backend integrations.

The shortest Gymnasium-based PPO setup is:

```rust
use r2l_api::{LearningSchedule, PPOAlgorithmBuilder, StepHookBound};

fn main() {
    let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
        .with_burn()
        .with_rollout_bound(StepHookBound::new(1024))
        .with_learning_schedule(LearningSchedule::total_step_bound(100_000));
    let mut algorithm = builder.build().unwrap();
    algorithm.train().unwrap();
}
```

This requires Python 3.11 or newer and the `gymnasium` Python package. A
Gymnasium environment id is passed to `GymEnvBuilder`, which maps supported
Gymnasium spaces to `r2l` space descriptions.

## Environments

Native environments implement `Env`. An environment returns an initial
observation from `reset`, accepts one action in `step`, and describes its
observation and action spaces.

```rust,noplayground
{{#include ../../crates/r2l-core/src/env/mod.rs:env}}
```

Algorithms receive an `EnvBuilder` rather than a concrete environment so that
each sampler worker can construct its environment in the place where it runs.
A closure or function returning `anyhow::Result<E>` automatically implements
`EnvBuilder`.

The complete workspace example demonstrates a custom environment, a custom
builder, a function builder, a closure builder, and `GymEnvBuilder`:

```rust,noplayground
{{#include ../../crates/r2l-examples/examples/env_building/main.rs:env_builders}}
```

`r2l-gym` currently maps `Discrete`, `Box`, `MultiDiscrete`, `MultiBinary`,
`Tuple`, and `Dict` spaces. Structured observations are flattened into
`TensorData`; discrete observations are one-hot encoded.

## Backends

The algorithm builders default to Candle. Use `with_candle(device)` to choose a
Candle device explicitly, or `with_burn()` to use the default Burn
autodifferentiation backend.

Backend selection changes the concrete builder type, so call it before methods
that are specific to a chosen backend when following compiler suggestions.

## Rollout collection

`StepHookBound::new(n)` collects `n` steps per environment for each rollout.
`EpisodeHookBound::new(n)` collects `n` completed episodes per environment.
Install either with `with_rollout_bound`.

The default `SamplerExecutionMode::SingleThreaded` steps workers sequentially
on the calling thread. `SamplerExecutionMode::MultiThreaded` runs workers on
dedicated threads:

```rust
use r2l_api::{PPOAlgorithmBuilder, SamplerExecutionMode};

let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
    .with_execution_mode(SamplerExecutionMode::MultiThreaded);
```

Gymnasium calls still execute under Python's interpreter lock, so threaded
execution should not be assumed to improve Python-environment throughput.

Observation normalization is enabled with
`with_observation_normalizer(Some(clip))`; `None` disables it. Step-bounded
rollouts can also normalize discounted rewards with
`with_reward_normalizer(gamma, clip_reward)`.

## Training schedules

`LearningSchedule::rollout_bound(n)` stops after `n` rollout collections.
`LearningSchedule::total_step_bound(n)` stops after at least `n` sampled
environment steps across all workers.

`LearningRateSchedule::Constant(rate)` keeps the configured rate fixed.
`LearningRateSchedule::Linear(rate)` decays it from `rate` to zero over the
configured training schedule.

## Complete examples

The PPO example covers Burn training, rollout configuration, saving the best
actor, loading it from SafeTensors, and evaluation:

```rust,noplayground
{{#include ../../crates/r2l-examples/examples/ppo/main.rs:ppo}}
```

The A2C example covers backend selection, statistics reporting, and a Candle
configuration:

```rust,noplayground
{{#include ../../crates/r2l-examples/examples/a2c/main.rs:a2c}}
```

For the underlying traits and hook points, continue with
[On-policy algorithms](./on_policy_algorithms.md). For exact builder methods,
use the [`r2l-api` reference](https://docs.rs/r2l-api/0.0.2/r2l_api/).
