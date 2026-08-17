# Getting started

For most applications, `r2l` is the main dependency. It provides complete PPO
and A2C builders while the lower-level workspace crates define environments,
samplers, agents, and backend integrations. This getting started guide will be
solely using the `r2l` crate, which itself builds on lower-level crates. If the
current setup does not satisfy you, the lower level hooks allow for a lot of
hackability.

## Shortest setup

The shortest Gymnasium-based PPO setup is:

```rust,no_run
# extern crate r2l;
use r2l::PPOBuilder;

fn main() {
    let builder = PPOBuilder::gym("Pendulum-v1", 4);
    let mut algorithm = builder.build().unwrap();
    algorithm.train().unwrap();
}
```

This requires Python 3.11 or newer and the `gymnasium` Python package. A
Gymnasium environment id is passed to `GymEnvBuilder`, which maps supported
Gymnasium spaces to `r2l` space descriptions. The builders support other
environment types through a different construction, which will be introduced
later on.

## Saving training artifacts

We rarely want to train algorithms for the sake of it. Once an algorithm is
learned, we are usually curious about

- how the chosen hyperparameters affect learning performance
- how to run inference using the model
- less often, maybe we are curious about the running performance of the
  algorithm

`r2l` allows saving these artifacts using the `with_training_artifacts` builder
method.

```rust,no_run
# extern crate r2l;
use r2l::{PPOBuilder, TrainingArtifactsConfig};

fn main() {
    let builder = PPOBuilder::gym("Pendulum-v1", 4);
    let artifacts_config = TrainingArtifactsConfig::new("runs/pendulum")
        .with_evaluation_results(true)
        .with_training_timings(true)
        .with_inference_artifacts(true);
    let mut algorithm = builder
        .with_training_artifacts(artifacts_config)
        .build()
        .unwrap();
    algorithm.train().unwrap();
}
```

Under the hood, `r2l` will evaluate the performance of the policy by launching a
clean environment after each training round, and remember the best performing
policy. These evaluation settings can be customized as well, by setting
`EvaluationSettings`. Once training is done the following new files are created
in the artifacts folder:

```sh
$ tree runs/pendulum
runs/pendulum
├── actor.safetensors
├── evaluations.csv
├── inference.yaml
└── performance.csv

1 directory, 4 files
```

## Running inference

Using the inference artifacts and a new environment, you can create an
`InferenceRunner`.

```rust,no_run
# extern crate r2l;
# extern crate r2l_gym;
use r2l::InferenceRunner;
use r2l_gym::GymEnv;

fn main() {
    let env = GymEnv::new("Pendulum-v1", Some("human".to_owned())).unwrap();
    let mut inference = InferenceRunner::load("runs/pendulum", env).unwrap();
    for _ in 0..4 {
        inference.run_episode().unwrap();
    }
}
```

The inference configuration describes only the policy shape and observation
normalization settings. It does not contain information about how the policy was
trained.

# Environments

Environments implement the `Env` trait.

```rust,ignore
{{#include ../../crates/r2l-core/src/env/mod.rs:env}}
```

Algorithm builders receive an `EnvBuilder` rather than a concrete environment so
that each sampler worker can construct its environment in the place where it
runs.

```rust,ignore
{{#include ../../crates/r2l-core/src/env/mod.rs:env_builder}}
```

A closure or function returning `anyhow::Result<E>` automatically implements
`EnvBuilder`.

```rust,ignore
let env_builder = || Ok(MyEnv);
let ppo_builder = PPOBuilder::new(env_builder, 10);
let ppo = ppo_builder.build().unwrap();
```

For a more detailed example of how to implement the `Env` and `EnvBuilder`
traits, see the [environment building example](./examples/env_building.md).

# Hyperparameters

Both the PPO and A2C builders expose a great deal of hyperparameters that can be
tuned.

## Backends

The algorithm builders default to Candle. Use `with_candle(device)` to choose a
Candle device explicitly, or `with_burn()` to use the default Burn
autodifferentiation backend.

Backend selection changes the concrete builder type, so call it before methods
that are specific to a chosen backend when following compiler suggestions.

## Rollout collection

`with_rollout_steps(n)` collects `n` steps per environment for each rollout.
`with_rollout_episodes(n)` switches to episode-bounded sampling and collects `n`
completed episodes per environment.

The builders default to `SamplerExecutionMode::MultiThreaded`, which runs
workers on dedicated threads. Use `SamplerExecutionMode::SingleThreaded` to step
workers sequentially on the calling thread:

```rust,no_run
# extern crate r2l;
use r2l::{PPOBuilder, SamplerExecutionMode};

let builder = PPOBuilder::gym("Pendulum-v1", 4)
    .with_execution_mode(SamplerExecutionMode::SingleThreaded);
```

Gymnasium calls still execute under Python's interpreter lock, so threaded
execution should not be assumed to improve Python-environment throughput.

Observation normalization is enabled with
`with_observation_normalizer(Some(clip))`; `None` disables it. Step-bounded
rollouts can also normalize discounted rewards with
`with_reward_normalizer(gamma, clip_reward)`.

## Training schedules

`TrainingLimit::rollouts(n)` stops after `n` rollout collections.
`TrainingLimit::steps(n)` stops after at least `n` sampled
environment steps across all workers.

`LearningRateSchedule::Constant(rate)` keeps the configured rate fixed.
`LearningRateSchedule::Linear(rate)` decays it from `rate` to zero over the
configured training schedule.

For the underlying traits and hook points, continue with
[On-policy algorithms](./on_policy_algorithms.md). For exact builder methods,
use the [`r2l` reference](https://docs.rs/r2l/0.0.2/r2l/).
