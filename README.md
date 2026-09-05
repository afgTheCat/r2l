# r2l - a Rust Reinforcement Learning Library

> [!WARNING]  
> **Pre-alpha:** This library is under active development. APIs may change
> between releases.

r2l is a Rust reinforcement learning library focused on on-policy methods such
as A2C and PPO.

## Example usage

The following example uses the optional `gym` integration. Enable it in your
`Cargo.toml`:

```toml
r2l = { version = "0.0.3", features = ["gym"] }
```

With the `gymnasium` Python package installed, a complete training and inference
run looks like this:

```rust,no_run
use r2l::{GymEnv, InferenceRunner, PPOBuilder, TrainingArtifactsConfig, TrainingLimit, Error};

fn main() -> Result<(), Error> {
    const ENV_NAME: &str = "Pendulum-v1";
    const ARTIFACT_DIR: &str = "runs/pendulum";

    let artifacts_config = TrainingArtifactsConfig::new(ARTIFACT_DIR);
    let env_builder = || GymEnv::new(ENV_NAME, None);

    let mut ppo = PPOBuilder::new(env_builder, 10)?
        .with_training_artifacts(artifacts_config)
        .with_policy_hidden_layers(vec![64, 64])
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_training_limit(TrainingLimit::rollouts(30))
        .build()?;

    ppo.train()?;

    let env = GymEnv::new(ENV_NAME, Some("human".to_owned()))?;
    let mut inference = InferenceRunner::load_from_env(ARTIFACT_DIR, env)?;
    for _ in 0..10 {
        inference.run_episode()?;
    }

    Ok(())
}
```

For more information on how to use r2l, read the
[book](https://afgthecat.github.io/r2l/).

## Current capabilities

The current published version is `v0.0.3`.

- On-policy PPO and A2C implementations
- Candle and Burn backends
- Single- and multithreaded rollout collection
- Native `Env` implementations and a Gymnasium adapter for Discrete spaces with
  `start = 0`, plus Box, MultiDiscrete, MultiBinary, Tuple, and Dict spaces
- Policy evaluation and best-policy checkpointing

PPO evaluation runs across 28 environments show broadly consistent learning
behavior between r2l's Candle and Burn backends. See the
[results](https://afgthecat.github.io/r2l/results.html).

# Contributing

Any and all contributions are welcome. If you have a feature request, let me
know by opening an issue about it.
