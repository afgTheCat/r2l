# r2l - a Rust Reinforcement Learning Library

> [!WARNING]  
> **Pre-alpha:** This library is under active development. APIs may change
> between releases.

## Getting started

With `gymnasium` installed, a complete training and inference workflow looks
like this:

```rust,no_run
use r2l::{InferenceRunner, PPOBuilder, TrainingArtifactsConfig, TrainingLimit};
use r2l_gym::GymEnv;

const ENV_NAME: &str = "Pendulum-v1";
const ARTIFACT_DIR: &str = "runs/pendulum";

fn main() {
    // Train the agent and persist the best policy for inference.
    let artifacts_config = TrainingArtifactsConfig::new(ARTIFACT_DIR);
    let mut ppo = PPOBuilder::gym(ENV_NAME, 10)
        .with_training_artifacts(artifacts_config)
        .with_policy_hidden_layers(vec![64, 64])
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_training_limit(TrainingLimit::rollouts(30))
        .build()
        .unwrap();
    ppo.train().unwrap();

    // Reload the artifacts later without rebuilding the policy by hand.
    let env = GymEnv::new(ENV_NAME, Some("human".to_owned())).unwrap();
    let mut inference = InferenceRunner::load(ARTIFACT_DIR, env).unwrap();
    for _ in 0..10 {
        inference.run_episode().unwrap();
    }
}
```

The evaluator periodically measures the current policy and persists the best one
as an inference-ready bundle. The bundle contains the policy configuration,
SafeTensors weights, and observation-normalizer state when normalization is
enabled. Run this example from the workspace root with:

```text
cargo run -p r2l-examples --example ppo
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
