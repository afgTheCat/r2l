use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use r2l_api::{
    ClippedNormalizer, Env, EnvBuilder, EnvDescription, InferenceRunnerBuilder, NormalizerMode,
    Snapshot, Space, TensorData,
};

struct TerminalEnv {
    reset_count: Arc<AtomicUsize>,
}

impl Env for TerminalEnv {
    type Tensor = TensorData;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor> {
        self.reset_count.fetch_add(1, Ordering::Relaxed);
        Ok(TensorData::from_vec(vec![2.0]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>> {
        Ok(Snapshot::new(
            TensorData::from_vec(vec![99.0]),
            1.0,
            true,
            false,
        ))
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: vec![1],
            },
            Space::Discrete(1),
        )
    }
}

struct TerminalEnvBuilder {
    reset_count: Arc<AtomicUsize>,
}

impl EnvBuilder for TerminalEnvBuilder {
    type Env = TerminalEnv;

    fn build_env(&self) -> Result<Self::Env> {
        Ok(TerminalEnv {
            reset_count: self.reset_count.clone(),
        })
    }
}

#[test]
fn inference_preserves_terminal_observation_until_reset() {
    let reset_count = Arc::new(AtomicUsize::new(0));
    let mut runner = InferenceRunnerBuilder::new(TerminalEnvBuilder {
        reset_count: reset_count.clone(),
    })
    .with_policy_hidden_layers(Vec::new())
    .with_seed(42)
    .build()
    .unwrap();

    assert_eq!(runner.current_observation().data, vec![2.0]);
    let memory = runner.step().unwrap();

    assert!(memory.is_done());
    assert_eq!(memory.state.data, vec![2.0]);
    assert_eq!(memory.next_state.data, vec![99.0]);
    assert_eq!(runner.current_observation().data, vec![99.0]);
    assert!(runner.episode_done());
    assert!(runner.step().is_err());
    assert_eq!(reset_count.load(Ordering::Relaxed), 1);

    assert_eq!(runner.reset().data, vec![2.0]);
    assert!(!runner.episode_done());
    assert_eq!(reset_count.load(Ordering::Relaxed), 2);
}

#[test]
fn inference_uses_the_normalized_pool_when_configured() {
    let normalizer = ClippedNormalizer::new(NormalizerMode::Update, 5.0, vec![1]);
    let mut runner = InferenceRunnerBuilder::new(TerminalEnvBuilder {
        reset_count: Arc::new(AtomicUsize::new(0)),
    })
    .with_policy_hidden_layers(Vec::new())
    .with_obs_normalizer(normalizer)
    .build()
    .unwrap();

    assert_eq!(runner.current_observation().data, vec![5.0]);
    let memory = runner.step().unwrap();

    assert_eq!(memory.state.data, vec![5.0]);
    assert_eq!(memory.next_state.data, vec![5.0]);
    assert_eq!(runner.current_observation().data, vec![5.0]);
}
