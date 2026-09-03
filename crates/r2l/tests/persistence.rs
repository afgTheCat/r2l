use std::path::Path;

use r2l::{
    Env, EnvDescription, EvaluationSettings, InferenceEnv, InferencePolicy, InferenceRunner,
    PPOBuilder, SamplerExecutionMode, Snapshot, Space, TrainingArtifactsConfig, TrainingLimit,
    VecTensor,
};
use r2l_core::{error::Error, tensor::R2lTensor};
use tempfile::TempDir;

#[derive(Default)]
struct TinyEnv;

impl Env for TinyEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor, Error> {
        Ok(VecTensor::from_vec(vec![1.0, 0.0]))
    }

    fn step(&mut self, action: Self::Tensor) -> Result<Snapshot<Self::Tensor>, Error> {
        let action = action.to_vec()?[0];
        Ok(Snapshot::new(
            VecTensor::from_vec(vec![0.0, 1.0]),
            action,
            true,
            false,
        ))
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: vec![2],
            },
            Space::Discrete(2),
        )
    }
}

struct TinyInferenceEnv;

impl InferenceEnv for TinyInferenceEnv {
    type Tensor = VecTensor;

    fn step(&mut self, _action: Self::Tensor) -> Result<Self::Tensor, Error> {
        Ok(VecTensor::from_vec(vec![0.0, 1.0]))
    }
}

fn artifact_config(path: &Path) -> TrainingArtifactsConfig {
    TrainingArtifactsConfig::new(path)
        .with_evaluation_results(false)
        .with_training_timings(false)
        .with_evaluation_settings(
            EvaluationSettings::new()
                .with_episodes_per_evaluation(1)
                .with_execution_mode(SamplerExecutionMode::SingleThreaded),
        )
}

fn assert_artifacts_exist(path: &Path, normalized: bool) {
    assert!(path.join("inference.yaml").is_file());
    assert!(path.join("actor.safetensors").is_file());
    assert_eq!(path.join("normalizer.yaml").is_file(), normalized);
}

#[allow(clippy::float_cmp)] // Exact equality is the behavior under test.
fn assert_repeated_loads_are_equivalent(path: &Path) {
    let first = InferencePolicy::load(path).unwrap();
    let second = InferencePolicy::load(path).unwrap();
    let observation = VecTensor::from_vec(vec![1.0, 0.0]);
    let first_action = first.mode_action(observation.clone()).unwrap();
    let second_action = second.mode_action(observation.clone()).unwrap();
    assert_eq!(
        first_action.to_vec().unwrap(),
        second_action.to_vec().unwrap()
    );

    let mut runner = InferenceRunner::load(path, TinyInferenceEnv, observation).unwrap();
    assert_eq!(
        runner.mode_step().unwrap().to_vec().unwrap(),
        vec![0.0, 1.0]
    );

    let mut env_runner = InferenceRunner::load_from_env(path, TinyEnv).unwrap();
    env_runner.mode_run_episode().unwrap();
}

#[test]
fn candle_policy_round_trips_through_inference_artifacts() {
    let output = TempDir::new().unwrap();
    let mut algorithm = PPOBuilder::new(|| Ok(TinyEnv), 1)
        .unwrap()
        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
        .with_rollout_steps(4)
        .with_training_limit(TrainingLimit::rollouts(1))
        .with_policy_hidden_layers(vec![4])
        .with_sample_size(4)
        .with_total_epochs(1)
        .with_seed(7)
        .with_training_artifacts(artifact_config(output.path()))
        .build()
        .unwrap();
    algorithm.train().unwrap();

    assert_artifacts_exist(output.path(), false);
    assert_repeated_loads_are_equivalent(output.path());
}

#[test]
fn burn_policy_round_trips_through_inference_artifacts() {
    let output = TempDir::new().unwrap();
    let mut algorithm = PPOBuilder::new(|| Ok(TinyEnv), 1)
        .unwrap()
        .with_burn()
        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
        .with_rollout_steps(4)
        .with_training_limit(TrainingLimit::rollouts(1))
        .with_policy_hidden_layers(vec![4])
        .with_sample_size(4)
        .with_total_epochs(1)
        .with_seed(7)
        .with_training_artifacts(artifact_config(output.path()))
        .build()
        .unwrap();
    algorithm.train().unwrap();

    assert_artifacts_exist(output.path(), false);
    assert_repeated_loads_are_equivalent(output.path());
}

#[test]
fn observation_normalizer_round_trips_with_policy() {
    let output = TempDir::new().unwrap();
    let mut algorithm = PPOBuilder::new(|| Ok(TinyEnv), 1)
        .unwrap()
        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
        .with_rollout_steps(4)
        .with_observation_normalizer(Some(10.0))
        .unwrap()
        .with_training_limit(TrainingLimit::rollouts(1))
        .with_policy_hidden_layers(vec![4])
        .with_sample_size(4)
        .with_total_epochs(1)
        .with_seed(7)
        .with_training_artifacts(artifact_config(output.path()))
        .build()
        .unwrap();
    algorithm.train().unwrap();

    assert_artifacts_exist(output.path(), true);
    assert_repeated_loads_are_equivalent(output.path());

    std::fs::remove_file(output.path().join("normalizer.yaml")).unwrap();
    let error = InferenceRunner::load_from_env(output.path(), TinyEnv)
        .err()
        .expect("missing observation normalizer should fail");
    assert!(error.to_string().contains("missing observation normalizer"));
}

#[test]
fn missing_and_corrupt_artifacts_report_contextual_errors() {
    let missing = TempDir::new().unwrap();
    let error = InferenceRunner::load_from_env(missing.path(), TinyEnv)
        .err()
        .expect("missing inference configuration should fail");
    assert!(
        error
            .to_string()
            .contains("missing inference configuration")
    );

    std::fs::write(missing.path().join("inference.yaml"), "not: [valid").unwrap();
    let error = InferenceRunner::load_from_env(missing.path(), TinyEnv)
        .err()
        .expect("corrupt inference configuration should fail");
    assert!(
        error
            .to_string()
            .contains("failed to decode inference configuration")
    );
}

#[test]
fn missing_actor_is_reported_after_valid_configuration_is_loaded() {
    let output = TempDir::new().unwrap();
    let mut algorithm = PPOBuilder::new(|| Ok(TinyEnv), 1)
        .unwrap()
        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
        .with_rollout_steps(2)
        .with_training_limit(TrainingLimit::rollouts(1))
        .with_policy_hidden_layers(vec![2])
        .with_sample_size(2)
        .with_total_epochs(1)
        .with_training_artifacts(artifact_config(output.path()))
        .build()
        .unwrap();
    algorithm.train().unwrap();
    std::fs::remove_file(output.path().join("actor.safetensors")).unwrap();

    let error = InferenceRunner::load_from_env(output.path(), TinyEnv)
        .err()
        .expect("missing actor should fail");
    assert!(error.to_string().contains("missing actor"));

    std::fs::write(output.path().join("actor.safetensors"), b"not safetensors").unwrap();
    let error = InferenceRunner::load_from_env(output.path(), TinyEnv)
        .err()
        .expect("corrupt actor should fail");
    assert!(error.to_string().contains("failed to decode actor"));
}
