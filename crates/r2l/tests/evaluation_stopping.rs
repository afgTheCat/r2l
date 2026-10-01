use std::{path::Path, sync::mpsc};

use r2l::{
    A2CBuilder, Env, EnvDescription, EvaluationSettings, InferencePolicy, ObsNormalizerConfig,
    PPOBuilder, SamplerExecutionMode, Snapshot, Space, TrainingArtifactsConfig, TrainingLimit,
    VecTensor,
};
use r2l_core::error::Result;
use tempfile::TempDir;

struct ConstantRewardEnv;

impl Env for ConstantRewardEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<VecTensor> {
        Ok(VecTensor::from_vec(vec![0.0]))
    }

    fn step(&mut self, _action: VecTensor) -> Result<Snapshot<VecTensor>> {
        Ok(Snapshot::new(
            VecTensor::from_vec(vec![0.0]),
            1.0,
            true,
            false,
        ))
    }

    fn env_description(&self) -> EnvDescription<VecTensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: [1].into(),
            },
            Space::Discrete(2),
        )
    }
}

fn ppo_builder() -> PPOBuilder<ConstantRewardEnv> {
    PPOBuilder::new(|| Ok(ConstantRewardEnv), 1)
        .unwrap()
        .with_total_epochs(1)
}

fn disabled_artifacts(path: &Path, interval: usize) -> TrainingArtifactsConfig {
    TrainingArtifactsConfig::new(path)
        .with_evaluation_results(false)
        .with_inference_artifacts(false)
        .with_training_timings(false)
        .with_evaluation_settings(
            EvaluationSettings::new()
                .with_rollouts_per_evaluation(interval)
                .with_episodes_per_evaluation(1)
                .with_execution_mode(SamplerExecutionMode::SingleThreaded),
        )
}

macro_rules! train_rollouts {
    ($builder:expr) => {{
        let (tx, rx) = mpsc::channel();
        let mut algorithm = $builder
            .with_execution_mode(SamplerExecutionMode::SingleThreaded)
            .with_rollout_steps(2)
            .with_training_limit(TrainingLimit::rollouts(4))
            .with_policy_hidden_layers(vec![2])
            .with_value_hidden_layers(vec![2])
            .with_sample_size(2)
            .with_log_progress(false)
            .with_rollout_reporter(Some(tx))
            .build()
            .unwrap();
        algorithm.train().unwrap();
        rx.try_iter().count()
    }};
}

#[test]
fn reward_threshold_stops_ppo_and_a2c_on_both_backends_without_artifacts() {
    macro_rules! exercise {
        ($builder:expr) => {
            for (threshold, expected_rollouts) in [
                (Some(-1.0), 1),
                (Some(0.5), 1),
                (Some(1.0), 1),
                (Some(2.0), 4),
                (None, 4),
            ] {
                assert_eq!(
                    train_rollouts!($builder.with_avg_reward_threshold(threshold)),
                    expected_rollouts
                );
            }
        };
    }

    exercise!(ppo_builder());
    exercise!(ppo_builder().with_burn());
    exercise!(A2CBuilder::new(|| Ok(ConstantRewardEnv), 1).unwrap());
    exercise!(
        A2CBuilder::new(|| Ok(ConstantRewardEnv), 1)
            .unwrap()
            .with_burn()
    );

    assert_eq!(
        train_rollouts!(
            ppo_builder()
                .with_avg_reward_threshold(Some(1.0))
                .with_avg_reward_threshold(None)
        ),
        4
    );
}

#[test]
fn stopping_respects_evaluation_interval_with_all_artifacts_disabled() {
    let output = TempDir::new().unwrap();
    let folder = output.path().join("unused");
    assert_eq!(
        train_rollouts!(
            ppo_builder()
                .with_avg_reward_threshold(Some(1.0))
                .with_training_artifacts(disabled_artifacts(&folder, 2))
        ),
        2
    );
    assert!(!folder.exists());
}

#[test]
fn stopping_saves_the_normalized_policy_before_finishing() {
    let output = TempDir::new().unwrap();
    let artifacts = TrainingArtifactsConfig::new(output.path())
        .with_training_timings(false)
        .with_evaluation_settings(
            EvaluationSettings::new()
                .with_episodes_per_evaluation(1)
                .with_execution_mode(SamplerExecutionMode::SingleThreaded),
        );
    assert_eq!(
        train_rollouts!(
            ppo_builder()
                .with_avg_reward_threshold(Some(1.0))
                .with_observation_normalizer(ObsNormalizerConfig::Enabled { clip: Some(10.0) })
                .with_burn()
                .with_training_artifacts(artifacts)
        ),
        1
    );
    assert_eq!(
        std::fs::read_to_string(output.path().join("evaluations.csv")).unwrap(),
        "average_reward,total_episodes\n1,1\n"
    );
    InferencePolicy::<VecTensor>::load(output.path()).unwrap();
}

#[test]
fn stopping_validates_evaluation_schedule_when_artifacts_are_disabled() {
    let output = TempDir::new().unwrap();
    let error = ppo_builder()
        .with_training_limit(TrainingLimit::rollouts(4))
        .with_avg_reward_threshold(Some(1.0))
        .with_training_artifacts(disabled_artifacts(output.path(), 5))
        .build()
        .err()
        .expect("an unreachable evaluation interval should fail");
    assert!(
        error
            .to_string()
            .contains("exceeds the configured training length")
    );
}
