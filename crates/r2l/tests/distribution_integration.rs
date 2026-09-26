use std::{collections::BTreeMap, path::Path};

use r2l::{
    A2CBuilder, ActivationFunction, AdamWConfig, Env, EnvDescription, EvaluationSettings,
    GradientClippingConfig, InferencePolicy, LearningRateSchedule, OnPolicyAlgorithm,
    OptimizerConfig, PPOBuilder, SamplerExecutionMode, Snapshot, Space, TrainingArtifactsConfig,
    TrainingLimit, VecTensor,
    builders::networks::{CnnConfig, CnnLayerConfig, MlpConfig, NetworkConfig},
};
use r2l_core::{
    error::Result,
    models::{Actor, Policy, ToSafetensors},
    on_policy::algorithm::{Agent, OnPolicyAlgorithmHooks, Sampler},
    tensor::R2lTensor,
};
use safetensors::SafeTensors;
use tempfile::TempDir;

#[derive(Default)]
struct CompositeEnv;

fn observation() -> VecTensor {
    VecTensor::new(vec![0.25, -0.5, 0.75, 1.0], [1, 2, 2]).unwrap()
}

impl Env for CompositeEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<VecTensor> {
        Ok(observation())
    }

    fn step(&mut self, action: VecTensor) -> Result<Snapshot<VecTensor>> {
        assert_eq!(action.to_shape().dims(), [5]);
        let action = action.to_vec()?;
        assert!(action.iter().all(|value| value.is_finite()));
        assert!([0.0, 1.0].contains(&action[0]));
        assert!([0.0, 1.0].contains(&action[2]));
        assert!([0.0, 1.0].contains(&action[3]));
        assert!([0.0, 1.0, 2.0].contains(&action[4]));
        Ok(Snapshot::new(
            observation(),
            1.0 + action[1] * 0.1,
            true,
            false,
        ))
    }

    fn env_description(&self) -> EnvDescription<VecTensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: [1, 2, 2].into(),
            },
            Space::Dict(BTreeMap::from([
                ("a_discrete".into(), Space::Discrete(2)),
                (
                    "b_nested".into(),
                    Space::Tuple(vec![
                        Space::Box {
                            min: None,
                            max: None,
                            shape: [1].into(),
                        },
                        Space::MultiBinary { shape: [1].into() },
                        Space::MultiDiscrete {
                            nvec: VecTensor::from_vec(vec![2.0, 3.0]),
                            shape: [2].into(),
                        },
                    ]),
                ),
            ])),
        )
    }
}

fn optimizer() -> AdamWConfig {
    AdamWConfig {
        learning_rate: LearningRateSchedule::Constant(0.01),
        gradient_clipping: GradientClippingConfig::Norm(0.5),
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    }
}

fn artifacts(path: &Path) -> TrainingArtifactsConfig {
    TrainingArtifactsConfig::new(path)
        .with_evaluation_results(false)
        .with_training_timings(false)
        .with_evaluation_settings(
            EvaluationSettings::new()
                .with_episodes_per_evaluation(1)
                .with_execution_mode(SamplerExecutionMode::SingleThreaded),
        )
}

fn tensors(bytes: &[u8]) -> BTreeMap<String, (Vec<usize>, Vec<u8>)> {
    SafeTensors::deserialize(bytes)
        .unwrap()
        .tensors()
        .into_iter()
        .map(|(name, tensor)| (name, (tensor.shape().to_vec(), tensor.data().to_vec())))
        .collect()
}

fn train_and_reload<A, S, H>(mut algorithm: OnPolicyAlgorithm<A, S, H>, directory: &Path)
where
    A: Agent<Actor: Policy + ToSafetensors>,
    S: Sampler,
    H: OnPolicyAlgorithmHooks<A = A, S = S>,
{
    let before = tensors(&algorithm.runtime.actor().to_safetensors().unwrap());
    algorithm.train().unwrap();
    let actor = algorithm.runtime.actor();
    let after = tensors(&actor.to_safetensors().unwrap());
    assert_ne!(before, after, "training must update actor parameters");
    assert_eq!(
        after,
        tensors(&std::fs::read(directory.join("actor.safetensors")).unwrap())
    );
    let obs = observation();
    let expected = actor
        .mode_action(A::Tensor::from_vec_and_shape(obs.to_vec().unwrap(), [1, 4]).unwrap())
        .unwrap();
    let loaded = InferencePolicy::load(directory).unwrap();
    let actual = loaded.mode_action(obs).unwrap();
    assert_eq!(actual.to_shape().dims(), [5]);
    for (actual, expected) in actual
        .to_vec()
        .unwrap()
        .into_iter()
        .zip(expected.to_vec().unwrap())
    {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn ppo_and_a2c_train_and_reload_all_distributions_with_joint_and_split_optimizers() {
    for split in [false, true] {
        for burn in [false, true] {
            macro_rules! exercise {
                ($builder:expr) => {{
                    let output = TempDir::new().unwrap();
                    let builder = $builder
                        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
                        .with_rollout_steps(3)
                        .with_training_limit(TrainingLimit::rollouts(1))
                        .with_policy_hidden_layers(vec![4])
                        .with_value_hidden_layers(vec![4])
                        .with_sample_size(4)
                        .with_entropy_coefficient(0.01)
                        .with_value_loss_coefficient(0.5)
                        .with_log_progress(false)
                        .with_training_artifacts(artifacts(output.path()));
                    let config = if split {
                        OptimizerConfig::Split {
                            policy: optimizer(),
                            value: optimizer(),
                        }
                    } else {
                        OptimizerConfig::Joint(optimizer())
                    };
                    let builder = builder.with_optimizer(config);
                    if burn {
                        train_and_reload(builder.with_burn().build().unwrap(), output.path());
                    } else {
                        train_and_reload(builder.build().unwrap(), output.path());
                    }
                }};
            }
            exercise!(
                PPOBuilder::new(|| Ok(CompositeEnv), 2)
                    .unwrap()
                    .with_total_epochs(2)
            );
            exercise!(A2CBuilder::new(|| Ok(CompositeEnv), 2).unwrap());
        }
    }
}

#[test]
fn burn_cnn_policy_and_value_networks_train_and_reload() {
    let output = TempDir::new().unwrap();
    let cnn = NetworkConfig::Cnn(CnnConfig {
        layers: vec![
            CnnLayerConfig::Conv2d {
                out_channels: 2,
                kernel_size: [1, 1],
                stride: [1, 1],
            },
            CnnLayerConfig::Activation(ActivationFunction::Tanh),
            CnnLayerConfig::MaxPool2d {
                kernel_size: [1, 1],
                stride: [1, 1],
            },
            CnnLayerConfig::AvgPool2d {
                kernel_size: [2, 2],
                stride: [2, 2],
            },
        ],
        mlp: MlpConfig {
            hidden_layers: vec![4],
            activation: ActivationFunction::Tanh,
        },
    });
    let algorithm = PPOBuilder::new(|| Ok(CompositeEnv), 1)
        .unwrap()
        .with_burn()
        .with_policy_network(cnn.clone())
        .with_value_network(cnn)
        .with_execution_mode(SamplerExecutionMode::SingleThreaded)
        .with_rollout_steps(3)
        .with_training_limit(TrainingLimit::rollouts(1))
        .with_sample_size(2)
        .with_total_epochs(1)
        .with_learning_rate(0.01)
        .with_log_progress(false)
        .with_training_artifacts(artifacts(output.path()))
        .build()
        .unwrap();
    train_and_reload(algorithm, output.path());
}
