use std::path::{Path, PathBuf};

use burn::backend::NdArray;
use candle_core::DType;
use candle_nn::VarBuilder;
use r2l_burn::distributions::BurnPolicyKind;
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    ActorWrapper,
    env::{Env, Snapshot, normalizer::ClippedNormalizer},
    models::Actor,
    rng::sample_u64,
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

use crate::{
    BurnBackendConfig, CandleBackend, PolicyBuilder, builders::normalizer::NormalizerBuilder,
};

pub(crate) const INFERENCE_CONFIG_FILE: &str = "inference.yaml";
pub(crate) const ACTOR_FILE: &str = "actor.safetensors";
pub(crate) const NORMALIZER_FILE: &str = "normalizer.yaml";

/// Observation processing applied during inference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceObservationMode {
    /// Uses observations exactly as produced by the environment.
    Raw,
    /// Applies observation normalization using separately stored statistics.
    Normalized,
}

/// Backend used to construct the inference policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "config")]
pub enum InferenceBackend {
    /// Candle backend configuration.
    Candle(CandleBackend),
    /// Default Burn backend configuration.
    Burn(BurnBackendConfig),
}

/// Serializable recipe for reconstructing an inference runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    policy_builder: PolicyBuilder,
    observation_mode: InferenceObservationMode,
    backend: InferenceBackend,
}

impl InferenceConfig {
    /// Creates an inference configuration.
    pub fn new(
        policy_builder: PolicyBuilder,
        observation_mode: InferenceObservationMode,
        backend: InferenceBackend,
    ) -> Self {
        Self {
            policy_builder,
            observation_mode,
            backend,
        }
    }

    /// Writes this configuration to `inference.yaml` in `inference_dir`.
    pub fn write_to_dir(&self, inference_dir: impl AsRef<Path>) -> anyhow::Result<()> {
        let inference_dir = inference_dir.as_ref();
        std::fs::create_dir_all(inference_dir)?;
        let serialized = yaml_serde::to_string(self)?;
        std::fs::write(inference_dir.join(INFERENCE_CONFIG_FILE), serialized)?;
        Ok(())
    }

    /// Loads `inference.yaml` from `inference_dir`.
    pub fn load_from_dir(inference_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let serialized =
            std::fs::read_to_string(inference_dir.as_ref().join(INFERENCE_CONFIG_FILE))?;
        Ok(yaml_serde::from_str(&serialized)?)
    }
}

/// An inference configuration bound to its learned artifact directory.
#[derive(Debug, Clone)]
pub struct InferenceArtifacts {
    config: InferenceConfig,
    directory: PathBuf,
}

impl InferenceArtifacts {
    /// Loads an inference configuration and binds it to `directory`.
    pub fn load(directory: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let directory = directory.into();
        let config = InferenceConfig::load_from_dir(&directory)?;
        Ok(Self { config, directory })
    }

    /// Returns the loaded inference configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Builds an inference runtime using the configured backend and learned artifacts.
    pub fn build<E: Env>(self, env: E) -> anyhow::Result<InferenceRunner<E>> {
        let obs_normalizer = match self.config.observation_mode {
            InferenceObservationMode::Raw => None,
            InferenceObservationMode::Normalized => {
                let serialized = std::fs::read_to_string(self.directory.join(NORMALIZER_FILE))?;
                let normalizer_builder: NormalizerBuilder = yaml_serde::from_str(&serialized)?;
                Some(normalizer_builder.into_normalizer())
            }
        };
        let env_description = env.env_description();
        let actor_bytes = std::fs::read(self.directory.join(ACTOR_FILE))?;
        let actor = match self.config.backend {
            InferenceBackend::Candle(backend) => {
                let var_builder = VarBuilder::from_buffered_safetensors(
                    actor_bytes,
                    DType::F32,
                    &backend.device,
                )?;
                let actor = CandlePolicyKind::build(
                    env_description.action_space.clone(),
                    &var_builder,
                    &self.config.policy_builder.hidden_layers,
                    env_description.observation_space.size(),
                    self.config.policy_builder.activation_function,
                    self.config.policy_builder.log_std_init,
                )?;
                InferenceActor::Candle(ActorWrapper::new(actor))
            }
            InferenceBackend::Burn(_) => {
                let actor = self
                    .config
                    .policy_builder
                    .build_burn::<NdArray, _>(
                        env_description.observation_space.size(),
                        env_description.action_space,
                    )
                    .load_from_bytes(actor_bytes)?;
                InferenceActor::Burn(ActorWrapper::new(actor))
            }
        };
        InferenceRunner::new(env, obs_normalizer, actor)
    }
}

/// Backend-independent actor adapted to an environment tensor type.
#[derive(Debug, Clone)]
pub enum InferenceActor<T: R2lTensor> {
    /// Candle-backed inference actor.
    Candle(ActorWrapper<CandlePolicyKind, T>),
    /// Burn-backed inference actor.
    Burn(ActorWrapper<BurnPolicyKind<NdArray>, T>),
}

impl<T: R2lTensor> Actor for InferenceActor<T> {
    type Tensor = T;

    fn action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Candle(actor) => actor.action(observation),
            Self::Burn(actor) => actor.action(observation),
        }
    }

    fn mode_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Candle(actor) => actor.mode_action(observation),
            Self::Burn(actor) => actor.mode_action(observation),
        }
    }
}

/// Stateful, single-environment inference runtime.
pub struct InferenceRunner<E: Env> {
    env: E,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    actor: InferenceActor<E::Tensor>,
    last_state: E::Tensor,
}

impl<E: Env> InferenceRunner<E> {
    fn new(
        mut env: E,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
        actor: InferenceActor<E::Tensor>,
    ) -> anyhow::Result<Self> {
        let mut last_state = env.reset(sample_u64())?;
        if let Some(obs_normalizer) = &obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut last_state);
        }
        Ok(Self {
            env,
            obs_normalizer,
            actor,
            last_state,
        })
    }

    /// Resets the environment and its current actor observation.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        let mut last_state = self.env.reset(sample_u64())?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut last_state);
        }
        self.last_state = last_state;
        Ok(())
    }

    /// Selects the modal action and advances the environment by one step.
    pub fn mode_step(&mut self) -> anyhow::Result<Snapshot<E::Tensor>> {
        let action = self.actor.mode_action(self.last_state.clone())?;
        let mut snapshot = self.env.step(action)?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut snapshot.state);
        }
        self.last_state = snapshot.state.clone();
        Ok(snapshot)
    }

    /// Chooses an action and advances the environment by one step.
    pub fn step(&mut self) -> anyhow::Result<Snapshot<E::Tensor>> {
        let action = self.actor.action(self.last_state.clone())?;
        let mut snapshot = self.env.step(action)?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut snapshot.state);
        }
        self.last_state = snapshot.state.clone();
        Ok(snapshot)
    }

    /// Runs the environment to completion and then resets it.
    pub fn run_episode(&mut self) {
        loop {
            let snapshot = self.step().unwrap();
            if snapshot.terminated || snapshot.truncated {
                break;
            }
        }
        self.reset().unwrap();
    }

    /// Runs the environment to completion using modal actions only and then resets it.
    pub fn mode_run_episode(&mut self) {
        loop {
            let snapshot = self.mode_step().unwrap();
            if snapshot.terminated || snapshot.truncated {
                break;
            }
        }
        self.reset().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use candle_core::{Device, Tensor};
    use r2l_core::{
        env::{Env, EnvDescription, Snapshot, Space},
        models::Actor,
    };

    use super::*;

    struct TestEnv;

    impl Env for TestEnv {
        type Tensor = Tensor;

        fn reset(&mut self, _seed: u64) -> anyhow::Result<Self::Tensor> {
            Ok(Tensor::zeros(3, candle_core::DType::F32, &Device::Cpu)?)
        }

        fn step(&mut self, _action: Self::Tensor) -> anyhow::Result<Snapshot<Self::Tensor>> {
            Ok(Snapshot::new(
                Tensor::zeros(3, candle_core::DType::F32, &Device::Cpu)?,
                0.0,
                true,
                false,
            ))
        }

        fn env_description(&self) -> EnvDescription<Self::Tensor> {
            EnvDescription::new(
                Space::Box {
                    min: None,
                    max: None,
                    shape: vec![3],
                },
                Space::Box {
                    min: None,
                    max: None,
                    shape: vec![2],
                },
            )
        }
    }

    #[test]
    fn candle_inference_builds_from_saved_safetensors() -> anyhow::Result<()> {
        let output_dir = unique_test_dir("candle-inference-safetensors");
        std::fs::create_dir_all(&output_dir)?;

        let policy_builder = PolicyBuilder::new().with_hidden_layers(vec![4]);
        let actor = policy_builder.build_candle::<Tensor>(
            3,
            Space::Box {
                min: None,
                max: None,
                shape: vec![2],
            },
            &Device::Cpu,
        )?;
        let actor_bytes = actor.try_serialize().unwrap();
        std::fs::write(output_dir.join(ACTOR_FILE), actor_bytes)?;

        let config = InferenceConfig::new(
            policy_builder,
            InferenceObservationMode::Raw,
            InferenceBackend::Candle(CandleBackend {
                device: Device::Cpu,
            }),
        );
        config.write_to_dir(&output_dir)?;

        let artifacts = InferenceArtifacts::load(&output_dir)?;
        let mut runner = artifacts.build(TestEnv)?;
        runner.mode_step()?;

        std::fs::remove_dir_all(output_dir)?;
        Ok(())
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "r2l-api-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }
}
