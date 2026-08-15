use std::path::{Path, PathBuf};

use burn::backend::NdArray;
use candle_core::DType;
use candle_nn::VarBuilder;
use r2l_burn::distributions::BurnPolicyKind;
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    ActorWrapper,
    env::{Env, Snapshot, normalizer::ClippedNormalizer},
    error::{BoxedError, BrokenArtifact, Error},
    models::Actor,
    rng::sample_u64,
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

use super::normalizer::NormalizerBuilder;
use super::{BurnBackendConfig, CandleBackend, PolicyBuilder};

pub(crate) const INFERENCE_CONFIG_FILE: &str = "inference.yaml";
pub(crate) const ACTOR_FILE: &str = "actor.safetensors";
pub(crate) const NORMALIZER_FILE: &str = "normalizer.yaml";

struct ArtifactFile {
    path: PathBuf,
    artifact_type: &'static str,
}

impl ArtifactFile {
    fn new(path: PathBuf, artifact_type: &'static str) -> Self {
        Self {
            path,
            artifact_type,
        }
    }

    fn read(&self) -> Result<Vec<u8>, Error> {
        std::fs::read(&self.path).map_err(|error| self.read_error(error))
    }

    fn read_to_string(&self) -> Result<String, Error> {
        std::fs::read_to_string(&self.path).map_err(|error| self.read_error(error))
    }

    fn read_error(&self, error: std::io::Error) -> Error {
        if error.kind() == std::io::ErrorKind::NotFound {
            BrokenArtifact::Missing {
                path: self.path.clone(),
                artifact_type: self.artifact_type.into(),
            }
            .into()
        } else {
            Error::wrap(error)
        }
    }

    fn decode_error(&self, source: BoxedError) -> Error {
        BrokenArtifact::Decode {
            path: self.path.clone(),
            artifact_type: self.artifact_type.into(),
            source,
        }
        .into()
    }
}

/// Observation processing applied during inference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum InferenceObservationMode {
    /// Uses observations exactly as produced by the environment.
    Raw,
    /// Applies observation normalization using separately stored statistics.
    Normalized,
}

/// Backend used to construct the inference policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum InferenceBackend {
    /// Candle backend configuration.
    Candle(CandleBackend),
    /// Default Burn backend configuration.
    Burn(BurnBackendConfig),
}

/// Serializable recipe for reconstructing an inference runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InferenceConfig {
    policy_builder: PolicyBuilder,
    observation_mode: InferenceObservationMode,
    backend: InferenceBackend,
}

impl InferenceConfig {
    /// Creates an inference configuration.
    pub(crate) fn new(
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
    pub(crate) fn write_to_dir(&self, inference_dir: impl AsRef<Path>) -> Result<(), Error> {
        let inference_dir = inference_dir.as_ref();
        std::fs::create_dir_all(inference_dir).map_err(Error::wrap)?;
        let serialized = yaml_serde::to_string(self).map_err(Error::wrap)?;
        std::fs::write(inference_dir.join(INFERENCE_CONFIG_FILE), serialized)
            .map_err(Error::wrap)?;
        Ok(())
    }

    /// Loads `inference.yaml` from `inference_dir`.
    pub(crate) fn load_from_dir(inference_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let artifact = ArtifactFile::new(
            inference_dir.as_ref().join(INFERENCE_CONFIG_FILE),
            "inference configuration",
        );
        let serialized = artifact.read_to_string()?;
        yaml_serde::from_str(&serialized).map_err(|error| artifact.decode_error(Box::new(error)))
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
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration cannot be read or deserialized.
    pub fn load(directory: impl Into<PathBuf>) -> Result<Self, Error> {
        let directory = directory.into();
        let config = InferenceConfig::load_from_dir(&directory)?;
        Ok(Self { config, directory })
    }

    /// Builds an inference runtime using the configured backend and learned artifacts.
    ///
    /// # Errors
    ///
    /// Returns an error if an artifact cannot be read or the configured model cannot be built.
    pub fn build<E: Env>(self, env: E) -> Result<InferenceRunner<E>, Error> {
        let obs_normalizer = match self.config.observation_mode {
            InferenceObservationMode::Raw => None,
            InferenceObservationMode::Normalized => {
                let artifact = ArtifactFile::new(
                    self.directory.join(NORMALIZER_FILE),
                    "observation normalizer",
                );
                let serialized = artifact.read_to_string()?;
                let normalizer_builder: NormalizerBuilder = yaml_serde::from_str(&serialized)
                    .map_err(|error| artifact.decode_error(Box::new(error)))?;
                Some(normalizer_builder.into_normalizer()?)
            }
        };
        let env_description = env.env_description();
        let actor_artifact = ArtifactFile::new(self.directory.join(ACTOR_FILE), "actor");
        let actor_bytes = actor_artifact.read()?;
        let actor = match self.config.backend {
            InferenceBackend::Candle(backend) => {
                let var_builder =
                    VarBuilder::from_buffered_safetensors(actor_bytes, DType::F32, &backend.device)
                        .map_err(|error| actor_artifact.decode_error(Box::new(error)))?;
                let actor = CandlePolicyKind::build(
                    env_description.action_space.clone(),
                    &var_builder,
                    &self.config.policy_builder.hidden_layers,
                    env_description.observation_space.size(),
                    self.config.policy_builder.activation_function,
                    self.config.policy_builder.log_std_init,
                )
                .map_err(|error| actor_artifact.decode_error(Box::new(error)))?;
                InferenceActor::Candle(ActorWrapper::new(actor))
            }
            InferenceBackend::Burn(_) => {
                let actor = self
                    .config
                    .policy_builder
                    .build_burn::<NdArray, _>(
                        env_description.observation_space.size(),
                        env_description.action_space,
                    )?
                    .load_from_bytes(actor_bytes)
                    .map_err(|error| actor_artifact.decode_error(Box::new(error)))?;
                InferenceActor::Burn(Box::new(ActorWrapper::new(actor)))
            }
        };
        InferenceRunner::new(env, obs_normalizer, actor)
    }
}

/// Backend-independent actor adapted to an environment tensor type.
#[derive(Debug, Clone)]
enum InferenceActor<T: R2lTensor> {
    /// Candle-backed inference actor.
    Candle(ActorWrapper<CandlePolicyKind, T>),
    /// Burn-backed inference actor.
    Burn(Box<ActorWrapper<BurnPolicyKind<NdArray>, T>>),
}

impl<T: R2lTensor> Actor for InferenceActor<T> {
    type Tensor = T;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor, Error> {
        match self {
            Self::Candle(actor) => actor.action(observation),
            Self::Burn(actor) => actor.action(observation),
        }
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor, Error> {
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
    ) -> Result<Self, Error> {
        let mut last_state = env.reset(sample_u64())?;
        if let Some(obs_normalizer) = &obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut last_state)?;
        }
        Ok(Self {
            env,
            obs_normalizer,
            actor,
            last_state,
        })
    }

    /// Resets the environment and its current actor observation.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment cannot be reset.
    pub fn reset(&mut self) -> Result<(), Error> {
        let mut last_state = self.env.reset(sample_u64())?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut last_state)?;
        }
        self.last_state = last_state;
        Ok(())
    }

    /// Selects the modal action and advances the environment by one step.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference or the environment step fails.
    pub fn mode_step(&mut self) -> Result<Snapshot<E::Tensor>, Error> {
        let action = self.actor.mode_action(self.last_state.clone())?;
        let mut snapshot = self.env.step(action)?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut snapshot.state)?;
        }
        self.last_state = snapshot.state.clone();
        Ok(snapshot)
    }

    /// Chooses an action and advances the environment by one step.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference or the environment step fails.
    pub fn step(&mut self) -> Result<Snapshot<E::Tensor>, Error> {
        let action = self.actor.action(self.last_state.clone())?;
        let mut snapshot = self.env.step(action)?;
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut snapshot.state)?;
        }
        self.last_state = snapshot.state.clone();
        Ok(snapshot)
    }

    /// Runs the environment to completion and then resets it.
    ///
    /// # Panics
    ///
    /// Panics if action inference, an environment step, or the final reset fails.
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
    ///
    /// # Panics
    ///
    /// Panics if action inference, an environment step, or the final reset fails.
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
