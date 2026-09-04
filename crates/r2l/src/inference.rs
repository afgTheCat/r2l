use std::path::{Path, PathBuf};

use burn::backend::NdArray;
use candle_core::DType;
use candle_nn::VarBuilder;
use r2l_burn::distributions::BurnPolicyKind;
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    ActorWrapper,
    env::{Env, normalizer::ClippedNormalizer},
    error::{BoxedError, BrokenArtifact, Error},
    models::Actor,
    rng::sample_u64,
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

use crate::builders::{
    BurnBackendConfig, CandleBackend, normalizer::NormalizerBuilder, policy::PolicyBuilder,
};

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

/// A loaded policy that applies its saved observation preprocessing before inference.
pub struct InferencePolicy<T: R2lTensor> {
    obs_normalizer: Option<ClippedNormalizer<T>>,
    actor: InferenceActor<T>,
}

impl<T: R2lTensor> InferencePolicy<T> {
    /// Loads a policy from learned artifacts.
    ///
    /// # Arguments
    ///
    /// * `directory` - Directory containing the saved inference configuration and model artifacts.
    ///
    /// # Errors
    ///
    /// Returns an error if an artifact cannot be read or decoded or the configured model cannot be
    /// built.
    pub fn load(directory: impl AsRef<Path>) -> Result<Self, Error> {
        let directory = directory.as_ref();
        let config = InferenceConfig::load_from_dir(directory)?;
        let obs_normalizer = match config.observation_mode {
            InferenceObservationMode::Raw => None,
            InferenceObservationMode::Normalized => {
                let artifact =
                    ArtifactFile::new(directory.join(NORMALIZER_FILE), "observation normalizer");
                let serialized = artifact.read_to_string()?;
                let normalizer_builder: NormalizerBuilder = yaml_serde::from_str(&serialized)
                    .map_err(|error| artifact.decode_error(Box::new(error)))?;
                Some(normalizer_builder.into_normalizer()?)
            }
        };
        let actor_artifact = ArtifactFile::new(directory.join(ACTOR_FILE), "actor");
        let actor_bytes = actor_artifact.read()?;
        let actor = match config.backend {
            InferenceBackend::Candle(backend) => {
                let var_builder =
                    VarBuilder::from_buffered_safetensors(actor_bytes, DType::F32, &backend.device)
                        .map_err(|error| actor_artifact.decode_error(Box::new(error)))?;
                let actor = CandlePolicyKind::build(
                    config.policy_builder.action_space.convert::<T>()?,
                    &var_builder,
                    &config.policy_builder.hidden_layers,
                    config.policy_builder.observation_size,
                    config.policy_builder.activation_function,
                    config.policy_builder.log_std_init,
                )
                .map_err(|error| actor_artifact.decode_error(Box::new(error)))?;
                InferenceActor::Candle(ActorWrapper::new(actor))
            }
            InferenceBackend::Burn(_) => {
                let actor = config
                    .policy_builder
                    .build_burn::<NdArray, T>()?
                    .load_from_bytes(actor_bytes)
                    .map_err(|error| actor_artifact.decode_error(Box::new(error)))?;
                InferenceActor::Burn(Box::new(ActorWrapper::new(actor)))
            }
        };
        Ok(Self {
            obs_normalizer,
            actor,
        })
    }

    /// Chooses an action for a raw observation.
    ///
    /// # Arguments
    ///
    /// * `observation` - Raw observation to normalize and pass to the policy.
    ///
    /// # Errors
    ///
    /// Returns an error if observation normalization or action inference fails.
    pub fn action(&self, mut observation: T) -> Result<T, Error> {
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut observation)?;
        }
        self.actor.action(observation)
    }

    /// Chooses the modal action for a raw observation.
    ///
    /// # Arguments
    ///
    /// * `observation` - Raw observation to normalize and pass to the policy.
    ///
    /// # Errors
    ///
    /// Returns an error if observation normalization or action inference fails.
    pub fn mode_action(&self, mut observation: T) -> Result<T, Error> {
        if let Some(obs_normalizer) = &self.obs_normalizer {
            obs_normalizer.apply_tensor_in_place(&mut observation)?;
        }
        self.actor.mode_action(observation)
    }
}

/// External system that accepts policy actions and returns subsequent observations.
pub trait InferenceEnv {
    /// Tensor type used for observations and actions.
    type Tensor: R2lTensor;

    /// Applies an action and returns the subsequent raw observation.
    ///
    /// # Arguments
    ///
    /// * `action` - Action to apply to the external system.
    ///
    /// # Errors
    ///
    /// Returns an error if the action cannot be applied or the observation cannot be obtained.
    fn step(&mut self, action: Self::Tensor) -> Result<Self::Tensor, Error>;
}

impl<E: Env> InferenceEnv for E {
    type Tensor = E::Tensor;

    fn step(&mut self, action: Self::Tensor) -> Result<Self::Tensor, Error> {
        Ok(self.step(action)?.state)
    }
}

/// Stateful, single-environment inference runtime.
pub struct InferenceRunner<E: InferenceEnv> {
    inference_env: E,
    policy: InferencePolicy<E::Tensor>,
    last_observation: E::Tensor,
}

impl<E: InferenceEnv> InferenceRunner<E> {
    /// Loads an inference runtime with an initial raw observation.
    ///
    /// # Arguments
    ///
    /// * `directory` - Directory containing the saved inference configuration and model artifacts.
    /// * `inference_env` - External system in which the loaded policy will run.
    /// * `initial_observation` - Raw observation from which inference starts.
    ///
    /// # Errors
    ///
    /// Returns an error if an artifact cannot be read or decoded or the configured model cannot be
    /// built.
    pub fn load(
        directory: impl AsRef<Path>,
        inference_env: E,
        initial_observation: E::Tensor,
    ) -> Result<Self, Error> {
        let policy = InferencePolicy::load(directory)?;
        Ok(Self {
            inference_env,
            policy,
            last_observation: initial_observation,
        })
    }

    /// Selects the modal action and advances the external system by one step.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference or the environment step fails.
    pub fn mode_step(&mut self) -> Result<E::Tensor, Error> {
        let action = self.policy.mode_action(self.last_observation.clone())?;
        let observation = self.inference_env.step(action)?;
        self.last_observation = observation.clone();
        Ok(observation)
    }

    /// Chooses an action and advances the external system by one step.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference or the environment step fails.
    pub fn step(&mut self) -> Result<E::Tensor, Error> {
        let action = self.policy.action(self.last_observation.clone())?;
        let observation = self.inference_env.step(action)?;
        self.last_observation = observation.clone();
        Ok(observation)
    }
}

impl<E: Env> InferenceRunner<E> {
    /// Loads an inference runtime and resets the environment to obtain its initial observation.
    ///
    /// # Arguments
    ///
    /// * `directory` - Directory containing the saved inference configuration and model artifacts.
    /// * `env` - Environment in which the loaded policy will run.
    ///
    /// # Errors
    ///
    /// Returns an error if an artifact cannot be read or decoded, the configured model cannot be
    /// built, or the environment cannot be reset.
    pub fn load_from_env(directory: impl AsRef<Path>, mut env: E) -> Result<Self, Error> {
        let policy = InferencePolicy::load(directory)?;
        let initial_observation = env.reset(sample_u64())?;
        Ok(Self {
            inference_env: env,
            policy,
            last_observation: initial_observation,
        })
    }

    /// Resets the environment and its current policy observation.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment cannot be reset.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.last_observation = self.inference_env.reset(sample_u64())?;
        Ok(())
    }

    /// Runs the environment to completion and then resets it.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference, an environment step, or the final reset fails.
    pub fn run_episode(&mut self) -> Result<(), Error> {
        loop {
            let action = self.policy.action(self.last_observation.clone())?;
            let snapshot = <E as Env>::step(&mut self.inference_env, action)?;
            self.last_observation = snapshot.state;
            if snapshot.terminated || snapshot.truncated {
                break;
            }
        }
        self.reset()
    }

    /// Runs the environment to completion using modal actions only and then resets it.
    ///
    /// # Errors
    ///
    /// Returns an error if action inference, an environment step, or the final reset fails.
    pub fn mode_run_episode(&mut self) -> Result<(), Error> {
        loop {
            let action = self.policy.mode_action(self.last_observation.clone())?;
            let snapshot = <E as Env>::step(&mut self.inference_env, action)?;
            self.last_observation = snapshot.state;
            if snapshot.terminated || snapshot.truncated {
                break;
            }
        }
        self.reset()
    }
}
