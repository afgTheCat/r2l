use std::path::{Path, PathBuf};

use anyhow::{bail, ensure};
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    ActorWrapper,
    env::{Env, Snapshot, normalizer::ClippedNormalizer},
    models::Actor,
    rng::sample_u64,
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

    /// Builds a Candle inference runtime for `env`.
    pub fn build_candle<E: Env>(
        self,
        env: E,
        normalizer_builder: Option<NormalizerBuilder>,
    ) -> anyhow::Result<CandleInferenceRunner<E>> {
        let backend = match self.backend {
            InferenceBackend::Candle(backend) => backend,
            InferenceBackend::Burn(_) => bail!("inference configuration uses the Burn backend"),
        };
        let obs_normalizer = match self.observation_mode {
            InferenceObservationMode::Raw => {
                ensure!(
                    normalizer_builder.is_none(),
                    "raw inference does not use an observation normalizer"
                );
                None
            }
            InferenceObservationMode::Normalized => {
                let normalizer_builder = normalizer_builder
                    .ok_or_else(|| anyhow::anyhow!("normalized inference requires a normalizer"))?;
                Some(normalizer_builder.into_normalizer())
            }
        };
        let env_description = env.env_description();
        let actor = self.policy_builder.build_candle(
            env_description.observation_space.size(),
            env_description.action_space,
            &backend.device,
        )?;
        InferenceRunner::new(env, obs_normalizer, ActorWrapper::new(actor))
    }

    /// Loads learned artifacts and builds a Candle inference runtime for `env`.
    pub fn build_candle_from_dir<E: Env>(
        self,
        env: E,
        inference_dir: impl AsRef<Path>,
    ) -> anyhow::Result<CandleInferenceRunner<E>> {
        let inference_dir = inference_dir.as_ref();
        let backend = match self.backend {
            InferenceBackend::Candle(backend) => backend,
            InferenceBackend::Burn(_) => bail!("inference configuration uses the Burn backend"),
        };
        let obs_normalizer = match self.observation_mode {
            InferenceObservationMode::Raw => None,
            InferenceObservationMode::Normalized => {
                let serialized = std::fs::read_to_string(inference_dir.join(NORMALIZER_FILE))?;
                let normalizer_builder: NormalizerBuilder = yaml_serde::from_str(&serialized)?;
                Some(normalizer_builder.into_normalizer())
            }
        };
        let actor_bytes = std::fs::read(inference_dir.join(ACTOR_FILE))?;
        let actor = CandlePolicyKind::from_bytes(&actor_bytes, backend.device);
        InferenceRunner::new(env, obs_normalizer, ActorWrapper::new(actor))
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

    /// Builds a Candle inference runtime using the bound learned artifacts.
    pub fn build_candle<E: Env>(self, env: E) -> anyhow::Result<CandleInferenceRunner<E>> {
        self.config.build_candle_from_dir(env, self.directory)
    }
}

/// Stateful, single-environment inference runtime.
pub struct InferenceRunner<E: Env, A: Actor<Tensor = E::Tensor>> {
    env: E,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    actor: A,
    last_state: E::Tensor,
}

/// Candle-backed inference runtime.
pub type CandleInferenceRunner<E> =
    InferenceRunner<E, ActorWrapper<CandlePolicyKind, <E as Env>::Tensor>>;

impl<E: Env, A: Actor<Tensor = E::Tensor>> InferenceRunner<E, A> {
    fn new(
        mut env: E,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
        actor: A,
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
}
