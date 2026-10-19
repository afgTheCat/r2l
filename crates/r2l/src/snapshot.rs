//! Loading recipes for Burn PPO training snapshots.

use std::{
    fs,
    path::{Path, PathBuf},
};

use r2l_core::{
    env::{Env, EnvBuilderType, normalizer::NormalizerSnapshot},
    error::{Error, Result},
    on_policy::algorithm::OnPolicyAlgorithm,
    running_mean::RunningMeanStdF32,
    tensor::VecTensor,
};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearnerSnapshot;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    OnPolicyBuilder, OnPolicyHooks, PPOBuilder, PPOBurn,
    backend::BurnBackendConfig,
    builders::{AlgoBackendConfig, BuilderConfig, PPOConfig},
    constants::{ACTOR_FILE, NORMALIZER_FILE},
    hooks::{on_policy::evaluation::ScheduledEvaluator, progress::TrainingCounters},
    utils::RewardNormalizer,
};

const BUILDER_FILE: &str = "builder.yaml";
const STATE_FILE: &str = "state.yaml";
const LEARNER_FILE: &str = "learner.bpk";

#[derive(Serialize, Deserialize)]
struct BuilderRecipe<C = BuilderConfig, A = AlgoBackendConfig<PPOConfig, BurnBackendConfig>> {
    version: u32,
    agent: String,
    config: C,
    algo_backend: A,
    n_envs: usize,
}

#[derive(Serialize, Deserialize)]
struct TrainingState {
    progress: TrainingCounters,
    observation_normalizer: Option<NormalizerSnapshot>,
    reward_statistics: Option<RunningMeanStdF32>,
    best_reward: Option<f32>,
}

/// Location of a Burn PPO snapshot's configuration, learner, and training state.
///
/// Save the builder before consuming it, then save state between completed training
/// iterations. Restoration starts fresh environments; environment state, random-number
/// generator positions, and unfinished episode rewards are not restored.
/// Existing evaluation and timing CSVs are appended to. Timing totals are per run.
#[derive(Debug, Clone)]
pub struct SnapshotAlgo {
    directory: PathBuf,
}

impl SnapshotAlgo {
    /// Points to a snapshot directory without reading or creating it.
    ///
    /// # Arguments
    ///
    /// * `directory` - Directory containing `builder.yaml`, `learner.bpk`, and `state.yaml`.
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Saves the construction recipe before the builder is consumed by `build`.
    ///
    /// # Arguments
    ///
    /// * `builder` - Burn PPO builder whose resulting algorithm will be snapshotted.
    ///
    /// # Errors
    /// Returns an error if the directory or configuration cannot be written, or already exists.
    pub fn save_builder<E: Env + 'static>(
        &self,
        builder: &PPOBuilder<E, BurnBackendConfig>,
    ) -> Result<()> {
        fs::create_dir_all(&self.directory)?;
        let recipe = BuilderRecipe {
            version: 1,
            agent: "ppo_burn".to_owned(),
            config: &builder.config,
            algo_backend: &builder.algo_backend,
            n_envs: builder.env_builder.num_envs(),
        };
        self.write_new(BUILDER_FILE, &recipe)
    }

    /// Reconstructs a Burn PPO builder from the saved recipe and environment factories.
    /// Restored artifact configuration appends to the original output directory.
    ///
    /// # Arguments
    ///
    /// * `env_builder` - Environment factories matching the saved spaces and environment count.
    ///
    /// # Errors
    /// Returns an error for unreadable configuration or incompatible environments.
    pub fn get_builder<E: Env + 'static>(
        &self,
        env_builder: EnvBuilderType<E>,
    ) -> Result<PPOBuilder<E, BurnBackendConfig>> {
        let recipe = self.read_builder()?;
        if recipe.n_envs != env_builder.num_envs() {
            return Err(Error::invalid_state(
                "restore builder",
                "environment count differs from the snapshot",
            ));
        }
        let description = env_builder.env_description()?;
        let policy = &recipe.config.learner_builder.policy_config;
        let observation = description.observation_space.convert::<VecTensor>()?;
        let action = description.action_space.convert::<VecTensor>()?;
        if yaml_serde::to_value(&observation).map_err(Error::wrap)?
            != yaml_serde::to_value(&policy.observation_space).map_err(Error::wrap)?
            || yaml_serde::to_value(&action).map_err(Error::wrap)?
                != yaml_serde::to_value(&policy.action_space).map_err(Error::wrap)?
        {
            return Err(Error::invalid_state(
                "restore builder",
                "environment spaces differ from the snapshot",
            ));
        }
        Ok(OnPolicyBuilder {
            algo_backend: recipe.algo_backend,
            config: recipe.config,
            env_builder,
        })
    }

    /// Loads the saved Burn model and optimizer records.
    ///
    /// # Errors
    /// Returns an error if `learner.bpk` is missing or malformed.
    pub fn get_snapshot(&self) -> Result<PolicyValueLearnerSnapshot> {
        PolicyValueLearnerSnapshot::from_file(self.directory.join(LEARNER_FILE))
    }

    /// Saves learner and shared training state after a completed training iteration.
    /// The builder recipe must already have been saved for this algorithm.
    /// Use a separate directory for each checkpoint.
    ///
    /// # Arguments
    ///
    /// * `algo` - Algorithm constructed from the saved builder, paused between iterations.
    ///
    /// # Errors
    /// Returns an error if state cannot be extracted or written, or a checkpoint already exists.
    pub fn save_state<E: Env + 'static>(
        &self,
        algo: &OnPolicyAlgorithm<OnPolicyHooks<PPOBurn, E>>,
    ) -> Result<()> {
        let recipe = self.read_builder()?;
        if self.directory.join(STATE_FILE).exists() {
            return Err(Error::invalid_state(
                "save snapshot",
                "checkpoint already exists; use a new directory",
            ));
        }
        let best_reward = match &algo.hooks.evaluator {
            ScheduledEvaluator::Disabled => None,
            ScheduledEvaluator::Enabled { evaluator, .. } => evaluator.best_reward(),
        };
        let state = TrainingState {
            progress: algo.hooks.progress.borrow().counters(),
            observation_normalizer: algo
                .runtime
                .sampler
                .obs_normalizer()
                .map(|n| n.snapshot())
                .transpose()?,
            reward_statistics: algo
                .runtime
                .sampler
                .hook()
                .reward_normalizer
                .as_ref()
                .map(RewardNormalizer::statistics),
            best_reward,
        };
        Self::validate_state(&recipe, &state)?;
        if best_reward.is_some() {
            let artifacts = recipe
                .config
                .hook_config
                .training_artifacts
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_state(
                        "save snapshot",
                        "best policy has no artifact configuration",
                    )
                })?;
            Self::copy_best_policy(
                &artifacts.output_dir,
                &self.directory,
                state.observation_normalizer.is_some(),
            )?;
        }
        algo.runtime
            .agent
            .lm
            .to_snapshot()
            .to_file(self.directory.join(LEARNER_FILE))?;
        // Publish the state last so incomplete snapshots cannot be restored.
        self.write_new(STATE_FILE, &state)
    }

    /// Builds Burn PPO and restores its learner, shared progress, normalizers, and best reward.
    /// Runtime control and reporting channels can be attached to the returned hooks.
    ///
    /// # Arguments
    ///
    /// * `env_builder` - Environment factories matching the saved spaces and environment count.
    ///
    /// # Errors
    /// Returns an error if snapshot data is missing, incompatible, or cannot be restored.
    pub fn restore<E: Env + 'static>(
        &self,
        env_builder: EnvBuilderType<E>,
    ) -> Result<OnPolicyAlgorithm<OnPolicyHooks<PPOBurn, E>>> {
        let state: TrainingState = self.read_yaml(STATE_FILE)?;
        let recipe = self.read_builder()?;
        Self::validate_state(&recipe, &state)?;
        let snapshot = self.get_snapshot()?;
        let builder = self.get_builder(env_builder)?;
        let mut algo = builder.build()?;
        algo.runtime.agent.lm = algo.runtime.agent.lm.load_snapshot(snapshot)?;
        if let Some(normalizer) = state.observation_normalizer {
            algo.runtime.sampler.restore_obs_normalizer(normalizer)?;
        }
        if let Some(statistics) = state.reward_statistics
            && let Some(normalizer) = &mut algo.runtime.sampler.hook_mut().reward_normalizer
        {
            normalizer.restore_statistics(statistics);
        }
        algo.hooks
            .progress
            .borrow_mut()
            .restore_counters(state.progress);
        if let ScheduledEvaluator::Enabled { evaluator, .. } = &mut algo.hooks.evaluator {
            evaluator.restore_best_reward(state.best_reward);
        }
        if state.best_reward.is_some() {
            let artifacts = recipe
                .config
                .hook_config
                .training_artifacts
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_state(
                        "restore snapshot",
                        "best policy has no artifact configuration",
                    )
                })?;
            Self::copy_best_policy(
                &self.directory,
                &artifacts.output_dir,
                algo.runtime.sampler.obs_normalizer().is_some(),
            )?;
        }
        Ok(algo)
    }

    fn read_builder(&self) -> Result<BuilderRecipe> {
        let recipe: BuilderRecipe = self.read_yaml(BUILDER_FILE)?;
        if recipe.version != 1 || recipe.agent != "ppo_burn" {
            return Err(Error::invalid_state(
                "read snapshot",
                "unsupported snapshot version or agent",
            ));
        }
        Ok(recipe)
    }

    fn validate_state(recipe: &BuilderRecipe, state: &TrainingState) -> Result<()> {
        use crate::{ObsNormalizerConfig, RewardNormalizerConfig};
        let config = &recipe.config.sampler_configuration;
        if state.observation_normalizer.is_some()
            != matches!(config.obs_normalizer, ObsNormalizerConfig::Enabled { .. })
            || state.reward_statistics.is_some()
                != matches!(
                    config.reward_normalizer,
                    RewardNormalizerConfig::Enabled { .. }
                )
        {
            return Err(Error::invalid_state(
                "restore snapshot",
                "normalizer state does not match the builder configuration",
            ));
        }
        if state
            .reward_statistics
            .as_ref()
            .is_some_and(|s| !s.is_valid())
        {
            return Err(Error::invalid_state(
                "restore snapshot",
                "invalid reward statistics",
            ));
        }
        if state.best_reward.is_some()
            && !recipe
                .config
                .hook_config
                .training_artifacts
                .as_ref()
                .is_some_and(|a| a.inference_artifacts)
        {
            return Err(Error::invalid_state(
                "restore snapshot",
                "best-policy state requires inference artifacts",
            ));
        }
        if state.best_reward.is_some_and(|r| !r.is_finite()) {
            return Err(Error::invalid_state(
                "restore snapshot",
                "best reward must be finite",
            ));
        }
        Ok(())
    }

    fn copy_best_policy(source: &Path, target: &Path, normalized: bool) -> Result<()> {
        fs::create_dir_all(target)?;
        if fs::canonicalize(source)? == fs::canonicalize(target)? {
            return Err(Error::invalid_state(
                "snapshot artifacts",
                "snapshot and training artifact directories must differ",
            ));
        }
        fs::copy(source.join(ACTOR_FILE), target.join(ACTOR_FILE))?;
        if normalized {
            fs::copy(source.join(NORMALIZER_FILE), target.join(NORMALIZER_FILE))?;
        }
        Ok(())
    }

    fn read_yaml<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        let serialized = fs::read_to_string(self.directory.join(name))?;
        yaml_serde::from_str(&serialized).map_err(Error::wrap)
    }

    fn write_new(&self, name: &str, value: &impl Serialize) -> Result<()> {
        use std::io::Write;
        let serialized = yaml_serde::to_string(value).map_err(Error::wrap)?;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(name))?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }
}
