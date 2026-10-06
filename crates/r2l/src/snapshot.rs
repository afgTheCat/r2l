//! Shared snapshot configuration and progress, with backend-specific learner state.

use std::{num::NonZeroUsize, sync::mpsc::Sender};

use r2l_agents::on_policy_algorithms::ppo::PPO;
use r2l_core::{
    env::{
        Env, EnvDescription,
        normalizer::{Normalizer, NormalizerMode},
    },
    error::Error,
    on_policy::algorithm::{OnPolicyAlgorithm, OnPolicyRuntime},
};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearnerSnapshot;
use r2l_sampler::{
    DirectSampler, DirectSamplerCore, SamplerExecutionMode, StagedSampler, StagedSamplerCore,
};

use crate::{
    EpisodeBoundHook, OnPolicyTrainingHooks, PPOBurn, PPORolloutStats, StepBoundHook,
    builders::{
        algorithm::AlgoConfig,
        learner::{LearnerConfig, LearningHookConfig},
        on_policy_hook::OnPolicyHookConfig,
    },
    evaluator::EvaluationSampler,
    hooks::progress::{SharedTrainingProgress, TrainingProgress},
    utils::RewardNormalizer,
};

enum SamplerSetup<E: Env> {
    DirectStep {
        rollout_steps: NonZeroUsize,
        reward_normalizer: Option<RewardNormalizer>,
    },
    DirectEpisode {
        rollout_episodes: NonZeroUsize,
    },
    StagedStep {
        rollout_steps: NonZeroUsize,
        reward_normalizer: Option<RewardNormalizer>,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    },
}

trait EnvBuildPlan<E: Env>: Send {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: NonZeroUsize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<EvaluationSampler<E>, Error>;

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<E>;

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<Normalizer<E::Tensor>>,
    ) -> Result<StagedSamplerCore<E>, Error>;

    fn env_description(&self) -> EnvDescription<E::Tensor>;

    fn n_envs(&self) -> NonZeroUsize;
}

struct SamplerConfiguration<E: Env> {
    setup: SamplerSetup<E>,
    execution_mode: SamplerExecutionMode,
    env_build_plan: Box<dyn EnvBuildPlan<E>>,
}

impl<E: Env> SamplerConfiguration<E> {
    #[allow(clippy::unnecessary_wraps)]
    fn direct_sampler_step_bound(
        &self,
        progress: SharedTrainingProgress,
    ) -> Result<DirectSampler<E, StepBoundHook<E>>, Error> {
        let SamplerSetup::DirectStep {
            reward_normalizer, ..
        } = &self.setup
        else {
            unreachable!("direct step-bound sampler type must use matching configuration")
        };
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.execution_mode);
        let step_bound_hook = StepBoundHook::new(progress, reward_normalizer.clone());
        Ok(DirectSampler::new(sampler_core, step_bound_hook))
    }

    #[allow(clippy::unnecessary_wraps)]
    fn direct_sampler_episode_bound(
        &self,
        progress: SharedTrainingProgress,
    ) -> Result<DirectSampler<E, EpisodeBoundHook<E>>, Error> {
        let SamplerSetup::DirectEpisode { .. } = &self.setup else {
            unreachable!("direct episode-bound sampler type must use matching configuration")
        };
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.execution_mode);
        let episode_bound_hook = EpisodeBoundHook::new(progress, None);
        Ok(DirectSampler::new(sampler_core, episode_bound_hook))
    }

    fn staged_sampler_step_bound(
        &self,
        progress: SharedTrainingProgress,
    ) -> Result<StagedSampler<E, StepBoundHook<E>>, Error> {
        let SamplerSetup::StagedStep {
            reward_normalizer,
            obs_normalizer,
            ..
        } = &self.setup
        else {
            unreachable!("staged step-bound sampler type must use matching configuration")
        };
        let obs_normalizer = obs_normalizer
            .as_ref()
            .map(|normalizer| normalizer.with_mode(NormalizerMode::Update));
        let sampler_core = self
            .env_build_plan
            .build_staged_sampler_core(self.execution_mode, obs_normalizer)?;
        let step_bound_hook = StepBoundHook::new(progress, reward_normalizer.clone());
        Ok(StagedSampler::new(sampler_core, step_bound_hook))
    }
}

/// Configuration and owned progress shared by backend-specific restoration paths.
///
/// `L` carries the backend's model and optimizer snapshot, independently of the
/// concrete policy type. Snapshot capture and additional restoration paths are
/// still under construction.
pub struct SnapshotAlgo<E: Env, L> {
    algo_config: AlgoConfig,
    learner: LearnerConfig,
    learner_state: L,
    progress: TrainingProgress,
    learning_hook: LearningHookConfig,
    sampler_configuration: SamplerConfiguration<E>,
    hook_config: OnPolicyHookConfig<E>,
}

impl<E: Env> SnapshotAlgo<E, PolicyValueLearnerSnapshot> {
    fn burn_algo(
        self,
        reporter: Option<Sender<PPORolloutStats>>,
    ) -> OnPolicyAlgorithm<
        PPOBurn,
        StagedSampler<E, StepBoundHook<E>>,
        OnPolicyTrainingHooks<PPOBurn, StagedSampler<E, StepBoundHook<E>>, E>,
    > {
        let Self {
            learner: learner_config,
            learner_state,
            progress,
            learning_hook,
            sampler_configuration,
            hook_config,
            algo_config,
        } = self;
        let progress = progress.into_shared();
        let learner = learner_config.build_burn_learner().unwrap();
        let learner = learner.load_snapshot(learner_state);
        let (params, settings) =
            algo_config.ppo_parts(reporter, sampler_configuration.env_build_plan.n_envs());
        let hooks = learning_hook.build(settings, &learner_config.optimizer, progress.clone());
        let agent = PPO {
            params,
            lm: learner,
            hooks,
        };
        let sampler = sampler_configuration
            .staged_sampler_step_bound(progress.clone())
            .unwrap();
        let obs_normalizer = sampler.obs_normalizer();
        let hooks = hook_config.build(obs_normalizer, progress);
        let runtime = OnPolicyRuntime { agent, sampler };
        OnPolicyAlgorithm { runtime, hooks }
    }
}
