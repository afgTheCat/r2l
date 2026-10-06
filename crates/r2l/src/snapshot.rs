//! Shared snapshot configuration and progress, with backend-specific learner state.

use std::sync::mpsc::Sender;

use r2l_core::{
    env::Env,
    error::Error,
    on_policy::algorithm::{OnPolicyAlgorithm, OnPolicyRuntime},
};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearnerSnapshot;
use r2l_sampler::StagedSampler;

use crate::{
    OnPolicyTrainingHooks, PPOBurn, PPORolloutStats, StepBoundHook,
    backend::{Backend, BurnBackendConfig},
    builders::{
        algorithm::AlgoConfig,
        learner::{LearnerConfig, LearningHookConfig},
        on_policy_hook::OnPolicyHookConfig,
        sampler::SamplerConfiguration,
    },
    hooks::progress::TrainingProgress,
};

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
    hook_config: OnPolicyHookConfig,
}

impl<E: Env> SnapshotAlgo<E, PolicyValueLearnerSnapshot> {
    fn burn_algo(
        self,
        reporter: Option<Sender<PPORolloutStats>>,
    ) -> Result<
        OnPolicyAlgorithm<
            PPOBurn,
            StagedSampler<E, StepBoundHook<E>>,
            OnPolicyTrainingHooks<PPOBurn, StagedSampler<E, StepBoundHook<E>>, E>,
        >,
        Error,
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
        hook_config.validate_evaluation_schedule(&progress.borrow())?;
        let learner = learner_config.build_burn_learner()?;
        let learner = learner.load_snapshot(learner_state);
        let agent = algo_config.build_ppo(
            learner,
            &learning_hook,
            &learner_config.optimizer,
            progress.clone(),
            reporter,
            sampler_configuration.env_build_plan.n_envs(),
        );
        let sampler = sampler_configuration.staged_sampler_step_bound(progress.clone())?;
        let hooks = hook_config.build(
            &sampler_configuration,
            &learner_config.policy_config,
            &Backend::Burn(BurnBackendConfig),
            progress,
        )?;
        let runtime = OnPolicyRuntime { agent, sampler };
        Ok(OnPolicyAlgorithm::new(runtime, hooks))
    }
}
