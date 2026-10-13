//! Shared snapshot configuration and progress, with backend-specific learner state.

use r2l_core::{
    env::Env,
    error::Error,
    on_policy::algorithm::{OnPolicyAlgorithm, OnPolicyRuntime},
};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearnerSnapshot;
use r2l_sampler::RolloutSampler;

use crate::{
    OnPolicyTrainingHooks, PPOBurn, SamplerHook,
    builders::{
        BuilderResources,
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
pub struct SnapshotAlgo<L> {
    algo_config: AlgoConfig,
    learner: LearnerConfig,
    learner_state: L,
    progress: TrainingProgress,
    learning_hook: LearningHookConfig,
    sampler_configuration: SamplerConfiguration,
    hook_config: OnPolicyHookConfig,
}

impl SnapshotAlgo<PolicyValueLearnerSnapshot> {
    fn burn_algo<E: Env + 'static>(
        self,
        mut resources: BuilderResources<E>,
    ) -> Result<
        OnPolicyAlgorithm<OnPolicyTrainingHooks<PPOBurn, RolloutSampler<SamplerHook<E>>, E>>,
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
        let learner = learner_config.build_burn_learner()?;
        let learner = learner.load_snapshot(learner_state);
        let agent = algo_config.build_ppo(
            learner,
            &learning_hook,
            &learner_config.optimizer,
            progress.clone(),
            resources.ppo_reporter.take(),
            resources.n_envs(),
        );
        let obs_normalizer = sampler_configuration.build_obs_normalizer(
            learner_config
                .policy_config
                .observation_space
                .observation_shape(),
        );
        let sampler =
            sampler_configuration.build_sampler(&resources.env_builder, obs_normalizer.clone())?;
        let hooks = hook_config.build(
            resources,
            &learner_config.policy_config,
            obs_normalizer,
            progress,
        )?;
        let runtime = OnPolicyRuntime::new(agent, sampler);
        Ok(OnPolicyAlgorithm::new(runtime, hooks))
    }
}
