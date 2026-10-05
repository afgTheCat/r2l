// super basic stuff, we begin with burn

use std::num::NonZeroUsize;

use r2l_agents::on_policy_algorithms::{
    a2c::A2CParams,
    ppo::{PPO, PPOParams},
};
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
    A2CSettings, BurnBackend, EpisodeBoundHook, OnPolicyTrainingHooks, PPOBurn, PPOCandle,
    PPOSettings, StepBoundHook,
    backend::Backend,
    builders::{
        learner::{AlgorithmConfiguration, LearnerConfig, LearningHookConfig},
        on_policy_hook::OnPolicyHookConfig,
    },
    evaluator::EvaluationSampler,
    hooks::progress::SharedTrainingProgress,
    utils::RewardNormalizer,
};

// #[derive(Debug, Clone, Copy)]
enum AlgoSettings {
    A2C(A2CSettings),
    PPO(PPOSettings),
}

enum AlgoParams {
    A2C(A2CParams),
    PPO(PPOParams),
}

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

pub struct SnapshotAlgo<E: Env> {
    backend: Backend,
    algo_settings: AlgoSettings,
    algo_params: AlgoParams,
    learner: LearnerConfig,
    learning_hook: LearningHookConfig,
    sampler_configuration: SamplerConfiguration<E>,
    hook_config: OnPolicyHookConfig<E>,
}

impl<E: Env> SnapshotAlgo<E> {
    // just prototype things
    fn candle_algo(
        &self,
    ) -> OnPolicyAlgorithm<
        PPOCandle,
        StagedSampler<E, StepBoundHook<E>>,
        OnPolicyTrainingHooks<PPOCandle, StagedSampler<E, StepBoundHook<E>>, E>,
    > {
        todo!()
    }

    // this we might not need as a serparate func, this is just black boxing the exact construction
    fn get_policy_learner_snapshot(&self) -> PolicyValueLearnerSnapshot<BurnBackend> {
        todo!()
    }

    // creates the shared progress things, will be used to build a log of shit
    fn shared_progress(&self) -> SharedTrainingProgress {
        todo!()
    }

    fn burn_algo(
        self,
    ) -> OnPolicyAlgorithm<
        PPOBurn<BurnBackend>,
        StagedSampler<E, StepBoundHook<E>>,
        OnPolicyTrainingHooks<PPOBurn<BurnBackend>, StagedSampler<E, StepBoundHook<E>>, E>,
    > {
        let snapshot = self.get_policy_learner_snapshot();
        let progress = self.shared_progress();
        let Self {
            backend,
            algo_settings,
            learner,
            learning_hook,
            algo_params,
            sampler_configuration,
            hook_config,
        } = self;
        let learner = learner.build_burn_learner().unwrap();
        let learner = learner.load_snapshot(snapshot);
        let AlgoSettings::PPO(ppo_settings) = algo_settings else {
            unreachable!()
        };
        let hooks = learning_hook.burn_ppo_hook(ppo_settings, progress.clone());
        let AlgoParams::PPO(params) = algo_params else {
            unreachable!()
        };
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
        OnPolicyAlgorithm {
            runtime: runtime,
            hooks,
        }
    }
}
