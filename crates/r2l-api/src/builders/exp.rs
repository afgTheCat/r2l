use std::io::Write;
use std::{fs::File, marker::PhantomData, time::Instant};

use r2l_core::env::EnvDescription;
use r2l_core::env::normalizer::{ClippedNormalizer, ClippedNormalizerInner, NormalizerMode};
use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    on_policy::algorithm::{Agent, OnPolicyAlgorithm, OnPolicyRuntime, Sampler},
};
use r2l_sampler::{
    DirectSampler, DirectSamplerCore, SamplerExecutionMode, StagedSampler, StagedSamplerCore,
};

use crate::evaluators::best_actor_evaluator::EvaluationSampler;
use crate::utils::RewardNormalizer;
use crate::{
    BestActorEvaluator, DefaultOnPolicyAlgorithmHooks, LearningRateSchedule, LearningSchedule,
    OnPolicyCommandReceiver, TrainingArtifactsConfig, hooks::on_policy::PerformanceLog,
};
use crate::{EpisodeBoundHook, StepBoundHook};

const PERFORMANCE_FILE: &str = "performance.csv";

enum SamplerConfiguration<E: Env> {
    Direct,
    Staged {
        clipped_normalizer_inner: Option<ClippedNormalizerInner<E::Tensor>>,
    },
}

trait EnvBuildPlan<E: Env> {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: usize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> EvaluationSampler<E>;

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<E>;

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> StagedSamplerCore<E>;
}

struct TypedEnvBuildPlan<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
}

impl<EB: EnvBuilder<Env: Env>> EnvBuildPlan<EB::Env> for TypedEnvBuildPlan<EB> {
    fn build_evaluator_sampler(
        &self,
        episodes_per_evaluation: usize,
        evaluation_execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
    ) -> EvaluationSampler<EB::Env> {
        EvaluationSampler::build(
            self.env_builder.clone(),
            episodes_per_evaluation,
            evaluation_execution_mode,
            obs_normalizer,
        )
    }

    fn build_direct_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
    ) -> DirectSamplerCore<EB::Env> {
        DirectSamplerCore::build(self.env_builder.clone(), execution_mode)
    }

    fn build_staged_sampler_core(
        &self,
        execution_mode: SamplerExecutionMode,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
    ) -> StagedSamplerCore<EB::Env> {
        StagedSamplerCore::build(self.env_builder.clone(), execution_mode, obs_normalizer)
    }
}

struct Builder<E: Env> {
    env_build_plan: Box<dyn EnvBuildPlan<E>>,
    env_desription: EnvDescription<E::Tensor>,
    sampler_configuraion: SamplerConfiguration<E>,

    // for the hooks
    learning_schedule: LearningSchedule,
    learning_rate_schedule: Option<LearningRateSchedule>,
    training_artifacts_config: Option<TrainingArtifactsConfig>,
    policy_command_rx: Option<OnPolicyCommandReceiver>,

    // for the sampler
    sampler_execution_mode: SamplerExecutionMode,
    reward_normalizer: Option<RewardNormalizer>,
    rollout_steps: usize,
    rollout_episodes: usize,
}

impl<E: Env> Builder<E> {
    fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, n_envs: usize) -> Self {
        let env_desription = env_builder.env_description().unwrap();
        Self {
            env_build_plan: Box::new(TypedEnvBuildPlan {
                env_builder: EnvBuilderType::homogeneous(env_builder, n_envs),
            }),
            env_desription,
            sampler_configuraion: SamplerConfiguration::Direct,
            learning_schedule: LearningSchedule::rollout_bound(300),
            learning_rate_schedule: None,
            training_artifacts_config: None,
            policy_command_rx: None,
            sampler_execution_mode: SamplerExecutionMode::MultiThreaded,
            reward_normalizer: None,
            rollout_steps: 1024,
            rollout_episodes: 1,
        }
    }

    fn obs_normalizer(
        &self,
        normalizer_mode: NormalizerMode,
    ) -> Option<ClippedNormalizer<E::Tensor>> {
        let SamplerConfiguration::Staged {
            clipped_normalizer_inner: Some(inner),
        } = &self.sampler_configuraion
        else {
            return None;
        };
        // TODO: this can error, but it's fine for now! In fact, catching this through the test
        // suite would be nice!
        let shape = self.env_desription.observation_space.shape().unwrap();
        let normalizer = ClippedNormalizer {
            normalizer_mode,
            inner: inner.clone(),
        };
        Some(normalizer)
    }

    fn evaluator<A: Agent>(
        &mut self,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Option<BestActorEvaluator<A::Actor, E>> {
        let config = self.training_artifacts_config.take()?;
        if !config.evaluation_results && !config.inference_artifacts {
            return None;
        }
        let evaluation_sampler = self.env_build_plan.build_evaluator_sampler(
            config.evaluation_settings.episodes_per_evaluation,
            config.evaluation_settings.evaluation_execution_mode,
            obs_normalizer,
        );
        Some(config.build_with_sampler(evaluation_sampler))
    }

    fn default_on_policy_hook<A: Agent, S: Sampler<Tensor = E::Tensor>>(
        mut self,
    ) -> DefaultOnPolicyAlgorithmHooks<A, S, E> {
        let performance_log = self.training_artifacts_config.as_ref().map(|config| -> _ {
            let output_dir = config.output_dir.clone();
            std::fs::create_dir_all(&output_dir).unwrap();
            let mut file = File::create(output_dir.join(PERFORMANCE_FILE)).unwrap();
            writeln!(
                file,
                "rollout,collect_ms,learn_ms,evaluate_ms,rollout_ms,total_ms"
            )
            .unwrap();
            let now = Instant::now();
            PerformanceLog {
                file,
                training_started: now,
                rollout_started: now,
                phase_started: now,
                collect_ms: 0.0,
                rollout: 0,
            }
        });
        let obs_normalizer = self.obs_normalizer(NormalizerMode::ReadOnly);
        let evaluator = self.evaluator::<A>(obs_normalizer);
        DefaultOnPolicyAlgorithmHooks {
            learning_schedule: self.learning_schedule,
            learning_rate_schedule: self.learning_rate_schedule,
            evaluator,
            performance_log,
            command_rx: self.policy_command_rx.take(),
            _phantom: PhantomData,
        }
    }

    fn direct_sampler_step_bound(&self) -> DirectSampler<E, StepBoundHook<E>> {
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.sampler_execution_mode);
        let reward_normalizer = self.reward_normalizer.clone();
        let step_bound_hook = StepBoundHook::new(self.rollout_steps, reward_normalizer);
        DirectSampler::new(sampler_core, step_bound_hook)
    }

    fn direct_sampler_episode_bound(&self) -> DirectSampler<E, EpisodeBoundHook<E>> {
        let sampler_core = self
            .env_build_plan
            .build_direct_sampler_core(self.sampler_execution_mode);
        let episode_bound_hook = EpisodeBoundHook::new(self.rollout_episodes);
        DirectSampler::new(sampler_core, episode_bound_hook)
    }

    fn staged_sampler_step_bound(&self) -> StagedSampler<E, StepBoundHook<E>> {
        let obs_normalizer = self.obs_normalizer(NormalizerMode::Update);
        let sampler_core = self
            .env_build_plan
            .build_staged_sampler_core(self.sampler_execution_mode, obs_normalizer);
        let reward_normalizer = self.reward_normalizer.clone();
        let step_bound_hook = StepBoundHook::new(self.rollout_steps, reward_normalizer);
        StagedSampler {
            core: sampler_core,
            hook: step_bound_hook,
        }
    }
}

trait Buildable<E: Env> {
    fn build(builder: &Builder<E>) -> Self;
}

struct Config<A: Agent<Tensor = S::Tensor>, S: Sampler, E: Env<Tensor = S::Tensor>>(
    PhantomData<(A, S, E)>,
);

struct OnPolicyAlgoBuilder<A: Agent<Tensor = S::Tensor>, S: Sampler, E: Env<Tensor = S::Tensor>> {
    builder: Builder<E>,
    config: Config<A, S, E>,
}

impl<A: Agent<Tensor = S::Tensor>, S: Sampler, E: Env<Tensor = S::Tensor>>
    OnPolicyAlgoBuilder<A, S, E>
{
    fn new<EB: EnvBuilder<Env = E>>(env_builder: EB, n_envs: usize) -> Self {
        Self {
            builder: Builder::new(env_builder, n_envs),
            config: Config(PhantomData),
        }
    }
}

// Agent: PPOCandle, A2CCandle, PPOBurn, A2CBurn
// Sampler: DirectSampler<E, StepBound>, DirectSampler<EpisodeBound>,

impl<E: Env> Buildable<E> for DirectSampler<E, StepBoundHook<E>> {
    fn build(builder: &Builder<E>) -> Self {
        builder.direct_sampler_step_bound()
    }
}

impl<E: Env> Buildable<E> for DirectSampler<E, EpisodeBoundHook<E>> {
    fn build(builder: &Builder<E>) -> Self {
        builder.direct_sampler_episode_bound()
    }
}

impl<E: Env> Buildable<E> for StagedSampler<E, StepBoundHook<E>> {
    fn build(builder: &Builder<E>) -> Self {
        builder.staged_sampler_step_bound()
    }
}

impl<
    A: Agent<Tensor = S::Tensor> + Buildable<E>,
    S: Sampler + Buildable<E>,
    E: Env<Tensor = S::Tensor>,
> OnPolicyAlgoBuilder<A, S, E>
{
    fn build(self) -> OnPolicyAlgorithm<A, S, DefaultOnPolicyAlgorithmHooks<A, S, E>> {
        let agent = A::build(&self.builder);
        let sampler = S::build(&self.builder);
        let hooks = self.builder.default_on_policy_hook();
        OnPolicyAlgorithm::new(OnPolicyRuntime { agent, sampler }, hooks)
    }
}
