use std::io::Write;
use std::sync::{Arc, Mutex};
use std::{fs::File, marker::PhantomData, time::Instant};

use r2l_core::env::EnvDescription;
use r2l_core::env::normalizer::{
    ClippedNormalizer, ClippedNormalizerInner, ClippedRunningMean, NormalizerMode,
};
use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    on_policy::algorithm::{
        Agent, OnPolicyAlgorithm, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler,
    },
};
use r2l_sampler::{DirectSampler, StagedSampler};

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

trait EnvBuildPlan<A: Agent, E: Env<Tensor = A::Tensor>> {
    fn build_evaluator(
        &self,
        config: TrainingArtifactsConfig,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> BestActorEvaluator<A::Actor, E>;
}

struct TypedEnvBuildPlan<EB: EnvBuilder> {
    env_builder: EnvBuilderType<EB>,
}

impl<A: Agent, EB: EnvBuilder<Env: Env<Tensor = A::Tensor>>> EnvBuildPlan<A, EB::Env>
    for TypedEnvBuildPlan<EB>
{
    fn build_evaluator(
        &self,
        config: TrainingArtifactsConfig,
        obs_normalizer: Option<ClippedNormalizer<<EB::Env as Env>::Tensor>>,
    ) -> BestActorEvaluator<A::Actor, EB::Env> {
        config.build(obs_normalizer, self.env_builder.clone())
    }
}

struct Builder<A: Agent, E: Env<Tensor = A::Tensor>> {
    env_build_plan: Box<dyn EnvBuildPlan<A, E>>,
    env_desription: EnvDescription<E::Tensor>,
    sampler_configuraion: SamplerConfiguration<E>,

    // for the hooks
    learning_schedule: LearningSchedule,
    learning_rate_schedule: Option<LearningRateSchedule>,
    training_artifacts_config: Option<TrainingArtifactsConfig>,
    policy_command_rx: Option<OnPolicyCommandReceiver>,
}

impl<A: Agent, E: Env<Tensor = A::Tensor>> Builder<A, E> {
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

    fn evaluator(
        &mut self,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Option<BestActorEvaluator<A::Actor, E>> {
        let config = self.training_artifacts_config.take()?;
        Some(self.env_build_plan.build_evaluator(config, obs_normalizer))
    }

    fn default_on_policy_hook<S: Sampler<Tensor = E::Tensor>>(
        mut self,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
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
        let evaluator = self.evaluator(obs_normalizer);
        DefaultOnPolicyAlgorithmHooks {
            learning_schedule: self.learning_schedule,
            learning_rate_schedule: self.learning_rate_schedule,
            evaluator,
            performance_log,
            command_rx: self.policy_command_rx.take(),
            _phantom: PhantomData,
        }
    }
}

trait Buildable {
    fn build<A: Agent, E: Env<Tensor = A::Tensor>>(builder: &Builder<A, E>) -> Self;
}

struct Config<A: Agent<Tensor = S::Tensor>, S: Sampler, E: Env<Tensor = S::Tensor>>(
    PhantomData<(A, S, E)>,
);

struct OnPolicyAlgoBuilder<A: Agent<Tensor = S::Tensor>, S: Sampler, E: Env<Tensor = S::Tensor>> {
    builder: Builder<A, E>,
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

// impl<E: Env> Buildable for DirectSampler<E, StepBoundHook<E>> {
//     fn build<A: Agent, E: Env<Tensor = A::Tensor>>(builder: &Builder<A, E>) -> Self {
//         todo!()
//     }
// }

// impl<E: Env> Buildable for DirectSampler<E, EpisodeBoundHook<E>> {
//     fn build<A: Agent, E: Env<Tensor = A::Tensor>>(builder: &Builder<A, E>) -> Self {
//         todo!()
//     }
// }

// impl<E: Env> Buildable for StagedSampler<E, StepBoundHook<E>> {
//     fn build<A: Agent, E: Env<Tensor = A::Tensor>>(builder: &Builder<A, E>) -> Self {
//         todo!()
//     }
// }

impl<A: Agent<Tensor = S::Tensor> + Buildable, S: Sampler + Buildable, E: Env<Tensor = S::Tensor>>
    OnPolicyAlgoBuilder<A, S, E>
{
    fn build(self) -> OnPolicyAlgorithm<A, S, DefaultOnPolicyAlgorithmHooks<A, S, E>> {
        let agent = A::build(&self.builder);
        let sampler = S::build(&self.builder);
        let hooks = self.builder.default_on_policy_hook(None);
        OnPolicyAlgorithm::new(OnPolicyRuntime { agent, sampler }, hooks)
    }
}
