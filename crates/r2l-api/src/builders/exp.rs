use std::io::Write;
use std::sync::Arc;
use std::{fs::File, marker::PhantomData, time::Instant};

use r2l_core::env::normalizer::{ClippedNormalizer, NormalizerMode};
use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType},
    on_policy::algorithm::{
        Agent, OnPolicyAlgorithm, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler,
    },
};

use crate::{
    BestActorEvaluator, DefaultOnPolicyAlgorithmHooks, LearningRateSchedule, LearningSchedule,
    OnPolicyCommandReceiver, TrainingArtifactsConfig, hooks::on_policy::PerformanceLog,
};

const PERFORMANCE_FILE: &str = "performance.csv";

enum SamplerConfiguration {
    Direct,
    Staged { obs_clip: Option<f32> },
}

struct Builder<E: Env> {
    env_builder: Arc<dyn EnvBuilder<Env = E>>,
    n_envs: usize,
    sampler_configuraion: SamplerConfiguration,

    // for the hooks
    learning_schedule: LearningSchedule,
    learning_rate_schedule: Option<LearningRateSchedule>,
    training_artifacts_config: Option<TrainingArtifactsConfig>,
    policy_command_rx: Option<OnPolicyCommandReceiver>,
}

impl<E: Env> Builder<E> {
    // fn obs_normalizer(&self) -> Option<ClippedNormalizer<E::Tensor>> {
    //     let SamplerConfiguration::Staged { obs_clip } = self.sampler_configuraion else {
    //         return None;
    //     };
    //     let env_description = self.env_builder.env_description().unwrap();
    //     let obs_size = env_description.observation_space.size();
    //     todo!()
    // }

    fn evaluator<A: Agent>(
        &self,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> Option<BestActorEvaluator<A::Actor, E>> {
        todo!()
        // let Some(config) = self.training_artifacts_config.as_ref() else {
        //     return None;
        // };
        // let env_builder = self.env_builder.clone();
        // let env_builder = move || env_builder.build_env();
        // let env_builder = EnvBuilderType::homogeneous(env_builder, self.n_envs);
        // let evaluator = config.build(obs_normalizer, env_builder);
        // Some(evaluator)
    }

    fn default_on_policy_hook<A: Agent, S: Sampler<Tensor = E::Tensor>>(
        mut self,
        obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    ) -> DefaultOnPolicyAlgorithmHooks<A, S, E> {
        let evaluator = self.evaluator::<A>(obs_normalizer);
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
    fn build<E: Env>(builder: &Builder<E>) -> Self;
}

struct Config<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>>(PhantomData<(A, S, E)>);

struct OnPolicyAlgoBuilder<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    builder: Builder<E>,
    config: Config<A, S, E>,
}

impl<A: Agent + Buildable, S: Sampler + Buildable, E: Env<Tensor = S::Tensor>>
    OnPolicyAlgoBuilder<A, S, E>
{
    fn build(mut self) -> OnPolicyAlgorithm<A, S, DefaultOnPolicyAlgorithmHooks<A, S, E>> {
        let agent = A::build(&self.builder);
        let sampler = S::build(&self.builder);
        // TODO: add that thing in here
        let hooks = self.builder.default_on_policy_hook(None);
        OnPolicyAlgorithm::new(OnPolicyRuntime { agent, sampler }, hooks)
    }
}
