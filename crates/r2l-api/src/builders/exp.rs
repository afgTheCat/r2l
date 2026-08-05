use std::io::Write;
use std::sync::Arc;
use std::{fs::File, marker::PhantomData, time::Instant};

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

enum SamplerType {
    Direct,
    Staged,
}

struct Builder<E: Env> {
    env_builder: Arc<dyn EnvBuilder<Env = E>>,

    // for the hooks
    learning_schedule: LearningSchedule,
    learning_rate_schedule: Option<LearningRateSchedule>,
    training_artifacts_config: Option<TrainingArtifactsConfig>,
    policy_command_rx: Option<OnPolicyCommandReceiver>,
}

impl<E: Env> Builder<E> {
    fn evaluator<A: Agent>(&self) -> Option<BestActorEvaluator<A::Actor, E>> {
        let Some(config) = self.training_artifacts_config else {
            return None;
        };
        // let env_builder = EnvBuilderType::homogeneous(builder, n_envs)
        // let evaluator = (config.evaluation_results || config.inference_artifacts)
        //     .then(|| config.build::<A::Actor, EB>(obs_normalizer, env_builder));

        todo!()
    }

    fn default_on_policy_hook<A: Agent, S: Sampler<Tensor = E::Tensor>>(
        mut self,
    ) -> DefaultOnPolicyAlgorithmHooks<A, S, E> {
        let evaluator = self.evaluator::<A>();
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
        let hooks = self.builder.default_on_policy_hook();
        OnPolicyAlgorithm::new(OnPolicyRuntime { agent, sampler }, hooks)
    }
}
