use std::{
    fs::File,
    marker::PhantomData,
    time::{Duration, Instant},
};

use r2l_core::{
    HookResult,
    env::Env,
    error::Error,
    models::{Actor, ToSafetensors},
    on_policy::algorithm::{Agent, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrainingLoopPhase {
    Collection,
    Training,
    Evaluation,
}

enum LearningSchedule {
    RolloutBound { total_rollouts: usize },
    TotalStepBound { total_steps: usize },
}

/// Learning-rate policy applied over the progress of an on-policy training run.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule {
    /// Keep the learning rate fixed throughout training.
    Constant(f64),
    /// Decay the initial learning rate linearly to zero.
    Linear(f64),
}

struct BestActorEvaluator<A: Actor, E: Env> {
    _p: PhantomData<(A, E)>,
}

impl<A: Actor, E: Env> BestActorEvaluator<A, E> {
    pub fn eval<AG: Agent<Actor = A>, TS: Sampler<Tensor = E::Tensor>>(
        &mut self,
        rt: &mut OnPolicyRuntime<AG, TS>,
        rollout_idx: usize,
    ) -> Result<(), Error> {
        todo!()
    }
}

struct OnPolicyCommander();

impl OnPolicyCommander {
    fn process_pending_commands<A: Agent, S: Sampler>(
        &self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) -> HookResult {
        todo!()
    }
}

#[derive(Default)]
struct TrainingLoopTimings {
    collect_time: Duration,
    training_time: Duration,
    evaluation_time: Duration,
    total_time: Duration,
}

struct TrainingTimingsRecorder {
    file: File,
    training_start: Instant,
    phase_start: Instant,
    current_timings: TrainingLoopTimings,
}

impl TrainingTimingsRecorder {
    fn init(&mut self) {
        self.phase_start = Instant::now();
        self.training_start = Instant::now();
    }

    fn record_phase_start(&mut self) {
        self.phase_start = Instant::now();
    }

    fn record_phase_end(&mut self, phase: TrainingLoopPhase) {
        let duration = Instant::now() - self.phase_start;
        match phase {
            TrainingLoopPhase::Collection => {
                self.current_timings.collect_time = duration;
            }
            TrainingLoopPhase::Training => {
                self.current_timings.training_time = duration;
            }
            TrainingLoopPhase::Evaluation => {
                self.current_timings.evaluation_time = duration;
                let timings = std::mem::take(&mut self.current_timings);
                // TODO: write this to the file
            }
        }
    }
}

struct TrainingLoopState {
    rollout: usize,
    steps_taken: usize,
}

struct DefaultOnPolicyAlgorithmHooks2<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    pub(crate) training_loop_state: TrainingLoopState,
    pub(crate) learning_schedule: LearningSchedule,
    pub(crate) learning_rate_schedule: LearningRateSchedule,
    pub(crate) best_actor_evaluator: BestActorEvaluator<A::Actor, E>,
    pub(crate) on_policy_commander: OnPolicyCommander,
    pub(crate) training_timings: TrainingTimingsRecorder,
    pub(crate) _phantom: PhantomData<(A, S, E)>,
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>>
    DefaultOnPolicyAlgorithmHooks2<A, S, E>
{
    fn progress_remaining(&self) -> f64 {
        match self.learning_schedule {
            LearningSchedule::RolloutBound { total_rollouts } => {
                1.0 - self.training_loop_state.rollout as f64 / total_rollouts as f64
            }
            LearningSchedule::TotalStepBound { total_steps } => {
                1.0 - self.training_loop_state.steps_taken as f64 / total_steps as f64
            }
        }
    }

    fn leraning_rate(&self) -> f64 {
        match self.learning_rate_schedule {
            LearningRateSchedule::Constant(learning_rate) => learning_rate,
            LearningRateSchedule::Linear(initial_learning_rate) => {
                let progress = self.progress_remaining();
                initial_learning_rate * progress.clamp(0.0, 1.0)
            }
        }
    }

    fn advance_state(
        &mut self,
        runtime: &mut OnPolicyRuntime<A, S>,
        training_loop_phase: TrainingLoopPhase,
    ) -> Result<(), Error> {
        self.training_timings.record_phase_end(training_loop_phase);
        match training_loop_phase {
            TrainingLoopPhase::Collection => {
                let rollouts = runtime.trajectory_containers();
                let rollout_steps: usize = rollouts.as_ref().iter().map(|e| e.actions.len()).sum();
                self.training_loop_state.steps_taken += rollout_steps;
                let learning_rate = self.leraning_rate();
                drop(rollouts);
                runtime.agent.set_learning_rate(learning_rate);
            }
            TrainingLoopPhase::Training => {
                self.training_timings.record_phase_start();
                self.best_actor_evaluator
                    .eval(runtime, self.training_loop_state.rollout)?;
                self.training_timings
                    .record_phase_end(TrainingLoopPhase::Evaluation);
                self.training_loop_state.rollout += 1
            }
            TrainingLoopPhase::Evaluation => unreachable!(),
        }
        Ok(())
    }
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyAlgorithmHooks
    for DefaultOnPolicyAlgorithmHooks2<A, S, E>
{
    type A = A;
    type S = S;

    fn init_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        self.training_timings.init();
        HookResult::Continue
    }

    fn post_rollout_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        self.advance_state(runtime, TrainingLoopPhase::Collection);
        let command_result = self.on_policy_commander.process_pending_commands(runtime);
        self.training_timings.record_phase_start();
        command_result
    }

    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> HookResult {
        self.advance_state(runtime, TrainingLoopPhase::Training);
        let command_result = self.on_policy_commander.process_pending_commands(runtime);
        let hook_result = if self.progress_remaining() <= 0. {
            HookResult::Break
        } else {
            command_result
        };
        self.training_timings.record_phase_start();
        hook_result
    }

    fn shutdown_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> Result<(), Error> {
        todo!()
    }
}
