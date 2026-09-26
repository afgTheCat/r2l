pub(crate) mod commands;
pub(crate) mod evaluation;
pub(crate) mod timing;

use std::marker::PhantomData;

use r2l_core::{
    HookResult,
    env::Env,
    error::Error,
    models::ToSafetensors,
    on_policy::algorithm::{Agent, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler},
};

use self::{
    commands::OnPolicyCommandHandler,
    evaluation::ScheduledEvaluator,
    timing::{Phase, TimingRecorder},
};
use super::progress::SharedTrainingProgress;

macro_rules! try_or_break {
    ($hooks:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => return $hooks.break_with_error(error),
        }
    };
}

/// Lifecycle hooks sharing progress with the sampler and learning hooks.
pub struct OnPolicyTrainingHooks<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    timing_recorder: TimingRecorder,
    progress: SharedTrainingProgress,
    evaluator: ScheduledEvaluator<A::Actor, E>,
    command_handler: OnPolicyCommandHandler,
    error: Option<Error>,
    _phantom: PhantomData<(A, S, E)>,
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>>
    OnPolicyTrainingHooks<A, S, E>
{
    /// Creates lifecycle hooks around shared training progress.
    ///
    /// # Arguments
    ///
    /// * `progress` - Progress shared with the agent and sampler hooks.
    /// * `evaluator` - Evaluation cadence and best-policy tracking.
    /// * `command_handler` - Handles control requests between training phases.
    /// * `timing_recorder` - Records durations for each completed training iteration.
    pub(crate) fn new(
        progress: SharedTrainingProgress,
        evaluator: ScheduledEvaluator<A::Actor, E>,
        command_handler: OnPolicyCommandHandler,
        timing_recorder: TimingRecorder,
    ) -> Self {
        Self {
            progress,
            evaluator,
            command_handler,
            timing_recorder,
            error: None,
            _phantom: PhantomData,
        }
    }

    fn break_with_error(&mut self, error: Error) -> HookResult {
        self.error.get_or_insert(error);
        HookResult::Break
    }
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>> OnPolicyAlgorithmHooks
    for OnPolicyTrainingHooks<A, S, E>
{
    type A = A;
    type S = S;

    fn init_hook(&mut self, _runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        self.timing_recorder.init();
        HookResult::Continue
    }

    fn post_rollout_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        self.timing_recorder.finish_current_phase_recording();
        self.progress.borrow_mut().update_rollout_progress(runtime);
        let command_result = try_or_break!(self, self.command_handler.process_pending(runtime));
        self.timing_recorder.record_new_phase(Phase::Training);
        command_result
    }

    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> HookResult {
        self.timing_recorder.finish_current_phase_recording();
        self.timing_recorder.record_new_phase(Phase::Evaluation);
        let evaluation_result = self.evaluator.evaluate(runtime);
        self.timing_recorder.finish_current_phase_recording();
        let timing_result = self.timing_recorder.flush();
        try_or_break!(self, evaluation_result.and(timing_result));
        let progress_result = self.progress.borrow().progress_result();
        let command_result = try_or_break!(self, self.command_handler.process_pending(runtime));
        self.timing_recorder.record_new_phase(Phase::Rollout);
        progress_result.and(command_result)
    }

    fn finish_training_hook(
        &mut self,
        _runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> Result<(), Error> {
        let evaluator_result = self.evaluator.finish_training();
        let notification_result = self.command_handler.notify_stopped();
        match self.error.take() {
            Some(error) => Err(error),
            None => evaluator_result.and(notification_result),
        }
    }
}
