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
    commands::{OnPolicyCommandHandler, OnPolicyControlEndpoint},
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

/// Lifecycle hooks sharing progress with the learning hooks.
pub struct OnPolicyTrainingHooks<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor> + 'static> {
    timing_recorder: TimingRecorder,
    pub(crate) progress: SharedTrainingProgress,
    pub(crate) evaluator: ScheduledEvaluator<E>,
    command_handler: OnPolicyCommandHandler,
    error: Option<Error>,
    _phantom: PhantomData<(A, S)>,
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>>
    OnPolicyTrainingHooks<A, S, E>
{
    /// Creates lifecycle hooks around shared training progress.
    ///
    /// # Arguments
    ///
    /// * `progress` - Progress shared with the learning, evaluation, and timing hooks.
    /// * `evaluator` - Evaluation cadence and best-policy tracking.
    /// * `command_handler` - Handles control requests between training phases.
    /// * `timing_recorder` - Records durations for each completed training iteration.
    pub(crate) fn new(
        progress: SharedTrainingProgress,
        evaluator: ScheduledEvaluator<E>,
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

    /// Attaches external control to the algorithm, replacing any previous endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - Algorithm-side endpoint created by [`OnPolicyControlEndpoint::channel`].
    pub fn set_control(&mut self, endpoint: OnPolicyControlEndpoint) {
        self.command_handler = OnPolicyCommandHandler::new(Some(endpoint));
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
        self.progress.borrow().progress_result()
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
        let evaluation_result = try_or_break!(self, {
            let eval_res = self.evaluator.evaluate(runtime);
            self.timing_recorder.finish_current_phase_recording();
            let timing_result = self.timing_recorder.flush();
            timing_result.and(eval_res)
        });
        let progress_result = self.progress.borrow().progress_result();
        let command_result = try_or_break!(self, self.command_handler.process_pending(runtime));
        self.timing_recorder.record_new_phase(Phase::Rollout);
        evaluation_result.and(progress_result).and(command_result)
    }

    fn finish_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> Result<(), Error> {
        let evaluator_result = self.evaluator.finish_training(runtime);
        let notification_result = self.command_handler.notify_stopped();
        match self.error.take() {
            Some(error) => Err(error),
            None => evaluator_result.and(notification_result),
        }
    }
}

/// Training lifecycle hooks using the standard on-policy sampler.
pub type OnPolicyHooks<A, E> = OnPolicyTrainingHooks<A, crate::OnPolicySampler<E>, E>;
