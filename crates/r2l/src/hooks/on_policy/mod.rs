pub(crate) mod coordinator;
pub(crate) mod learning;

use std::{
    fs::File,
    io::Write,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, Sender, channel},
    time::{Duration, Instant},
};

use r2l_core::{
    HookResult,
    env::Env,
    error::{Error, ResourceInterrupted},
    models::{Actor, ToSafetensors},
    on_policy::algorithm::{Agent, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler},
    tensor::R2lTensor,
};

use self::coordinator::SharedCoordinator;
use crate::{constants::TRAINING_TIMINGS_FILE, evaluator::BestPolicyEvaluator};

#[derive(Debug, Clone, Copy)]
enum Phase {
    Rollout,
    Training,
    Evaluation,
}

struct RecordedPhase {
    phase: Phase,
    started: Instant,
}

impl RecordedPhase {
    fn new(phase: Phase) -> Self {
        Self {
            phase,
            started: Instant::now(),
        }
    }
}

#[derive(Default)]
struct TrainingLoopTimings {
    collection: Duration,
    training: Duration,
    evaluation: Duration,
}

pub(crate) struct EnabledTimingRecorder {
    file: File,
    coordinator: SharedCoordinator,
    timings: TrainingLoopTimings,
    phase_recorder: Option<RecordedPhase>,
    training_started: Instant,
}

impl EnabledTimingRecorder {
    fn finish_current_phase_recording(&mut self) {
        let Some(RecordedPhase { phase, started }) = self.phase_recorder.take() else {
            return;
        };
        let elapsed = started.elapsed();
        match phase {
            Phase::Rollout => self.timings.collection = elapsed,
            Phase::Training => self.timings.training = elapsed,
            Phase::Evaluation => self.timings.evaluation = elapsed,
        }
    }

    fn record_new_phase(&mut self, phase: Phase) {
        self.phase_recorder = Some(RecordedPhase::new(phase));
    }

    fn flush(&mut self) -> Result<(), Error> {
        let completed_rollouts = self.coordinator.borrow().completed_rollouts();
        let timings = std::mem::take(&mut self.timings);
        writeln!(
            self.file,
            "{},{:.3},{:.3},{:.3},{:.3}",
            completed_rollouts,
            timings.collection.as_secs_f64() * 1000.0,
            timings.training.as_secs_f64() * 1000.0,
            timings.evaluation.as_secs_f64() * 1000.0,
            self.training_started.elapsed().as_secs_f64() * 1000.0,
        )
        .map_err(Error::wrap)
    }
}

pub(crate) enum TimingRecorder {
    Disabled,
    Enabled(EnabledTimingRecorder),
}

impl TimingRecorder {
    pub(crate) fn disabled() -> Self {
        Self::Disabled
    }

    /// Creates a CSV recorder for training phase durations.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory in which the timings CSV is created.
    /// * `coordinator` - Training progress used to label each timings row.
    pub(crate) fn create(output_dir: &Path, coordinator: SharedCoordinator) -> Result<Self, Error> {
        std::fs::create_dir_all(output_dir).map_err(Error::wrap)?;
        let mut file = File::create(output_dir.join(TRAINING_TIMINGS_FILE)).map_err(Error::wrap)?;
        writeln!(file, "rollout,collect_ms,learn_ms,evaluate_ms,total_ms").map_err(Error::wrap)?;
        Ok(Self::Enabled(EnabledTimingRecorder {
            file,
            coordinator,
            timings: TrainingLoopTimings::default(),
            phase_recorder: None,
            training_started: Instant::now(),
        }))
    }

    fn init(&mut self) {
        if let Self::Enabled(recorder) = self {
            recorder.training_started = Instant::now();
            recorder.timings = TrainingLoopTimings::default();
            recorder.phase_recorder = None;
        }
        self.record_new_phase(Phase::Rollout);
    }

    fn record_new_phase(&mut self, phase: Phase) {
        match self {
            Self::Disabled => {}
            Self::Enabled(enabled) => enabled.record_new_phase(phase),
        }
    }

    fn finish_current_phase_recording(&mut self) {
        match self {
            Self::Disabled => {}
            Self::Enabled(enabled) => enabled.finish_current_phase_recording(),
        }
    }

    fn flush(&mut self) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Enabled(enabled) => enabled.flush(),
        }
    }
}

/// Learning-rate policy applied to shared collection progress.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule {
    /// Keep the learning rate fixed throughout training.
    Constant(f64),
    /// Decay the initial learning rate to zero, including the current collection in progress.
    /// The final learning pass uses zero learning rate, including a one-rollout run.
    Linear(f64),
}

impl LearningRateSchedule {
    /// Returns the learning rate for the remaining training fraction.
    ///
    /// # Arguments
    ///
    /// * `progress_remaining` - Remaining fraction, clamped to `[0, 1]` for linear decay.
    pub(crate) fn value(self, progress_remaining: f64) -> f64 {
        match self {
            Self::Constant(learning_rate) => learning_rate,
            Self::Linear(initial_learning_rate) => {
                initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
            }
        }
    }
}

/// Stop policy for the on-policy training loop.
#[derive(Debug, Clone, Copy)]
pub enum TrainingLimit {
    /// Stop after `total_rollouts` completed rollouts.
    RolloutBound {
        /// Number of rollouts after which training stops.
        total_rollouts: usize,
    },
    /// Stop after at least `total_steps` sampled environment steps.
    TotalStepBound {
        /// Number of sampled steps after which training stops.
        total_steps: usize,
    },
}

impl TrainingLimit {
    /// Creates a schedule bounded by total sampled environment steps.
    ///
    /// # Arguments
    ///
    /// * `total_steps` - Minimum number of sampled steps after which training stops.
    ///
    /// # Panics
    ///
    /// Panics if `total_steps` is zero.
    #[must_use]
    pub fn steps(total_steps: usize) -> Self {
        assert!(total_steps > 0, "total steps must be greater than zero");
        Self::TotalStepBound { total_steps }
    }

    /// Creates a schedule bounded by completed rollouts.
    ///
    /// # Arguments
    ///
    /// * `total_rollouts` - Number of completed rollouts after which training stops.
    ///
    /// # Panics
    ///
    /// Panics if `total_rollouts` is zero.
    #[must_use]
    pub fn rollouts(total_rollouts: usize) -> Self {
        assert!(
            total_rollouts > 0,
            "total rollouts must be greater than zero"
        );
        Self::RolloutBound { total_rollouts }
    }
}

enum OnPolicyCommand {
    StopTraining,
    SerializeCurrentPolicy(PathBuf),
}

enum OnPolicyCommandResult {
    Stopping,
    Stopped,
    CurrentPolicySerialized(Result<(), Error>),
}

/// Algorithm-side endpoint of an on-policy control channel.
pub(crate) struct OnPolicyControlEndpoint {
    /// Receives commands from the control handle.
    rx: Receiver<OnPolicyCommand>,
    /// Sends command results to the control handle.
    tx: Sender<OnPolicyCommandResult>,
}

impl OnPolicyControlEndpoint {
    /// Creates an algorithm-side control endpoint from its command and result channels.
    #[must_use]
    fn new(rx: Receiver<OnPolicyCommand>, tx: Sender<OnPolicyCommandResult>) -> Self {
        Self { rx, tx }
    }
}

/// Handle for controlling a running on-policy training loop.
#[derive(Debug)]
pub struct OnPolicyControlHandle {
    rx: Receiver<OnPolicyCommandResult>,
    tx: Sender<OnPolicyCommand>,
}

impl OnPolicyControlHandle {
    /// Creates a control handle from its result and command channels.
    #[must_use]
    fn new(rx: Receiver<OnPolicyCommandResult>, tx: Sender<OnPolicyCommand>) -> Self {
        Self { rx, tx }
    }

    fn send(&self, command: OnPolicyCommand) -> Result<(), Error> {
        self.tx.send(command).map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy control channel".into(),
                details: error.to_string(),
            })
        })
    }

    fn receive(&self) -> Result<OnPolicyCommandResult, Error> {
        self.rx.recv().map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy control channel".into(),
                details: error.to_string(),
            })
        })
    }

    /// Requests serialization of the current policy and waits for the result.
    ///
    /// # Arguments
    ///
    /// * `path` - Destination path for the serialized policy.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails, the training loop is stopping,
    /// or the training-side control endpoint disconnects.
    pub fn serialize_current_policy(&self, path: impl Into<PathBuf>) -> Result<(), Error> {
        self.send(OnPolicyCommand::SerializeCurrentPolicy(path.into()))?;
        match self.receive()? {
            OnPolicyCommandResult::CurrentPolicySerialized(result) => result,
            OnPolicyCommandResult::Stopping | OnPolicyCommandResult::Stopped => {
                Err(Error::InvalidState {
                    operation: "serialize current policy".into(),
                    details: "the training loop is stopping".into(),
                })
            }
        }
    }

    /// Requests that the current training loop stop and waits for it to finish.
    ///
    /// # Errors
    ///
    /// Returns an error if the training-side control endpoint has disconnected.
    pub fn stop_training(&self) -> Result<(), Error> {
        self.send(OnPolicyCommand::StopTraining)?;
        loop {
            if matches!(self.receive()?, OnPolicyCommandResult::Stopped) {
                return Ok(());
            }
        }
    }
}

/// Creates paired training and caller endpoints for on-policy control.
pub(crate) fn on_policy_control_channel() -> (OnPolicyControlEndpoint, OnPolicyControlHandle) {
    let (command_tx, command_rx) = channel();
    let (result_tx, result_rx) = channel();
    (
        OnPolicyControlEndpoint::new(command_rx, result_tx),
        OnPolicyControlHandle::new(result_rx, command_tx),
    )
}

pub(crate) struct OnPolicyCommandHandler {
    endpoint: Option<OnPolicyControlEndpoint>,
}

impl OnPolicyCommandHandler {
    pub(crate) fn new(endpoint: Option<OnPolicyControlEndpoint>) -> Self {
        Self { endpoint }
    }

    fn send_result(
        endpoint: &OnPolicyControlEndpoint,
        result: OnPolicyCommandResult,
    ) -> Result<(), Error> {
        endpoint.tx.send(result).map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy command result channel".into(),
                details: error.to_string(),
            })
        })
    }

    fn process_pending<A: Agent<Actor: ToSafetensors>, S: Sampler>(
        &self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) -> Result<HookResult, Error> {
        let Some(endpoint) = &self.endpoint else {
            return Ok(HookResult::Continue);
        };
        while let Ok(command) = endpoint.rx.try_recv() {
            match command {
                OnPolicyCommand::StopTraining => {
                    Self::send_result(endpoint, OnPolicyCommandResult::Stopping)?;
                    return Ok(HookResult::Break);
                }
                OnPolicyCommand::SerializeCurrentPolicy(path) => {
                    let result = runtime
                        .actor()
                        .to_safetensors()
                        .and_then(|bytes| std::fs::write(path, bytes).map_err(Error::wrap));
                    Self::send_result(
                        endpoint,
                        OnPolicyCommandResult::CurrentPolicySerialized(result),
                    )?;
                }
            }
        }
        Ok(HookResult::Continue)
    }

    fn notify_stopped(&self) -> Result<(), Error> {
        if let Some(endpoint) = &self.endpoint {
            Self::send_result(endpoint, OnPolicyCommandResult::Stopped)
        } else {
            Ok(())
        }
    }
}

pub(crate) enum ScheduledEvaluator<A: Actor, E: Env> {
    Disabled,
    Enabled {
        evaluator: BestPolicyEvaluator<A, E>,
        rollouts_per_evaluation: usize,
        coordinator: SharedCoordinator,
    },
}

impl<A: Actor + Clone + ToSafetensors, E: Env<Tensor: R2lTensor>> ScheduledEvaluator<A, E> {
    pub(crate) fn disabled() -> Self {
        Self::Disabled
    }

    /// Creates an evaluator scheduled using shared training progress.
    ///
    /// # Arguments
    ///
    /// * `evaluator` - Evaluates and tracks the best policy.
    /// * `rollouts_per_evaluation` - Number of training rollouts between evaluations.
    /// * `coordinator` - Training progress used to determine when evaluation is due.
    pub(crate) fn new(
        evaluator: BestPolicyEvaluator<A, E>,
        rollouts_per_evaluation: usize,
        coordinator: SharedCoordinator,
    ) -> Self {
        assert!(
            rollouts_per_evaluation > 0,
            "rollouts per evaluation must be greater than zero"
        );
        Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
            coordinator,
        }
    }

    fn evaluate<AG: Agent<Actor = A>, S: Sampler<Tensor = E::Tensor>>(
        &mut self,
        runtime: &mut OnPolicyRuntime<AG, S>,
    ) -> Result<(), Error> {
        let Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
            coordinator,
        } = self
        else {
            return Ok(());
        };
        let completed_rollouts = coordinator.borrow().completed_rollouts();
        if completed_rollouts.is_multiple_of(*rollouts_per_evaluation) {
            return evaluator.evaluate(runtime);
        }
        Ok(())
    }

    fn finish_training(&self) -> Result<(), Error> {
        let Self::Enabled { evaluator, .. } = self else {
            return Ok(());
        };
        evaluator.finish_training()
    }
}

/// Lifecycle hooks sharing progress with the sampler and learning hooks.
pub struct OnPolicyTrainingHooks<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    timing_recorder: TimingRecorder,
    coordinator: SharedCoordinator,
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
    /// * `coordinator` - Progress shared with the agent and sampler hooks.
    /// * `evaluator` - Evaluation cadence and best-policy tracking.
    /// * `command_handler` - Handles control requests between training phases.
    /// * `timing_recorder` - Records durations for each completed training iteration.
    pub(crate) fn new(
        coordinator: SharedCoordinator,
        evaluator: ScheduledEvaluator<A::Actor, E>,
        command_handler: OnPolicyCommandHandler,
        timing_recorder: TimingRecorder,
    ) -> Self {
        Self {
            coordinator,
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
        self.coordinator
            .borrow_mut()
            .update_rollout_progress(runtime);
        match self.command_handler.process_pending(runtime) {
            Ok(HookResult::Continue) => {
                self.timing_recorder.record_new_phase(Phase::Training);
                HookResult::Continue
            }
            Ok(HookResult::Break) => HookResult::Break,
            Err(error) => self.break_with_error(error),
        }
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
        if let Err(error) = evaluation_result.and(timing_result) {
            return self.break_with_error(error);
        }
        let command_result = match self.command_handler.process_pending(runtime) {
            Ok(result) => result,
            Err(error) => return self.break_with_error(error),
        };
        let hook_result = if self.coordinator.borrow().progress_remaining() <= 0.0 {
            HookResult::Break
        } else {
            command_result
        };
        self.timing_recorder.record_new_phase(Phase::Rollout);
        hook_result
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
