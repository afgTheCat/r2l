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

use crate::evaluator::BestActorEvaluator;

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

/// Creates paired algorithm and caller endpoints for controlling on-policy training.
#[must_use]
pub(crate) fn on_policy_control_channel() -> (OnPolicyControlEndpoint, OnPolicyControlHandle) {
    let (command_tx, command_rx) = channel();
    let (result_tx, result_rx) = channel();
    (
        OnPolicyControlEndpoint::new(command_rx, result_tx),
        OnPolicyControlHandle::new(result_rx, command_tx),
    )
}

const TRAINING_TIMINGS_FILE: &str = "training_timings.csv";

macro_rules! break_on_error {
    ($hooks:expr, $body:block) => {{
        #[allow(clippy::redundant_closure_call)]
        match (|| -> Result<HookResult, Error> { $body })() {
            Ok(value) => value,
            Err(error) => return ($hooks).break_with_error(error),
        }
    }};
}

#[derive(Clone, Copy)]
enum TrainingPhase {
    Collection,
    Training,
}

/// Training-stop policy for the on-policy training loop.
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

/// Learning-rate policy applied over the progress of an on-policy training run.
#[derive(Debug, Clone, Copy)]
pub enum LearningRateSchedule {
    /// Keep the learning rate fixed throughout training.
    Constant(f64),
    /// Decay the initial learning rate linearly to zero.
    Linear(f64),
}

pub(crate) struct LearningRateScheduler {
    schedule: Option<LearningRateSchedule>,
}

impl LearningRateScheduler {
    fn new(schedule: Option<LearningRateSchedule>) -> Self {
        Self { schedule }
    }

    fn update<A: Agent, S: Sampler>(
        &self,
        progress_remaining: f64,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) {
        if let Some(schedule) = self.schedule {
            let learning_rate = match schedule {
                LearningRateSchedule::Constant(learning_rate) => learning_rate,
                LearningRateSchedule::Linear(initial_learning_rate) => {
                    initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
                }
            };
            runtime.agent.set_learning_rate(learning_rate);
        }
    }
}

pub(crate) enum ScheduledEvaluator<A: Actor, E: Env> {
    Disabled,
    Enabled {
        evaluator: BestActorEvaluator<A, E>,
        rollouts_per_evaluation: usize,
    },
}

impl<A: Actor + Clone + ToSafetensors, E: Env<Tensor: R2lTensor>> ScheduledEvaluator<A, E> {
    pub(crate) fn disabled() -> Self {
        Self::Disabled
    }

    pub(crate) fn new(evaluator: BestActorEvaluator<A, E>, rollouts_per_evaluation: usize) -> Self {
        assert!(
            rollouts_per_evaluation > 0,
            "rollouts per evaluation must be greater than zero"
        );
        Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
        }
    }

    fn evaluate<AG: Agent<Actor = A>, S: Sampler<Tensor = E::Tensor>>(
        &mut self,
        runtime: &mut OnPolicyRuntime<AG, S>,
        completed_rollouts: usize,
    ) -> Result<(), Error> {
        let Self::Enabled {
            evaluator,
            rollouts_per_evaluation,
        } = self
        else {
            return Ok(());
        };
        if (completed_rollouts + 1).is_multiple_of(*rollouts_per_evaluation) {
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

#[derive(Default)]
struct TrainingLoopTimings {
    collection: Duration,
    training: Duration,
    evaluation: Duration,
    total: Duration,
}

pub(crate) enum TrainingTimingRecorder {
    Disabled,
    Enabled(EnabledTrainingTimingRecorder),
}

pub(crate) struct EnabledTrainingTimingRecorder {
    file: File,
    training_started: Instant,
    phase_started: Instant,
    current: TrainingLoopTimings,
}

impl TrainingTimingRecorder {
    pub(crate) fn disabled() -> Self {
        Self::Disabled
    }

    pub(crate) fn create(output_dir: &Path) -> Result<Self, Error> {
        std::fs::create_dir_all(output_dir).map_err(Error::wrap)?;
        let mut file = File::create(output_dir.join(TRAINING_TIMINGS_FILE)).map_err(Error::wrap)?;
        writeln!(file, "rollout,collect_ms,learn_ms,evaluate_ms,total_ms").map_err(Error::wrap)?;
        let now = Instant::now();
        Ok(Self::Enabled(EnabledTrainingTimingRecorder {
            file,
            training_started: now,
            phase_started: now,
            current: TrainingLoopTimings::default(),
        }))
    }

    fn init(&mut self) {
        let Self::Enabled(recorder) = self else {
            return;
        };
        let now = Instant::now();
        recorder.training_started = now;
        recorder.phase_started = now;
    }

    fn start_phase(&mut self) {
        let Self::Enabled(recorder) = self else {
            return;
        };
        recorder.phase_started = Instant::now();
    }

    fn finish_phase(&mut self, phase: TrainingPhase) {
        let Self::Enabled(recorder) = self else {
            return;
        };
        recorder.finish_phase(phase);
    }

    fn finish_evaluation(&mut self, completed_rollouts: usize) -> Result<(), Error> {
        let Self::Enabled(recorder) = self else {
            return Ok(());
        };
        recorder.finish_evaluation(completed_rollouts)
    }
}

impl EnabledTrainingTimingRecorder {
    fn finish_phase(&mut self, phase: TrainingPhase) {
        let now = Instant::now();
        let duration = now - self.phase_started;
        match phase {
            TrainingPhase::Collection => self.current.collection = duration,
            TrainingPhase::Training => self.current.training = duration,
        }
    }

    fn finish_evaluation(&mut self, completed_rollouts: usize) -> Result<(), Error> {
        let now = Instant::now();
        self.current.evaluation = now - self.phase_started;
        self.current.total = now - self.training_started;
        let timings = std::mem::take(&mut self.current);
        writeln!(
            self.file,
            "{},{:.3},{:.3},{:.3},{:.3}",
            completed_rollouts,
            timings.collection.as_secs_f64() * 1000.0,
            timings.training.as_secs_f64() * 1000.0,
            timings.evaluation.as_secs_f64() * 1000.0,
            timings.total.as_secs_f64() * 1000.0,
        )
        .map_err(Error::wrap)
    }
}

#[derive(Default)]
struct TrainingLoopState {
    completed_rollouts: usize,
    steps_taken: usize,
}

/// Lifecycle hooks for an on-policy training loop.
pub struct OnPolicyTrainingHooks<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    state: TrainingLoopState,
    training_limit: TrainingLimit,
    learning_rate_scheduler: LearningRateScheduler,
    evaluator: ScheduledEvaluator<A::Actor, E>,
    command_handler: OnPolicyCommandHandler,
    timing_recorder: TrainingTimingRecorder,
    error: Option<Error>,
    _phantom: PhantomData<S>,
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>>
    OnPolicyTrainingHooks<A, S, E>
{
    pub(crate) fn new(
        training_limit: TrainingLimit,
        learning_rate_schedule: Option<LearningRateSchedule>,
        evaluator: ScheduledEvaluator<A::Actor, E>,
        command_handler: OnPolicyCommandHandler,
        timing_recorder: TrainingTimingRecorder,
    ) -> Self {
        Self {
            state: TrainingLoopState::default(),
            training_limit,
            learning_rate_scheduler: LearningRateScheduler::new(learning_rate_schedule),
            evaluator,
            command_handler,
            timing_recorder,
            error: None,
            _phantom: PhantomData,
        }
    }

    fn progress_remaining(&self) -> f64 {
        match &self.training_limit {
            TrainingLimit::RolloutBound { total_rollouts } => {
                1.0 - self.state.completed_rollouts as f64 / *total_rollouts as f64
            }
            TrainingLimit::TotalStepBound { total_steps } => {
                1.0 - self.state.steps_taken as f64 / *total_steps as f64
            }
        }
    }

    fn finish_collection(&mut self, runtime: &mut OnPolicyRuntime<A, S>) {
        self.timing_recorder.finish_phase(TrainingPhase::Collection);
        let rollouts = runtime.trajectory_containers();
        self.state.steps_taken += rollouts
            .as_ref()
            .iter()
            .map(|trajectory| trajectory.actions.len())
            .sum::<usize>();
        drop(rollouts);
        self.learning_rate_scheduler
            .update(self.progress_remaining(), runtime);
    }

    fn finish_training_and_evaluate(
        &mut self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) -> Result<(), Error> {
        self.timing_recorder.finish_phase(TrainingPhase::Training);
        self.timing_recorder.start_phase();
        let evaluation_result = self
            .evaluator
            .evaluate(runtime, self.state.completed_rollouts);
        let completed_rollouts = self.state.completed_rollouts + 1;
        let timing_result = self.timing_recorder.finish_evaluation(completed_rollouts);
        self.state.completed_rollouts += 1;
        evaluation_result.and(timing_result)
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
        let command_result = break_on_error!(self, {
            self.finish_collection(runtime);
            self.command_handler.process_pending(runtime)
        });
        self.timing_recorder.start_phase();
        command_result
    }

    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> HookResult {
        let command_result = break_on_error!(self, {
            self.finish_training_and_evaluate(runtime)?;
            self.command_handler.process_pending(runtime)
        });
        let hook_result = if self.progress_remaining() <= 0.0 {
            HookResult::Break
        } else {
            command_result
        };
        self.timing_recorder.start_phase();
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
