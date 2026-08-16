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

use crate::evaluators::best_actor_evaluator::BestActorEvaluator;

/// Commands processed by the default on-policy hooks at training boundaries.
pub enum OnPolicyCommand {
    /// Stops training before the next learning phase or after the current one.
    Shutdown,
    /// Serializes the current runtime actor to the given path.
    SerializeCurrentPolicy(String),
}

/// Acknowledgements sent after an on-policy command has been processed.
pub enum OnPolicyCommandResult {
    /// Training is stopping and runtime cleanup will follow.
    Stopping,
    /// Training stopped completely and runtime cleanup has happened.
    Stopped,
    /// Result of attempting to serialize the current runtime actor.
    CurrentPolicySerialized(Result<(), Error>),
}

/// Algorithm-side endpoint for receiving on-policy commands.
pub struct OnPolicyCommandReceiver {
    /// Receives commands from the user-side endpoint.
    pub rx: Receiver<OnPolicyCommand>,
    /// Sends command results to the user-side endpoint.
    pub tx: Sender<OnPolicyCommandResult>,
}

impl OnPolicyCommandReceiver {
    /// Creates an algorithm-side endpoint from its command and result channels.
    #[must_use]
    pub fn new(rx: Receiver<OnPolicyCommand>, tx: Sender<OnPolicyCommandResult>) -> Self {
        Self { rx, tx }
    }
}

/// User-side endpoint for sending commands to an on-policy training loop.
#[derive(Debug)]
pub struct OnPolicyCommandSender {
    /// Receives command results from the training loop.
    pub rx: Receiver<OnPolicyCommandResult>,
    /// Sends commands to the training loop.
    pub tx: Sender<OnPolicyCommand>,
}

impl OnPolicyCommandSender {
    /// Creates a user-side endpoint from its result and command channels.
    #[must_use]
    pub fn new(rx: Receiver<OnPolicyCommandResult>, tx: Sender<OnPolicyCommand>) -> Self {
        Self { rx, tx }
    }

    /// Shuts down the on-policy algorithm gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if the training-side command receiver has disconnected.
    pub fn shutdown(&self) -> Result<(), Error> {
        self.tx.send(OnPolicyCommand::Shutdown).map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy command channel".into(),
                details: error.to_string(),
            })
        })?;
        while self.rx.recv().is_ok() {}
        Ok(())
    }
}

/// Creates the algorithm-side receiver and user-side sender for on-policy commands.
#[must_use]
pub fn on_policy_command_channel() -> (OnPolicyCommandReceiver, OnPolicyCommandSender) {
    let (command_tx, command_rx) = channel();
    let (result_tx, result_rx) = channel();
    (
        OnPolicyCommandReceiver::new(command_rx, result_tx),
        OnPolicyCommandSender::new(result_rx, command_tx),
    )
}

const TRAINING_TIMINGS_FILE: &str = "training_timings.csv";

macro_rules! break_on_error {
    ($hooks:expr, $body:block) => {{
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
pub enum LearningSchedule {
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

impl LearningSchedule {
    /// Creates a schedule bounded by total sampled environment steps.
    #[must_use]
    pub fn total_step_bound(total_steps: usize) -> Self {
        assert!(total_steps > 0, "total steps must be greater than zero");
        Self::TotalStepBound { total_steps }
    }

    /// Creates a schedule bounded by completed rollouts.
    #[must_use]
    pub fn rollout_bound(total_rollouts: usize) -> Self {
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

    fn shutdown(&mut self) -> Result<(), Error> {
        let Self::Enabled { evaluator, .. } = self else {
            return Ok(());
        };
        let result = evaluator.try_write_artifacts();
        evaluator.shutdown();
        result
    }
}

pub(crate) struct OnPolicyCommandHandler {
    receiver: Option<OnPolicyCommandReceiver>,
}

impl OnPolicyCommandHandler {
    pub(crate) fn new(receiver: Option<OnPolicyCommandReceiver>) -> Self {
        Self { receiver }
    }

    fn process_pending<A: Agent<Actor: ToSafetensors>, S: Sampler>(
        &self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) -> Result<HookResult, Error> {
        let Some(receiver) = &self.receiver else {
            return Ok(HookResult::Continue);
        };
        while let Ok(command) = receiver.rx.try_recv() {
            match command {
                OnPolicyCommand::Shutdown => {
                    Self::send_result(receiver, OnPolicyCommandResult::Stopping)?;
                    return Ok(HookResult::Break);
                }
                OnPolicyCommand::SerializeCurrentPolicy(path) => {
                    let result = runtime.actor().to_safetensors().and_then(|bytes| {
                        std::fs::write(PathBuf::from(path), bytes).map_err(Error::wrap)
                    });
                    Self::send_result(
                        receiver,
                        OnPolicyCommandResult::CurrentPolicySerialized(result),
                    )?;
                }
            }
        }
        Ok(HookResult::Continue)
    }

    fn send_result(
        receiver: &OnPolicyCommandReceiver,
        result: OnPolicyCommandResult,
    ) -> Result<(), Error> {
        receiver.tx.send(result).map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy command result channel".into(),
                details: error.to_string(),
            })
        })
    }

    fn notify_stopped(&self) {
        if let Some(receiver) = &self.receiver {
            let _ = receiver.tx.send(OnPolicyCommandResult::Stopped);
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

    fn finish_phase(&mut self, phase: TrainingPhase) -> Result<(), Error> {
        let Self::Enabled(recorder) = self else {
            return Ok(());
        };
        recorder.finish_phase(phase)
    }

    fn finish_evaluation(&mut self, completed_rollouts: usize) -> Result<(), Error> {
        let Self::Enabled(recorder) = self else {
            return Ok(());
        };
        recorder.finish_evaluation(completed_rollouts)
    }
}

impl EnabledTrainingTimingRecorder {
    fn finish_phase(&mut self, phase: TrainingPhase) -> Result<(), Error> {
        let now = Instant::now();
        let duration = now - self.phase_started;
        match phase {
            TrainingPhase::Collection => self.current.collection = duration,
            TrainingPhase::Training => self.current.training = duration,
        }
        Ok(())
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

/// Default lifecycle coordination for an on-policy training loop.
pub struct DefaultOnPolicyAlgorithmHooks<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    state: TrainingLoopState,
    learning_schedule: LearningSchedule,
    learning_rate_scheduler: LearningRateScheduler,
    evaluator: ScheduledEvaluator<A::Actor, E>,
    command_handler: OnPolicyCommandHandler,
    timing_recorder: TrainingTimingRecorder,
    error: Option<Error>,
    _phantom: PhantomData<S>,
}

impl<A: Agent<Actor: ToSafetensors>, S: Sampler, E: Env<Tensor = S::Tensor>>
    DefaultOnPolicyAlgorithmHooks<A, S, E>
{
    pub(crate) fn new(
        learning_schedule: LearningSchedule,
        learning_rate_schedule: Option<LearningRateSchedule>,
        evaluator: ScheduledEvaluator<A::Actor, E>,
        command_handler: OnPolicyCommandHandler,
        timing_recorder: TrainingTimingRecorder,
    ) -> Self {
        Self {
            state: TrainingLoopState::default(),
            learning_schedule,
            learning_rate_scheduler: LearningRateScheduler::new(learning_rate_schedule),
            evaluator,
            command_handler,
            timing_recorder,
            error: None,
            _phantom: PhantomData,
        }
    }

    fn progress_remaining(&self) -> f64 {
        match &self.learning_schedule {
            LearningSchedule::RolloutBound { total_rollouts } => {
                1.0 - self.state.completed_rollouts as f64 / *total_rollouts as f64
            }
            LearningSchedule::TotalStepBound { total_steps } => {
                1.0 - self.state.steps_taken as f64 / *total_steps as f64
            }
        }
    }

    fn finish_collection(&mut self, runtime: &mut OnPolicyRuntime<A, S>) -> Result<(), Error> {
        self.timing_recorder
            .finish_phase(TrainingPhase::Collection)?;
        let rollouts = runtime.trajectory_containers();
        self.state.steps_taken += rollouts
            .as_ref()
            .iter()
            .map(|trajectory| trajectory.actions.len())
            .sum::<usize>();
        drop(rollouts);
        self.learning_rate_scheduler
            .update(self.progress_remaining(), runtime);
        Ok(())
    }

    fn finish_training_and_evaluate(
        &mut self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) -> Result<(), Error> {
        self.timing_recorder.finish_phase(TrainingPhase::Training)?;
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
    for DefaultOnPolicyAlgorithmHooks<A, S, E>
{
    type A = A;
    type S = S;

    fn init_hook(&mut self, _runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        self.timing_recorder.init();
        HookResult::Continue
    }

    fn post_rollout_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        let command_result = break_on_error!(self, {
            self.finish_collection(runtime)?;
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

    fn shutdown_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> Result<(), Error> {
        let evaluator_result = self.evaluator.shutdown();
        runtime.shutdown();
        self.command_handler.notify_stopped();
        match self.error.take() {
            Some(error) => Err(error),
            None => evaluator_result,
        }
    }
}
