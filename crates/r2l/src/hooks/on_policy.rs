use std::{
    fs::File,
    io::Write,
    marker::PhantomData,
    path::PathBuf,
    sync::mpsc::{Receiver, Sender},
    time::Instant,
};

use anyhow::Result;
use r2l_core::{
    HookResult,
    env::Env,
    models::Actor,
    on_policy::algorithm::{Agent, OnPolicyAlgorithmHooks, OnPolicyRuntime, Sampler},
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

use crate::evaluators::best_actor_evaluator::BestActorEvaluator;

pub struct PerformanceLog {
    pub file: File,
    pub training_started: Instant,
    pub rollout_started: Instant,
    pub phase_started: Instant,
    pub collect_ms: f64,
    pub rollout: usize,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// Training-stop policy for [`DefaultOnPolicyAlgorithmHooks`].
///
/// This determines when the outer on-policy training loop should terminate,
/// either after a fixed number of rollouts or after a fixed number of sampled
/// environment steps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LearningSchedule {
    /// Stop after `total_rollouts` completed rollout collections.
    RolloutBound {
        /// Number of rollout collections after which training stops.
        total_rollouts: usize,
        /// Number of rollout collections completed so far.
        current_rollout: usize,
    },
    /// Stop after at least `total_steps` sampled environment steps.
    TotalStepBound {
        /// Number of sampled steps after which training stops.
        total_steps: usize,
        /// Number of sampled steps completed so far.
        current_step: usize,
    },
}

impl LearningSchedule {
    /// Creates a schedule bounded by total sampled environment steps.
    pub fn total_step_bound(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps,
            current_step: 0,
        }
    }

    /// Creates a schedule bounded by completed rollout collections.
    pub fn rollout_bound(total_rollouts: usize) -> Self {
        Self::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }
    }
}

/// Learning-rate policy applied over the progress of an on-policy training run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Keep the learning rate fixed throughout training.
    Constant(f64),
    /// Decay the initial learning rate linearly to zero.
    Linear(f64),
}

impl LearningRateSchedule {
    /// Returns the learning rate for the remaining fraction of training.
    pub fn value(self, progress_remaining: f64) -> f64 {
        match self {
            Self::Constant(learning_rate) => learning_rate,
            Self::Linear(initial_learning_rate) => {
                initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
            }
        }
    }
}

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
    /// Training stopped completely, runtime cleanup has happened
    Stopped,
    /// The current runtime actor was serialized.
    CurrentPolicySerialized,
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
    pub fn new(rx: Receiver<OnPolicyCommandResult>, tx: Sender<OnPolicyCommand>) -> Self {
        Self { rx, tx }
    }

    /// Shuts down the OnPolicyAlgorithm gracefully.
    pub fn shutdown(&self) {
        self.tx.send(OnPolicyCommand::Shutdown).unwrap();
        // empty the response queue
        while self.rx.recv().is_ok() {}
    }
}

/// Creates the algorithm-side receiver and user-side sender for on-policy commands.
pub fn on_policy_command_channel() -> (OnPolicyCommandReceiver, OnPolicyCommandSender) {
    let (command_tx, command_rx) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    (
        OnPolicyCommandReceiver::new(command_rx, result_tx),
        OnPolicyCommandSender::new(result_rx, command_tx),
    )
}

/// Default outer-loop hooks used by high-level on-policy algorithm builders.
///
/// This hook is responsible for lifecycle behavior around training rather than
/// algorithm-specific loss logic. It tracks rollout progress, applies the
/// configured [`LearningSchedule`] to decide when training should stop,
/// optionally evaluates the current actor, and shuts down the runtime when the
/// algorithm exits.
pub struct DefaultOnPolicyAlgorithmHooks<A: Agent, S: Sampler, E: Env<Tensor = S::Tensor>> {
    pub(crate) learning_schedule: LearningSchedule,
    pub(crate) learning_rate_schedule: Option<LearningRateSchedule>,
    pub(crate) evaluator: Option<BestActorEvaluator<A::Actor, E>>,
    pub(crate) performance_log: Option<PerformanceLog>,
    pub(crate) command_rx: Option<OnPolicyCommandReceiver>,
    pub(crate) _phantom: PhantomData<(A, S, E)>,
}

impl<A: Agent, S: Sampler<Tensor: R2lTensor>, E: Env<Tensor = S::Tensor>>
    DefaultOnPolicyAlgorithmHooks<A, S, E>
{
    fn process_pending_commands(&self, runtime: &mut OnPolicyRuntime<A, S>) -> HookResult {
        let Some(command_rx) = &self.command_rx else {
            return HookResult::Continue;
        };
        while let Ok(command) = command_rx.rx.try_recv() {
            match command {
                OnPolicyCommand::Shutdown => {
                    command_rx.tx.send(OnPolicyCommandResult::Stopping).unwrap();
                    return HookResult::Break;
                }
                OnPolicyCommand::SerializeCurrentPolicy(path) => {
                    let path = PathBuf::from(path);
                    let policy_serialized = runtime.actor().try_serialize().unwrap();
                    std::fs::write(path, policy_serialized).unwrap();
                    command_rx
                        .tx
                        .send(OnPolicyCommandResult::CurrentPolicySerialized)
                        .unwrap();
                }
            }
        }
        HookResult::Continue
    }

    fn mark_progress(&mut self, runtime: &mut OnPolicyRuntime<A, S>) {
        match &mut self.learning_schedule {
            LearningSchedule::RolloutBound {
                current_rollout, ..
            } => *current_rollout += 1,
            LearningSchedule::TotalStepBound { current_step, .. } => {
                let rollouts = runtime.trajectory_containers();
                let rollout_steps: usize = rollouts.as_ref().iter().map(|e| e.actions.len()).sum();
                *current_step += rollout_steps;
            }
        }
    }

    fn progress_remaining(&self) -> f64 {
        match self.learning_schedule {
            LearningSchedule::RolloutBound {
                total_rollouts,
                current_rollout,
            } => 1.0 - current_rollout as f64 / total_rollouts as f64,
            LearningSchedule::TotalStepBound {
                total_steps,
                current_step,
            } => 1.0 - current_step as f64 / total_steps as f64,
        }
    }
}

impl<A: Agent, S: Sampler<Tensor: R2lTensor>, E: Env<Tensor = S::Tensor>> OnPolicyAlgorithmHooks
    for DefaultOnPolicyAlgorithmHooks<A, S, E>
{
    type A = A;
    type S = S;

    fn init_hook(&mut self, _runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        if let Some(performance_log) = &mut self.performance_log {
            let now = Instant::now();
            performance_log.training_started = now;
            performance_log.rollout_started = now;
            performance_log.phase_started = now;
        }
        HookResult::Continue
    }

    fn post_rollout_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult {
        if let Some(performance_log) = &mut self.performance_log {
            performance_log.collect_ms = elapsed_ms(performance_log.phase_started);
        }
        self.mark_progress(runtime);
        if let Some(learning_rate_schedule) = self.learning_rate_schedule {
            let learning_rate = match learning_rate_schedule {
                LearningRateSchedule::Constant(learning_rate) => learning_rate,
                LearningRateSchedule::Linear(initial_learning_rate) => {
                    let progress_remaining = self.progress_remaining();
                    initial_learning_rate * progress_remaining.clamp(0.0, 1.0)
                }
            };
            runtime.agent.set_learning_rate(learning_rate);
        }
        let command_result = self.process_pending_commands(runtime);
        if let Some(performance_log) = &mut self.performance_log {
            performance_log.phase_started = Instant::now();
        }
        command_result
    }

    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S>,
    ) -> HookResult {
        let learn_ms = self
            .performance_log
            .as_ref()
            .map_or(0.0, |log| elapsed_ms(log.phase_started));
        let evaluate_ms = self.evaluator.as_mut().and_then(|evaluator| {
            let evaluation_started = Instant::now();
            evaluator
                .eval(runtime)
                .then(|| elapsed_ms(evaluation_started))
        });
        let command_res = self.process_pending_commands(runtime);
        let hook_result = if self.progress_remaining() <= 0. {
            HookResult::Break
        } else {
            command_res
        };
        if let Some(performance_log) = &mut self.performance_log {
            performance_log.rollout += 1;
            let evaluate_ms = evaluate_ms
                .map(|duration| format!("{duration:.3}"))
                .unwrap_or_default();
            writeln!(
                performance_log.file,
                "{},{:.3},{:.3},{},{:.3},{:.3}",
                performance_log.rollout,
                performance_log.collect_ms,
                learn_ms,
                evaluate_ms,
                elapsed_ms(performance_log.rollout_started),
                elapsed_ms(performance_log.training_started),
            )
            .unwrap();
            performance_log.rollout_started = Instant::now();
            performance_log.phase_started = performance_log.rollout_started;
        }
        hook_result
    }

    fn shutdown_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> Result<()> {
        if let Some(evaluator) = &mut self.evaluator {
            evaluator.try_write_artifacts()?;
            evaluator.shutdown();
        }
        runtime.shutdown();
        if let Some(command_rx) = &self.command_rx {
            command_rx.tx.send(OnPolicyCommandResult::Stopped).unwrap();
        }
        Ok(())
    }
}
