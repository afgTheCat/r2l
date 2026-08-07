use anyhow::Result;

use crate::{
    HookResult, break_on_hook_result,
    buffers::{TrajectoryBatch, buffer::TrajectoryView},
    models::Actor,
    return_on_hook_result,
    tensor::R2lTensor,
    utils::{actor_wrapper::ActorWrapper, buffer_wrapper::TrajectoryViewsWrapper},
};

/// Trainable on-policy component that owns an actor and learns from rollouts.
pub trait Agent {
    /// Tensor type shared with the sampler and rollout buffers.
    type Tensor: R2lTensor;

    /// Actor type used by samplers to collect new rollouts.
    type Actor: Actor<Tensor = Self::Tensor> + Clone;

    /// Returns an actor snapshot for rollout collection.
    fn actor(&self) -> Self::Actor;

    /// Learns from a batch of trajectory containers.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent update fails.
    fn learn<B: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[B]) -> Result<()>;

    /// Sets the learning rate used by future updates.
    fn set_learning_rate(&mut self, learning_rate: f64);

    /// Releases agent resources before the training loop exits.
    fn shutdown(&mut self) {}
}

/// Rollout collector used by an on-policy training loop.
pub trait Sampler {
    /// Tensor type stored in collected trajectories.
    type Tensor: R2lTensor;

    /// Resets all environments managed by the sampler.
    fn reset_all_envs(&mut self) {}

    /// Collects rollout data using the provided actor.
    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A);

    /// Creates a view for the agents.
    fn trajectory_views(&mut self) -> impl AsRef<[TrajectoryView<'_, Self::Tensor>]>;

    /// Releases sampler resources before the training loop exits.
    fn shutdown(&mut self) {}
}

/// Coupled runtime unit that binds an agent and sampler together.
pub struct OnPolicyRuntime<A: Agent, S: Sampler> {
    /// Trainable agent.
    pub agent: A,
    /// Rollout collector.
    pub sampler: S,
}

impl<A: Agent, S: Sampler> OnPolicyRuntime<A, S> {
    /// Collects a fresh set of rollouts using the sampler-facing actor.
    pub fn collect(&mut self) {
        let actor = self.agent.actor();
        let actor = ActorWrapper::new(actor);
        self.sampler.collect_rollouts(actor);
    }

    /// Returns the last collected trajectory containers from the sampler.
    pub fn trajectory_containers(&mut self) -> impl AsRef<[TrajectoryView<'_, S::Tensor>]> {
        self.sampler.trajectory_views()
    }

    /// Adapts the sampler buffers and runs an agent update.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent cannot learn from the collected trajectories.
    pub fn learn(&mut self) -> Result<()> {
        let views = self.sampler.trajectory_views();
        let buffers = views
            .as_ref()
            .iter()
            .map(TrajectoryViewsWrapper::from_view)
            .collect::<Vec<_>>();
        self.agent.learn(&buffers)
    }

    /// Returns the agent-facing actor snapshot.
    pub fn actor(&self) -> A::Actor {
        self.agent.actor()
    }

    /// Releases agent and sampler resources.
    pub fn shutdown(&mut self) {
        self.agent.shutdown();
        self.sampler.shutdown();
    }
}

/// Lifecycle hooks that control an [`OnPolicyAlgorithm`] training loop.
pub trait OnPolicyAlgorithmHooks {
    /// Agent type controlled by the training loop.
    type A: Agent;
    /// Sampler type controlled by the training loop.
    type S: Sampler;

    /// Called once before rollout/training starts.
    fn init_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult;

    /// Called after rollouts are collected and before agent learning.
    fn post_rollout_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> HookResult;

    /// Called after the agent has learned from the latest rollouts.
    fn post_training_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>)
    -> HookResult;

    /// Called once when the loop exits.
    ///
    /// # Errors
    ///
    /// Returns an error if hook shutdown fails.
    fn shutdown_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S>) -> Result<()>;
}

/// Generic on-policy training loop combining a runtime with lifecycle hooks.
pub struct OnPolicyAlgorithm<A: Agent, S: Sampler, H: OnPolicyAlgorithmHooks<A = A, S = S>> {
    /// Coupled training runtime.
    pub runtime: OnPolicyRuntime<A, S>,
    /// Lifecycle hooks.
    pub hooks: H,
}

impl<A: Agent, S: Sampler, H: OnPolicyAlgorithmHooks<A = A, S = S>> OnPolicyAlgorithm<A, S, H> {
    /// Creates an on-policy algorithm from its runtime and lifecycle hooks.
    pub fn new(runtime: OnPolicyRuntime<A, S>, hooks: H) -> Self {
        Self { runtime, hooks }
    }

    /// Runs training until a hook requests termination.
    ///
    /// # Errors
    ///
    /// Returns an error if learning or hook shutdown fails.
    pub fn train(&mut self) -> Result<()> {
        return_on_hook_result!(self.hooks.init_hook(&mut self.runtime));
        loop {
            self.runtime.collect();
            break_on_hook_result!(self.hooks.post_rollout_hook(&mut self.runtime));

            self.runtime.learn()?;
            let hook_result = self.hooks.post_training_hook(&mut self.runtime);
            break_on_hook_result!(hook_result);
        }

        self.hooks.shutdown_hook(&mut self.runtime)
    }
}
