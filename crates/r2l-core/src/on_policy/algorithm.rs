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
    type Actor: Actor<Tensor = Self::Tensor>;

    /// Returns an actor snapshot for rollout collection.
    fn actor(&self) -> Self::Actor;

    /// Learns from a batch of trajectory containers.
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
    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]>;

    /// Releases sampler resources before the training loop exits.
    fn shutdown(&mut self) {}
}

/// Converts actors and trajectory buffers between sampler and agent tensor types.
pub trait OnPolicyAdapters<A: Actor, S: Sampler> {
    /// Actor representation accepted by the sampler.
    type SamplerActor: Actor<Tensor = S::Tensor> + Clone;
    /// Agent-facing view of one sampler trajectory.
    type AgentBuffer<'a>: TrajectoryBatch<A::Tensor>
    where
        Self: 'a,
        S: 'a;

    /// Converts an agent actor into the sampler representation.
    fn adapt_actor(&self, actor: A) -> Self::SamplerActor;

    /// Converts sampler trajectory views into agent-facing batches.
    fn adapt_buffer<'a>(
        &self,
        buffers: &'a [TrajectoryView<'a, S::Tensor>],
    ) -> impl AsRef<[Self::AgentBuffer<'a>]>
    where
        Self: 'a,
        S: 'a;
}

/// Default adapter that converts tensors through [`R2lTensor`].
pub struct DefaultAdapter;

impl<A: Actor + Clone, S: Sampler> OnPolicyAdapters<A, S> for DefaultAdapter {
    type SamplerActor = ActorWrapper<A, S::Tensor>;
    type AgentBuffer<'a>
        = TrajectoryViewsWrapper<'a, A::Tensor>
    where
        Self: 'a,
        S: 'a;

    fn adapt_actor(&self, actor: A) -> Self::SamplerActor {
        ActorWrapper::new(actor)
    }

    fn adapt_buffer<'a>(
        &self,
        buffers: &'a [TrajectoryView<'a, S::Tensor>],
    ) -> impl AsRef<[Self::AgentBuffer<'a>]>
    where
        Self: 'a,
        S: 'a,
    {
        let views: Vec<TrajectoryViewsWrapper<'a, A::Tensor>> = buffers
            .iter()
            .map(TrajectoryViewsWrapper::from_view::<S::Tensor>)
            .collect();
        views
    }
}

/// Coupled runtime unit that binds an agent, sampler, and adapter together.
pub struct OnPolicyRuntime<A: Agent, S: Sampler, C: OnPolicyAdapters<A::Actor, S> = DefaultAdapter>
{
    /// Trainable agent.
    pub agent: A,
    /// Rollout collector.
    pub sampler: S,
    /// Adapter bridging sampler and agent types.
    pub adapter: C,
}

impl<A: Agent, S: Sampler, C: OnPolicyAdapters<A::Actor, S>> OnPolicyRuntime<A, S, C> {
    /// Collects a fresh set of rollouts using the adapted actor.
    pub fn collect(&mut self) {
        let actor = self.agent.actor();
        let actor = self.adapter.adapt_actor(actor);
        self.sampler.collect_rollouts(actor);
    }

    /// Returns the last collected trajectory containers from the sampler.
    pub fn trajectory_containers(&mut self) -> impl AsRef<[TrajectoryView<'_, S::Tensor>]> {
        self.sampler.trajectory_views()
    }

    /// Adapts the sampler buffers and runs an agent update.
    pub fn learn(&mut self) -> Result<()> {
        let views = self.sampler.trajectory_views();
        let buffers = self.adapter.adapt_buffer(views.as_ref());
        self.agent.learn(buffers.as_ref())
    }

    /// Returns the agent-facing actor snapshot.
    pub fn actor(&self) -> A::Actor {
        self.agent.actor()
    }

    /// Returns the sampler-facing adapted actor snapshot.
    pub fn adapted_actor(&self) -> C::SamplerActor {
        let actor = self.agent.actor();
        self.adapter.adapt_actor(actor)
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
    /// Adapter used to bridge agent and sampler types.
    type C: OnPolicyAdapters<<Self::A as Agent>::Actor, Self::S>;

    /// Called once before rollout/training starts.
    fn init_hook(&mut self, runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>)
    -> HookResult;

    /// Called after rollouts are collected and before agent learning.
    fn post_rollout_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult;

    /// Called after the agent has learned from the latest rollouts.
    fn post_training_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> HookResult;

    /// Called once when the loop exits.
    fn shutdown_hook(
        &mut self,
        runtime: &mut OnPolicyRuntime<Self::A, Self::S, Self::C>,
    ) -> Result<()>;
}

/// Generic on-policy training loop combining a runtime with lifecycle hooks.
pub struct OnPolicyAlgorithm<
    A: Agent,
    S: Sampler,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters<A::Actor, S> = DefaultAdapter,
> {
    /// Coupled training runtime.
    pub runtime: OnPolicyRuntime<A, S, C>,
    /// Lifecycle hooks.
    pub hooks: H,
}

impl<
    A: Agent,
    S: Sampler,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters<A::Actor, S>,
> OnPolicyAlgorithm<A, S, H, C>
{
    /// Runs training until a hook requests termination.
    pub fn train(&mut self) -> Result<()> {
        return_on_hook_result!(self.hooks.init_hook(&mut self.runtime));
        loop {
            self.runtime.collect();
            break_on_hook_result!(self.hooks.post_rollout_hook(&mut self.runtime));

            self.runtime.learn()?;
            break_on_hook_result!(self.hooks.post_training_hook(&mut self.runtime));
        }

        self.hooks.shutdown_hook(&mut self.runtime)
    }
}
