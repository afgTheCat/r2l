use anyhow::Result;

use crate::buffers::TrajectoryContainer;
use crate::models::Actor;
use crate::tensor::R2lTensor;
use crate::utils::actor_wrapper::ActorWrapper;
use crate::utils::buffer_wrapper::BufferWrapper;
use crate::{HookResult, break_on_hook_result, return_on_hook_result};

// ANCHOR: agent
/// Trainable component that updates from collected trajectories.
pub trait Agent {
    /// Tensor type shared with the sampler and rollout buffers.
    type Tensor: R2lTensor;

    /// Actor type used by samplers to collect new rollouts.
    type Actor: Actor<Tensor = Self::Tensor>;

    /// Returns an actor snapshot for rollout collection.
    fn actor(&self) -> Self::Actor;

    /// Learns from a batch of trajectory containers.
    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(&mut self, buffers: &[C])
    -> Result<()>;

    /// Releases agent resources before the training loop exits.
    fn shutdown(&mut self) {}
}
// ANCHOR_END: agent

// ANCHOR: sampler
/// Rollout collector used by [`OnPolicyAlgorithm`].
pub trait Sampler {
    /// Tensor type produced by environments and consumed by actors.
    type Tensor: R2lTensor;
    /// Trajectory buffer type returned after rollout collection.
    type TrajectoryContainer: TrajectoryContainer<Tensor = Self::Tensor>;

    /// Collects rollout data using the provided actor.
    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A);

    fn trajectory_containers(&mut self) -> impl AsRef<[Self::TrajectoryContainer]>;

    /// Releases sampler resources before the training loop exits.
    fn shutdown(&mut self) {}
}
// ANCHOR_END: sampler

/// Lifecycle hooks for [`OnPolicyAlgorithm`].
///
/// Hook methods return [`HookResult::Break`] to stop the training loop at that
/// point.
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

// TODO: we probably only need to
/// Adapter layer between the agent-facing and sampler-facing data types.
pub trait OnPolicyAdapters<A: Actor, S: Sampler> {
    type SamplerActor: Actor<Tensor = S::Tensor> + Clone;
    type AgentBuffer<'a>: TrajectoryContainer<Tensor = A::Tensor>
    where
        Self: 'a,
        S::TrajectoryContainer: 'a;

    fn adapt_actor(&self, actor: A) -> Self::SamplerActor;

    fn adapt_buffer<'a>(&self, buffer: &'a S::TrajectoryContainer) -> Self::AgentBuffer<'a>;
}

/// Default adapter based on tensor-converting actor and buffer wrappers.
pub struct DefaultAdapter;

impl<A: Actor + Clone, S: Sampler> OnPolicyAdapters<A, S> for DefaultAdapter
where
    S::Tensor: From<A::Tensor>,
    A::Tensor: From<S::Tensor>,
    A::Tensor: From<<S::TrajectoryContainer as TrajectoryContainer>::Tensor>,
{
    type SamplerActor = ActorWrapper<A, S::Tensor>;
    type AgentBuffer<'a>
        = BufferWrapper<
        'a,
        <S::TrajectoryContainer as TrajectoryContainer>::Tensor,
        A::Tensor,
        S::TrajectoryContainer,
    >
    where
        Self: 'a,
        S::TrajectoryContainer: 'a;

    fn adapt_actor(&self, actor: A) -> Self::SamplerActor {
        ActorWrapper::new(actor)
    }

    fn adapt_buffer<'a>(&self, buffer: &'a S::TrajectoryContainer) -> Self::AgentBuffer<'a> {
        BufferWrapper::new(buffer)
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
    pub fn trajectory_containers(&mut self) -> impl AsRef<[S::TrajectoryContainer]> {
        self.sampler.trajectory_containers()
    }

    /// Adapts the sampler buffers and runs an agent update.
    pub fn learn(&mut self) -> Result<()> {
        let trajectory_buffers = self.sampler.trajectory_containers();
        let buffers = trajectory_buffers
            .as_ref()
            .iter()
            .map(|b| self.adapter.adapt_buffer(b))
            .collect::<Vec<_>>();
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

// ANCHOR: on_policy_algorithm
/// Default on-policy training loop over an [`Agent`], [`Sampler`], and hooks.
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
// ANCHOR_END: on_policy_algorithm

impl<
    A: Agent,
    S: Sampler,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters<A::Actor, S>,
> OnPolicyAlgorithm<A, S, H, C>
{
    // ANCHOR: train_loop
    /// Runs rollout collection and learning until a hook requests shutdown.
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
    // ANCHOR_END: train_loop
}
