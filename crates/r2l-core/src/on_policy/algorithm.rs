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
    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> impl AsRef<[Self::TrajectoryContainer]>;

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
    type C: OnPolicyAdapters<Self::A, Self::S>;

    /// Called once before rollout/training starts.
    fn init_hook(&mut self) -> HookResult;

    /// Called after rollouts are collected and before agent learning.
    fn post_rollout_hook(
        &mut self,
        rollouts: &[<Self::S as Sampler>::TrajectoryContainer],
    ) -> HookResult;

    /// Called after the agent has learned from the latest rollouts.
    fn post_training_hook(
        &mut self,
        actor: <Self::A as Agent>::Actor,
        adapter: &Self::C,
    ) -> HookResult;

    /// Called once when the loop exits.
    fn shutdown_hook(
        &mut self,
        agent: &mut Self::A,
        sampler: &mut Self::S,
        adapter: &Self::C,
    ) -> Result<()>;
}

/// Adapter layer between the agent-facing and sampler-facing data types.
pub trait OnPolicyAdapters<A: Agent, S: Sampler> {
    type SamplerActor: Actor<Tensor = S::Tensor> + Clone;
    type AgentBuffer<'a>: TrajectoryContainer<Tensor = A::Tensor>
    where
        Self: 'a,
        S::TrajectoryContainer: 'a;

    fn adapt_actor(&self, actor: A::Actor) -> Self::SamplerActor;

    fn adapt_buffer<'a>(&self, buffer: &'a S::TrajectoryContainer) -> Self::AgentBuffer<'a>;
}

/// Default adapter based on tensor-converting actor and buffer wrappers.
pub struct DefaultAdapter;

impl<A: Agent<Actor: Clone>, S: Sampler> OnPolicyAdapters<A, S> for DefaultAdapter
where
    S::Tensor: From<A::Tensor>,
    A::Tensor: From<S::Tensor>,
    A::Tensor: From<<S::TrajectoryContainer as TrajectoryContainer>::Tensor>,
{
    type SamplerActor = ActorWrapper<A::Actor, S::Tensor>;
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

    fn adapt_actor(&self, actor: A::Actor) -> Self::SamplerActor {
        ActorWrapper::new(actor)
    }

    fn adapt_buffer<'a>(&self, buffer: &'a S::TrajectoryContainer) -> Self::AgentBuffer<'a> {
        BufferWrapper::new(buffer)
    }
}

// ANCHOR: on_policy_algorithm
/// Default on-policy training loop over an [`Agent`], [`Sampler`], and hooks.
pub struct OnPolicyAlgorithm<
    A: Agent,
    S: Sampler,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters<A, S> = DefaultAdapter,
> {
    /// Rollout collector.
    pub sampler: S,
    /// Trainable agent.
    pub agent: A,
    /// Lifecycle hooks.
    pub hooks: H,
    /// Adapter bridging sampler and agent types.
    pub adapter: C,
}
// ANCHOR_END: on_policy_algorithm

impl<
    A: Agent,
    S: Sampler,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters<A, S>,
> OnPolicyAlgorithm<A, S, H, C>
{
    // ANCHOR: train_loop
    /// Runs rollout collection and learning until a hook requests shutdown.
    pub fn train(&mut self) -> Result<()> {
        return_on_hook_result!(self.hooks.init_hook());
        loop {
            let actor = self.agent.actor();
            let actor = self.adapter.adapt_actor(actor);
            let buffers = self.sampler.collect_rollouts(actor);
            break_on_hook_result!(self.hooks.post_rollout_hook(buffers.as_ref()));

            let buffers = buffers
                .as_ref()
                .iter()
                .map(|b| self.adapter.adapt_buffer(b))
                .collect::<Vec<_>>();
            self.agent.learn(&buffers)?;
            let actor = self.agent.actor();
            break_on_hook_result!(self.hooks.post_training_hook(actor, &self.adapter));
        }

        self.hooks
            .shutdown_hook(&mut self.agent, &mut self.sampler, &self.adapter)
    }
    // ANCHOR_END: train_loop
}
