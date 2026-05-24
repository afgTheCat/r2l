use anyhow::Result;

use crate::{
    HookResult, break_on_hook_result,
    buffers::{buffer::TrajectoryView, gen_buffer::TrajectoryBatchT},
    models::Actor,
    return_on_hook_result,
    tensor::R2lTensor,
    utils::{actor_wrapper::ActorWrapper, buffer_wrapper2::TrajectoryViewsWrapper},
};

pub trait Agent2 {
    /// Tensor type shared with the sampler and rollout buffers.
    type Tensor: R2lTensor;

    /// Actor type used by samplers to collect new rollouts.
    type Actor: Actor<Tensor = Self::Tensor>;

    /// Returns an actor snapshot for rollout collection.
    fn actor(&self) -> Self::Actor;

    /// Learns from a batch of trajectory containers.
    fn learn<B: TrajectoryBatchT<Self::Tensor>>(&mut self, buffers: &[B]) -> Result<()>;

    /// Releases agent resources before the training loop exits.
    fn shutdown(&mut self) {}
}

pub trait Sampler2 {
    type Tensor: R2lTensor;

    /// Collects rollout data using the provided actor.
    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A);

    /// Creates a view for the agents.
    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]>;

    /// Releases sampler resources before the training loop exits.
    fn shutdown(&mut self) {}
}

pub trait OnPolicyAdapters2<A: Actor, S: Sampler2> {
    type SamplerActor: Actor<Tensor = S::Tensor> + Clone;
    type AgentBuffer<'a>: TrajectoryBatchT<A::Tensor>
    where
        Self: 'a,
        S: 'a;

    fn adapt_actor(&self, actor: A) -> Self::SamplerActor;

    fn adapt_buffer<'a>(
        &self,
        buffers: &'a [TrajectoryView<'a, S::Tensor>],
    ) -> impl AsRef<[Self::AgentBuffer<'a>]>
    where
        Self: 'a,
        S: 'a;
}

pub struct DefaultAdapter;

impl<A: Actor + Clone, S: Sampler2> OnPolicyAdapters2<A, S> for DefaultAdapter
where
    S::Tensor: From<A::Tensor>,
    A::Tensor: From<S::Tensor>,
{
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
pub struct OnPolicyRuntime<
    A: Agent2,
    S: Sampler2,
    C: OnPolicyAdapters2<A::Actor, S> = DefaultAdapter,
> {
    /// Trainable agent.
    pub agent: A,
    /// Rollout collector.
    pub sampler: S,
    /// Adapter bridging sampler and agent types.
    pub adapter: C,
}

impl<A: Agent2, S: Sampler2, C: OnPolicyAdapters2<A::Actor, S>> OnPolicyRuntime<A, S, C> {
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

pub trait OnPolicyAlgorithmHooks {
    /// Agent type controlled by the training loop.
    type A: Agent2;
    /// Sampler type controlled by the training loop.
    type S: Sampler2;
    /// Adapter used to bridge agent and sampler types.
    type C: OnPolicyAdapters2<<Self::A as Agent2>::Actor, Self::S>;

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

pub struct OnPolicyAlgorithm<
    A: Agent2,
    S: Sampler2,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters2<A::Actor, S> = DefaultAdapter,
> {
    /// Coupled training runtime.
    pub runtime: OnPolicyRuntime<A, S, C>,
    /// Lifecycle hooks.
    pub hooks: H,
}

impl<
    A: Agent2,
    S: Sampler2,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters2<A::Actor, S>,
> OnPolicyAlgorithm<A, S, H, C>
{
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
