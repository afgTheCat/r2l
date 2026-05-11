use anyhow::Result;

use crate::{
    HookResult, break_on_hook_result,
    buffers::TrajectoryContainer,
    models::Actor,
    on_policy::algorithm::{Agent, Sampler},
    return_on_hook_result,
    utils::{actor_wrapper::ActorWrapper, buffer_wrapper::BufferWrapper},
};

pub trait OnPolicyAlgorithmHooks {
    /// Agent type controlled by the training loop.
    type A: Agent;
    /// Sampler type controlled by the training loop.
    type S: Sampler;
    /// Adapter tp converse between types
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

pub trait OnPolicyAdapters<A: Agent, S: Sampler> {
    type SamplerActor: Actor<Tensor = S::Tensor> + Clone;
    type AgentBuffer<'a>: TrajectoryContainer<Tensor = A::Tensor>
    where
        Self: 'a,
        S::TrajectoryContainer: 'a;

    fn adapt_actor(&self, actor: A::Actor) -> Self::SamplerActor;

    fn adapt_buffer<'a>(&self, buffer: &'a S::TrajectoryContainer) -> Self::AgentBuffer<'a>;
}

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

pub struct OnPolicyAlgorithm<
    A: Agent<Actor: Clone>,
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
    /// Adapter
    pub adapter: C,
}

impl<
    A: Agent<Actor: Clone>,
    S: Sampler,
    H: OnPolicyAlgorithmHooks<A = A, S = S, C = C>,
    C: OnPolicyAdapters<A, S>,
> OnPolicyAlgorithm<A, S, H, C>
{
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
}
