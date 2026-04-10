use crate::sampler::ActorWrapper;
use crate::sampler::buffer::wrapper::BufferWrapper;
use crate::{
    agents::Agent,
    sampler::{Sampler, buffer::TrajectoryContainer},
};
use anyhow::Result;

macro_rules! break_on_hook_res {
    ($hook_res:expr) => {
        if $hook_res {
            break;
        }
    };
}

pub trait OnPolicyAlgorithmHooks {
    type A: Agent;
    type S: Sampler;

    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(&mut self, rollouts: &[<Self::S as Sampler>::TrajectoryContainer])
    -> bool;

    fn post_training_hook(&mut self, policy: <Self::A as Agent>::Actor) -> bool;

    fn shutdown_hook(&mut self, agent: &mut Self::A, sampler: &mut Self::S) -> Result<()>;
}

pub struct OnPolicyAlgorithm<A: Agent, S: Sampler, H: OnPolicyAlgorithmHooks<A = A, S = S>> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

impl<
    B: TrajectoryContainer,
    A: Agent,
    S: Sampler<TrajectoryContainer = B>,
    H: OnPolicyAlgorithmHooks<A = A, S = S>,
> OnPolicyAlgorithm<A, S, H>
where
    A::Actor: Clone,
    A::Tensor: From<S::Tensor>,
    A::Tensor: From<B::Tensor>,
    S::Tensor: From<A::Tensor>,
{
    pub fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            let policy = self.agent.actor();
            let policy = ActorWrapper::new(policy);
            let buffers = self.sampler.collect_rollouts(policy);
            break_on_hook_res!(self.hooks.post_rollout_hook(buffers.as_ref()));

            let buffers = buffers
                .as_ref()
                .iter()
                .map(|b| BufferWrapper::new(b))
                .collect::<Vec<_>>();
            self.agent.learn(&buffers)?;
            let policy = self.agent.actor();
            break_on_hook_res!(self.hooks.post_training_hook(policy));
        }

        self.hooks.shutdown_hook(&mut self.agent, &mut self.sampler)
    }
}
