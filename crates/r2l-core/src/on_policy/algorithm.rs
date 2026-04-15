use anyhow::Result;

use crate::buffers::TrajectoryContainer;
use crate::models::Actor;
use crate::tensor::R2lTensor;
use crate::utils::actor_wrapper::ActorWrapper;
use crate::utils::buffer_wrapper::BufferWrapper;

macro_rules! break_on_hook_res {
    ($hook_res:expr) => {
        if $hook_res {
            break;
        }
    };
}

pub trait Agent {
    type Tensor: R2lTensor;

    type Actor: Actor<Tensor = Self::Tensor>;

    fn actor(&self) -> Self::Actor;

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> Result<()>;

    fn shutdown(&mut self) {}
}

pub trait Sampler {
    type Tensor: R2lTensor;
    type TrajectoryContainer: TrajectoryContainer<Tensor = Self::Tensor>;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(
        &mut self,
        actor: A,
    ) -> impl AsRef<[Self::TrajectoryContainer]>;

    fn shutdown(&mut self) {}
}

pub trait OnPolicyAlgorithmHooks {
    type A: Agent;
    type S: Sampler;

    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(&mut self, rollouts: &[<Self::S as Sampler>::TrajectoryContainer])
        -> bool;

    fn post_training_hook(&mut self, actor: <Self::A as Agent>::Actor) -> bool;

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
            let actor = self.agent.actor();
            let actor = ActorWrapper::new(actor);
            let buffers = self.sampler.collect_rollouts(actor);
            break_on_hook_res!(self.hooks.post_rollout_hook(buffers.as_ref()));

            let buffers = buffers
                .as_ref()
                .iter()
                .map(|b| BufferWrapper::new(b))
                .collect::<Vec<_>>();
            self.agent.learn(&buffers)?;
            let actor = self.agent.actor();
            break_on_hook_res!(self.hooks.post_training_hook(actor));
        }

        self.hooks.shutdown_hook(&mut self.agent, &mut self.sampler)
    }
}
