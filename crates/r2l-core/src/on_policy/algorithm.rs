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

    /// Releases sampler resources before the training loop exits.
    fn shutdown(&mut self) {}
}

/// Lifecycle hooks for [`OnPolicyAlgorithm`].
///
/// Hook methods return `true` to stop the training loop at that point.
pub trait OnPolicyAlgorithmHooks {
    /// Agent type controlled by the training loop.
    type A: Agent;
    /// Sampler type controlled by the training loop.
    type S: Sampler;

    /// Called once before rollout/training starts.
    fn init_hook(&mut self) -> bool;

    /// Called after rollouts are collected and before agent learning.
    fn post_rollout_hook(&mut self, rollouts: &[<Self::S as Sampler>::TrajectoryContainer])
    -> bool;

    /// Called after the agent has learned from the latest rollouts.
    fn post_training_hook(&mut self, actor: <Self::A as Agent>::Actor) -> bool;

    /// Called once when the loop exits.
    fn shutdown_hook(&mut self, agent: &mut Self::A, sampler: &mut Self::S) -> Result<()>;
}

/// Default on-policy training loop over an [`Agent`], [`Sampler`], and hooks.
pub struct OnPolicyAlgorithm<A: Agent, S: Sampler, H: OnPolicyAlgorithmHooks<A = A, S = S>> {
    /// Rollout collector.
    pub sampler: S,
    /// Trainable agent.
    pub agent: A,
    /// Lifecycle hooks.
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
    /// Runs rollout collection and learning until a hook requests shutdown.
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
