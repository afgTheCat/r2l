use crate::sampler::ActorWrapper;
use crate::sampler::buffer::wrapper::BufferWrapper;
use crate::{
    agents::Agent,
    sampler::{Sampler, buffer::TrajectoryContainer},
};
use anyhow::Result;
use std::marker::PhantomData;

macro_rules! break_on_hook_res {
    ($hook_res:expr) => {
        if $hook_res {
            break;
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub enum LearningSchedule {
    RolloutBound {
        total_rollouts: usize,
        current_rollout: usize,
    },
    TotalStepBound {
        total_steps: usize,
        current_step: usize,
    },
}

impl LearningSchedule {
    pub fn total_step_bound(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps,
            current_step: 0,
        }
    }
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

pub struct DefaultOnPolicyAlgorightmsHooks<A: Agent, S: Sampler> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _phantom: PhantomData<(A, S)>,
}

impl<A: Agent, S: Sampler> DefaultOnPolicyAlgorightmsHooks<A, S> {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _phantom: PhantomData,
        }
    }
}

impl<A: Agent, S: Sampler> OnPolicyAlgorithmHooks for DefaultOnPolicyAlgorightmsHooks<A, S> {
    type A = A;
    type S = S;

    fn init_hook(&mut self) -> bool {
        false
    }

    fn post_rollout_hook(
        &mut self,
        rollouts: &[<Self::S as Sampler>::TrajectoryContainer],
    ) -> bool {
        self.rollout_idx += 1;
        match &mut self.learning_schedule {
            LearningSchedule::RolloutBound {
                total_rollouts,
                current_rollout,
            } => {
                *current_rollout += 1;
                current_rollout >= total_rollouts
            }
            LearningSchedule::TotalStepBound {
                total_steps,
                current_step,
            } => {
                let rollout_steps: usize = rollouts.iter().map(|e| e.actions().count()).sum();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        }
    }

    fn post_training_hook(&mut self, _policy: <Self::A as Agent>::Actor) -> bool {
        false
    }

    fn shutdown_hook(&mut self, agent: &mut Self::A, sampler: &mut Self::S) -> Result<()> {
        agent.shutdown();
        sampler.shutdown();
        Ok(())
    }
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
