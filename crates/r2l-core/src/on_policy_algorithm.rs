use crate::sampler::PolicyWrapper;
use crate::sampler::buffer::wrapper::BufferWrapper;
use crate::{
    agents::Agent,
    distributions::Policy,
    env::Env,
    sampler::{Sampler5, buffer::TrajectoryContainer},
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

pub struct DefaultOnPolicyAlgorightmsHooks4<E: Env, P: Policy> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _buffer: PhantomData<E>,
    _policy: PhantomData<P>,
}

impl<E: Env, P: Policy> DefaultOnPolicyAlgorightmsHooks4<E, P> {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _buffer: PhantomData,
            _policy: PhantomData,
        }
    }
}

pub trait OnPolicyAlgorithmHooks5 {
    type A: Agent;
    type S: Sampler5;

    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(
        &mut self,
        rollouts: &[<Self::S as Sampler5>::TrajectoryContainer],
    ) -> bool;

    fn post_training_hook(&mut self, policy: <Self::A as Agent>::Policy) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
}

pub struct OnPolicyAlgorithm5<A: Agent, S: Sampler5, H: OnPolicyAlgorithmHooks5<A = A, S = S>> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

pub struct DefaultOnPolicyAlgorightmsHooks5<A: Agent, S: Sampler5> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _phantom: PhantomData<(A, S)>,
}

impl<A: Agent, S: Sampler5> DefaultOnPolicyAlgorightmsHooks5<A, S> {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _phantom: PhantomData,
        }
    }
}

impl<A: Agent, S: Sampler5> OnPolicyAlgorithmHooks5 for DefaultOnPolicyAlgorightmsHooks5<A, S> {
    type A = A;
    type S = S;

    fn init_hook(&mut self) -> bool {
        false
    }

    fn post_rollout_hook(
        &mut self,
        rollouts: &[<Self::S as Sampler5>::TrajectoryContainer],
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

    fn post_training_hook(&mut self, _policy: <Self::A as Agent>::Policy) -> bool {
        false
    }

    fn shutdown_hook(&mut self) -> Result<()> {
        Ok(())
    }
}

impl<
    B: TrajectoryContainer,
    A: Agent,
    S: Sampler5<TrajectoryContainer = B>,
    H: OnPolicyAlgorithmHooks5<A = A, S = S>,
> OnPolicyAlgorithm5<A, S, H>
where
    A::Policy: Clone,
    A::Tensor: From<S::Tensor>,
    A::Tensor: From<B::Tensor>,
    S::Tensor: From<A::Tensor>,
{
    pub fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            let policy = self.agent.policy();
            let policy = PolicyWrapper::new(policy);
            let buffers = self.sampler.collect_rollouts(policy);
            break_on_hook_res!(self.hooks.post_rollout_hook(buffers.as_ref()));

            let buffers = buffers
                .as_ref()
                .iter()
                .map(|b| BufferWrapper::new(b))
                .collect::<Vec<_>>();
            self.agent.learn(&buffers)?;
            let policy = self.agent.policy();
            break_on_hook_res!(self.hooks.post_training_hook(policy));
        }

        self.hooks.shutdown_hook()
    }
}
