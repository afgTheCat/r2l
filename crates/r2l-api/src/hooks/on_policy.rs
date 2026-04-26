use std::marker::PhantomData;

use anyhow::Result;
use r2l_core::{
    buffers::TrajectoryContainer,
    on_policy::algorithm::{Agent, OnPolicyAlgorithmHooks, Sampler},
};

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

    pub fn rollout_bound(total_rollouts: usize) -> Self {
        Self::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }
    }
}

pub struct DefaultOnPolicyAlgorithmHooks<A: Agent, S: Sampler> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _phantom: PhantomData<(A, S)>,
}

impl<A: Agent, S: Sampler> DefaultOnPolicyAlgorithmHooks<A, S> {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _phantom: PhantomData,
        }
    }
}

impl<A: Agent, S: Sampler> OnPolicyAlgorithmHooks for DefaultOnPolicyAlgorithmHooks<A, S> {
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
