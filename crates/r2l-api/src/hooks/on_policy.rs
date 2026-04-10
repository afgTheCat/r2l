use anyhow::Result;
use r2l_core::{
    agents::Agent,
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithmHooks},
    sampler::{Sampler, buffer::TrajectoryContainer},
};
use std::marker::PhantomData;

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
