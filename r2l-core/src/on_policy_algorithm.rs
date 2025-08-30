use crate::{
    Algorithm,
    agents::{Agent, TensorOfAgent},
    env::{Sampler, TensorOfSampler},
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Result;

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
            total_steps: total_steps,
            current_step: 0,
        }
    }
}

pub trait OnPolicyAlgorithmHooks<T: Clone> {
    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(&mut self, rollouts: &mut [RolloutBuffer<T>]) -> bool;

    fn post_training_hook(&mut self) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
}

pub struct DefaultOnPolicyAlgorightmsHooks {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
}

impl DefaultOnPolicyAlgorightmsHooks {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
        }
    }
}

impl<T: Clone> OnPolicyAlgorithmHooks<T> for DefaultOnPolicyAlgorightmsHooks {
    fn init_hook(&mut self) -> bool {
        false
    }

    fn post_rollout_hook(&mut self, rollouts: &mut [RolloutBuffer<T>]) -> bool {
        let total_reward = rollouts
            .iter()
            .map(|s| s.rewards.iter().sum::<f32>())
            .sum::<f32>();
        let episodes: usize = rollouts
            .iter()
            .flat_map(|s| &s.dones)
            .filter(|d| **d)
            .count();
        println!(
            "rollout: {:<3} episodes: {:<5} total reward: {:<5.2} avg reward per episode: {:.2}",
            self.rollout_idx,
            episodes,
            total_reward,
            total_reward / episodes as f32
        );
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
                let rollout_steps: usize = rollouts.iter().map(|e| e.actions.len()).sum();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        }
    }

    fn post_training_hook(&mut self) -> bool {
        false
    }

    fn shutdown_hook(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct OnPolicyAlgorithm<S: Sampler, A: Agent, H: OnPolicyAlgorithmHooks<TensorOfAgent<A>>> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

impl<S: Sampler, A: Agent, H: OnPolicyAlgorithmHooks<TensorOfAgent<A>>> Algorithm
    for OnPolicyAlgorithm<S, A, H>
where
    TensorOfSampler<S>: From<TensorOfAgent<A>>,
    TensorOfSampler<S>: Into<TensorOfAgent<A>>,
{
    fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            // rollout phase
            let distribution = self.agent.distribution();
            let mut rollouts = self.sampler.collect_rollouts(distribution)?;
            break_on_hook_res!(self.hooks.post_rollout_hook(&mut rollouts));

            // learning phase
            self.agent.learn(rollouts)?;
            break_on_hook_res!(self.hooks.post_training_hook());
        }
        self.hooks.shutdown_hook()
    }
}
