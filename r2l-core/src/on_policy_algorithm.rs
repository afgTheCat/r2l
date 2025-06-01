use crate::Algorithm;
use crate::agents::Agent;
use crate::env::EnvPool;
use candle_core::Result;

#[derive(Debug, Clone, Copy)]
pub struct LearningSchedule {
    pub total_rollouts: usize,
    pub current_rollout: usize,
}

impl LearningSchedule {
    pub fn new(total_rollouts: usize) -> Self {
        Self {
            total_rollouts,
            current_rollout: 0,
        }
    }
}

pub struct OnPolicyAlgorithm<E: EnvPool, A: Agent> {
    pub env_pool: E,
    pub agent: A,
    pub learning_schedule: LearningSchedule,
}

impl<E: EnvPool, A: Agent> OnPolicyAlgorithm<E, A> {
    pub fn new(env_pool: E, agent: A, learning_schedule: LearningSchedule) -> Self {
        Self {
            env_pool,
            agent,
            learning_schedule,
        }
    }
}

impl<E: EnvPool, A: Agent> Algorithm for OnPolicyAlgorithm<E, A> {
    fn train(&mut self) -> Result<()> {
        for epoch_idx in 0..self.learning_schedule.total_rollouts {
            let distr = self.agent.distribution();
            let rollouts = self.env_pool.collect_rollouts(distr)?;
            // TODO: debug logging here in a hook
            // let total_reward: f32 = rollouts[0].rewards.iter().sum();
            // let episodes = rollouts[0].dones.iter().filter(|x| **x).count();
            // println!(
            //     "epoch: {:<3} episodes: {:<5} total reward: {:<5.2} avg reward per episode: {:.2}",
            //     epoch_idx,
            //     episodes,
            //     total_reward,
            //     total_reward / episodes as f32
            // );
            self.agent.learn(rollouts)?;
        }
        Ok(())
    }
}
