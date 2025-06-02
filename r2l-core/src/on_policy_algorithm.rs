use crate::Algorithm;
use crate::agents::Agent;
use crate::env::EnvPool;
use crate::utils::rollout_buffer::RolloutBuffer;
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

trait BeforeTrainingHook {
    fn call_hook(&mut self) -> Result<bool>;
}

#[allow(clippy::ptr_arg)]
trait TrainingHook {
    fn call_hook(&mut self, epoch_idx: usize, rollouts: &Vec<RolloutBuffer>) -> Result<bool>;
}

trait AfterTrainingHook {
    fn call_hook(&mut self) -> Result<()>;
}

#[derive(Default)]
pub struct OnPolicyHooks {
    before_training_hook: Option<Box<dyn BeforeTrainingHook>>,
    training_hook: Option<Box<dyn TrainingHook>>,
    after_training_hook: Option<Box<dyn AfterTrainingHook>>,
}

impl OnPolicyHooks {
    fn call_before_training_hook(&mut self) -> Result<bool> {
        if let Some(hook) = &mut self.before_training_hook {
            hook.call_hook()
        } else {
            Ok(false)
        }
    }

    fn call_training_hook(
        &mut self,
        epoch_idx: usize,
        rollouts: &Vec<RolloutBuffer>,
    ) -> Result<bool> {
        if let Some(hook) = &mut self.training_hook {
            hook.call_hook(epoch_idx, rollouts)
        } else {
            Ok(false)
        }
    }

    fn call_after_training_hook(&mut self) -> Result<()> {
        if let Some(hook) = &mut self.after_training_hook {
            hook.call_hook()
        } else {
            Ok(())
        }
    }
}

pub struct OnPolicyAlgorithm<E: EnvPool, A: Agent> {
    pub env_pool: E,
    pub agent: A,
    pub learning_schedule: LearningSchedule,
    pub hooks: OnPolicyHooks,
}

impl<E: EnvPool, A: Agent> OnPolicyAlgorithm<E, A> {
    pub fn new(env_pool: E, agent: A, learning_schedule: LearningSchedule) -> Self {
        Self {
            env_pool,
            agent,
            learning_schedule,
            hooks: OnPolicyHooks::default(),
        }
    }
}

impl<E: EnvPool, A: Agent> Algorithm for OnPolicyAlgorithm<E, A> {
    fn train(&mut self) -> Result<()> {
        if self.hooks.call_before_training_hook()? {
            return Ok(());
        }
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
            if self.hooks.call_training_hook(epoch_idx, &rollouts)? {
                break;
            }
            self.agent.learn(rollouts)?;
        }
        self.hooks.call_after_training_hook()
    }
}
