// TODO: heavily under construction like everything else it seems
use candle_core::Result;
use r2l_core::{
    on_policy_algorithm::{IntoTrainingHook, LearningSchedule, TrainingHook},
    utils::rollout_buffer::RolloutBuffer,
};

#[derive(Debug, Default)]
pub struct LoggerTrainingHook {
    epoch_idx: usize,
}

impl TrainingHook for LoggerTrainingHook {
    fn call_hook(
        &mut self,
        _learning_schedule: &LearningSchedule,
        rollouts: &Vec<RolloutBuffer>,
    ) -> Result<bool> {
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
            "epoch: {:<3} episodes: {:<5} total reward: {:<5.2} avg reward per episode: {:.2}",
            self.epoch_idx,
            episodes,
            total_reward,
            total_reward / episodes as f32
        );
        self.epoch_idx += 1;
        Ok(false)
    }
}

impl IntoTrainingHook<()> for LoggerTrainingHook {
    fn into_boxed(self) -> Box<dyn TrainingHook> {
        Box::new(self)
    }
}
