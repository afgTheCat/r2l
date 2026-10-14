use std::num::NonZeroUsize;

use r2l_core::{
    buffers::buffer::TrajectoryBuffer, running_mean::RunningMeanStdF32, tensor::R2lTensor,
};

pub(crate) fn mean(numbers: impl ExactSizeIterator<Item = f32>) -> f32 {
    let count = numbers.len();
    numbers.sum::<f32>() / count as f32
}

pub(crate) fn fmt_stat(x: f32) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x.abs() < 0.001 {
        format!("{x:.2e}")
    } else {
        format!("{x:.4}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

const EPSILON: f32 = 1e-8;

/// Normalizes rewards using the running variance of discounted returns.
#[derive(Clone)]
pub(crate) struct RewardNormalizer {
    reward_accumulator: Vec<f32>,
    return_rms: RunningMeanStdF32,
    gamma: f32,
    clip_reward: f32,
}

impl RewardNormalizer {
    /// Creates a reward normalizer with the given discount and clipping bounds.
    ///
    /// # Arguments
    ///
    /// * `n_envs` - Number of independent environment return streams to track.
    /// * `gamma` - Discount factor used to accumulate returns.
    /// * `clip_reward` - Absolute limit applied to normalized rewards.
    pub(crate) fn new(n_envs: NonZeroUsize, gamma: f32, clip_reward: f32) -> Self {
        Self {
            reward_accumulator: vec![0.0; n_envs.get()],
            return_rms: RunningMeanStdF32::new(),
            gamma,
            clip_reward,
        }
    }

    /// Normalizes rewards in step order, updating statistics only from active environments.
    /// Buffers must remain in environment order across calls; their lengths may differ.
    pub(crate) fn normalize<T: R2lTensor>(&mut self, buffers: &mut [TrajectoryBuffer<T>]) {
        let n_steps = buffers.iter().map(TrajectoryBuffer::len).max().unwrap_or(0);
        let mut active_returns = Vec::with_capacity(buffers.len());
        for step in 0..n_steps {
            active_returns.clear();
            for (discounted_return, buffer) in
                self.reward_accumulator.iter_mut().zip(buffers.iter())
            {
                if let Some(&reward) = buffer.rewards().get(step) {
                    *discounted_return = *discounted_return * self.gamma + reward;
                    active_returns.push(*discounted_return);
                }
            }
            self.return_rms.update(&active_returns);
            let reward_scale = (self.return_rms.var + EPSILON).sqrt();
            for (env_idx, buffer) in buffers.iter_mut().enumerate() {
                if step >= buffer.len() {
                    continue;
                }
                let done = buffer.terminated()[step] || buffer.truncated()[step];
                let reward = buffer.rewards()[step];
                buffer.rewards_mut()[step] =
                    (reward / reward_scale).clamp(-self.clip_reward, self.clip_reward);
                if done {
                    self.reward_accumulator[env_idx] = 0.0;
                }
            }
        }
    }

    pub(crate) fn statistics(&self) -> RunningMeanStdF32 {
        self.return_rms.clone()
    }

    pub(crate) fn restore_statistics(&mut self, statistics: RunningMeanStdF32) {
        self.return_rms = statistics;
        self.reset_returns();
    }

    /// Clears environment-local discounted returns while preserving running statistics.
    pub(crate) fn reset_returns(&mut self) {
        self.reward_accumulator.fill(0.0);
    }
}
