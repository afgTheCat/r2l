use r2l_core::{
    buffers::buffer::TrajectoryBuffer, running_mean::RunningMeanStdF32, tensor::R2lTensor,
};

pub fn mean(numbers: &[f32]) -> f32 {
    let sum: f32 = numbers.iter().sum();
    sum / numbers.len() as f32
}

pub fn fmt_stat(x: f32) -> String {
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
pub struct RewardNormalizer {
    reward_accumulator: Vec<f32>,
    return_rms: RunningMeanStdF32,
    gamma: f32,
    clip_reward: f32,
}

impl RewardNormalizer {
    /// Creates a reward normalizer with the given discount and clipping bounds.
    pub fn new(n_envs: usize, gamma: f32, clip_reward: f32) -> Self {
        Self {
            reward_accumulator: vec![0.0; n_envs],
            return_rms: RunningMeanStdF32::new(),
            gamma,
            clip_reward,
        }
    }

    pub(crate) fn normalize<T: R2lTensor>(&mut self, buffers: &mut [TrajectoryBuffer<T>]) {
        let n_steps = buffers[0].len();
        for step in 0..n_steps {
            for (discounted_return, buffer) in
                self.reward_accumulator.iter_mut().zip(buffers.iter())
            {
                *discounted_return = *discounted_return * self.gamma + buffer.rewards()[step];
            }
            self.return_rms.update(&self.reward_accumulator);
            let reward_scale = (self.return_rms.var + EPSILON).sqrt();
            for (env_idx, buffer) in buffers.iter_mut().enumerate() {
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

    /// Clears environment-local discounted returns while preserving running statistics.
    pub fn reset_returns(&mut self) {
        self.reward_accumulator.fill(0.0);
    }
}
