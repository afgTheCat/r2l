use candle_core::{Device, Result, Tensor};
use r2l_core::{
    distributions::Distribution, env::Env, numeric::Buffer,
    sampler::trajectory_buffers::variable_size_buffer::VariableSizedTrajectoryBuffer,
};
use std::sync::{Arc, Mutex};

pub struct Evaluator<E: Env> {
    pub trajectory_buffer: VariableSizedTrajectoryBuffer<E>,
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
    pub evaluations_results: Arc<Mutex<Vec<Vec<f32>>>>,
    pub device: Device,
}

impl<E: Env<Tensor = Buffer>> Evaluator<E> {
    pub fn new(
        env: E,
        eval_episodes: usize,
        eval_freq: usize,
        eval_step: usize,
        evaluations_results: Arc<Mutex<Vec<Vec<f32>>>>,
    ) -> Self {
        Self {
            trajectory_buffer: VariableSizedTrajectoryBuffer::new(env),
            eval_episodes,
            eval_freq,
            eval_step,
            evaluations_results,
            device: Device::Cpu,
        }
    }

    pub fn eval_res(&self) -> Arc<Mutex<Vec<Vec<f32>>>> {
        self.evaluations_results.clone()
    }

    pub fn evaluate(
        &mut self,
        dist: &dyn Distribution<Tensor = Tensor>,
        n_envs: usize,
    ) -> Result<()> {
        if self.eval_step < self.eval_freq {
            self.eval_step += n_envs;
            Ok(())
        } else {
            // TODO: VariableSizedTrajectoryBuf
            let mut all_rewards = vec![];
            let mut dones = vec![];
            for _ in 0..self.eval_episodes {
                // TODO: variable sized buffer should be able to handle single episodes
                self.trajectory_buffer.run_episode(dist, 1);
                let rollout_buffer = self.trajectory_buffer.to_rollout_buffer();
                all_rewards.push(rollout_buffer.rewards.clone());
                dones.push(rollout_buffer.dones.clone());
            }
            let eps = dones
                .iter()
                .map(|x| x.iter().filter(|y| **y).count())
                .sum::<usize>();
            let sum_rewards_per_eps: Vec<f32> =
                all_rewards.iter().map(|x| x.iter().sum::<f32>()).collect();
            let avg_rewards = sum_rewards_per_eps.iter().sum::<f32>() / eps as f32;
            println!("Avg rew: {}", avg_rewards);
            let mut eval_results = self.evaluations_results.lock().unwrap();
            eval_results.push(sum_rewards_per_eps);
            self.eval_step = 0;
            Ok(())
        }
    }
}
