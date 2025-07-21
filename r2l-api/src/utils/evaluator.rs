use std::sync::{Arc, Mutex};

use candle_core::{Device, Result, Tensor};
use r2l_core::{
    distributions::Distribution,
    env::{Env, sequential_vec_env::SequentialVecEnvHooks, single_step_env_with_buffer},
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};

pub struct Evaluator<E: Env> {
    pub env: E,
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
    pub evaluations_results: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl<E: Env> Evaluator<E> {
    pub fn new(
        env: E,
        eval_episodes: usize,
        eval_freq: usize,
        eval_step: usize,
        evaluations_results: Arc<Mutex<Vec<Vec<f32>>>>,
    ) -> Self {
        Self {
            env,
            eval_episodes,
            eval_freq,
            eval_step,
            evaluations_results,
        }
    }

    pub fn eval_res(&self) -> Arc<Mutex<Vec<Vec<f32>>>> {
        self.evaluations_results.clone()
    }

    fn evaluate(&mut self, dist: &dyn Distribution, n_envs: usize) -> Result<()> {
        if self.eval_step < self.eval_freq {
            self.eval_step += n_envs;
            Ok(())
        } else {
            let mut all_rewards = vec![];
            let mut dones = vec![];
            for _ in 0..self.eval_episodes {
                let mut state = self.env.reset(rand::random())?;
                let mut rollout_buffer = RolloutBuffer::default();
                while let (next_state, false) =
                    single_step_env_with_buffer(dist, &state, &mut self.env, &mut rollout_buffer)?
                {
                    state = next_state;
                }
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
