use candle_core::{Device, Result};
use r2l_core::{
    distributions::Policy,
    env::Env,
    sampler::trajectory_buffers::variable_size_buffer::VariableSizedTrajectoryBuffer,
    // sampler2::{Preprocessor, env_pools::builder::BufferKind},
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

impl<E: Env> Evaluator<E> {
    pub fn new(
        env: E,
        eval_episodes: usize,
        eval_freq: usize,
        eval_step: usize,
        evaluations_results: Arc<Mutex<Vec<Vec<f32>>>>,
        device: Device,
    ) -> Self {
        Self {
            trajectory_buffer: VariableSizedTrajectoryBuffer::new(env),
            eval_episodes,
            eval_freq,
            eval_step,
            evaluations_results,
            device,
        }
    }

    pub fn eval_res(&self) -> Arc<Mutex<Vec<Vec<f32>>>> {
        self.evaluations_results.clone()
    }

    pub fn evaluate(&mut self, dist: &dyn Policy<Tensor = E::Tensor>, n_envs: usize) -> Result<()> {
        if self.eval_step < self.eval_freq {
            self.eval_step += n_envs;
            Ok(())
        } else {
            self.trajectory_buffer
                .run_episodes(dist, self.eval_episodes);
            let rb = self.trajectory_buffer.take_rollout_buffer();
            let sum_rewads = rb.rewards.iter().sum::<f32>();
            let avg_rewards = sum_rewads / self.eval_episodes as f32;
            println!("Avg rew: {}", avg_rewards);

            // TODO: if I recall correctly, this is pretty much whta sb3 does. Maybe we can have a
            // batter solution?
            let mut evaluation_results = vec![];
            let mut current_res = 0f32;
            for (rew, done) in rb.rewards.iter().zip(rb.dones.iter()) {
                current_res += *rew;
                if *done {
                    evaluation_results.push(current_res);
                    current_res = 0.;
                }
            }

            let mut eval_results = self.evaluations_results.lock().unwrap();
            eval_results.push(evaluation_results);
            self.eval_step = 0;
            Ok(())
        }
    }
}

// impl<E: Env> Preprocessor<E, BufferKind<E>> for Evaluator<E> {
//     fn preprocess_states(
//         &mut self,
//         policy: &dyn Policy<Tensor = <E as Env>::Tensor>,
//         buffers: &mut Vec<BufferKind<E>>,
//     ) {
//         // TODO: evalution should not happen here
//     }
// }
