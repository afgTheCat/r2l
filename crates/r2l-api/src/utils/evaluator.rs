use std::sync::{Arc, Mutex};

use candle_core::{Device, Result};
use r2l_core::{
    buffers::variable_sized::VariableSizedStateBuffer,
    env::{Env, Snapshot},
    models::Actor,
    rng::RNG,
};
use rand::RngExt;

pub struct Evaluator<E: Env> {
    pub env: E,
    pub trajectory_buffer: VariableSizedStateBuffer<E::Tensor>,
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
    pub evaluations_results: Arc<Mutex<Vec<Vec<f32>>>>,
    pub device: Device,
}

fn run_episode<E: Env>(
    env: &mut E,
    dist: &dyn Actor<Tensor = E::Tensor>,
) -> Vec<Snapshot<E::Tensor>> {
    let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
    let mut state = env.reset(seed).unwrap();
    let mut snapshots = vec![];
    loop {
        let action = dist.action(state.clone()).unwrap();
        let snapshot = env.step(action).unwrap();
        let should_stop = snapshot.terminated || snapshot.truncated;
        state = snapshot.state.clone();
        snapshots.push(snapshot);
        if should_stop {
            break;
        }
    }
    snapshots
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
            env,
            trajectory_buffer: VariableSizedStateBuffer::new(),
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

    pub fn evaluate(&mut self, dist: &dyn Actor<Tensor = E::Tensor>, n_envs: usize) -> Result<()> {
        if self.eval_step < self.eval_freq {
            self.eval_step += n_envs;
            Ok(())
        } else {
            let snapshots: Vec<_> = (0..self.eval_episodes)
                .flat_map(|_| run_episode(&mut self.env, dist))
                .collect();
            let rewards = snapshots.iter().map(|s| s.reward).collect::<Vec<_>>();
            let dones = snapshots
                .iter()
                .map(|s| s.terminated || s.truncated)
                .collect::<Vec<_>>();
            let sum_rewads = rewards.iter().sum::<f32>();
            let avg_rewards = sum_rewads / self.eval_episodes as f32;
            println!("Avg rew: {}", avg_rewards);
            // TODO: if I recall correctly, this is pretty much whta sb3 does. Maybe we can have a
            // batter solution?
            let mut evaluation_results = vec![];
            let mut current_res = 0f32;
            for (rew, done) in rewards.iter().zip(dones.iter()) {
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
