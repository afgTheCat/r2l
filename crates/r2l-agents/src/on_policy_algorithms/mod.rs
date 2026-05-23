//! Shared utilities and concrete implementations for on-policy algorithms.
//!
//! This module provides common rollout-processing helpers such as generalized
//! advantage estimation together with the concrete
//! [`mod@crate::on_policy_algorithms::a2c`],
//! [`mod@crate::on_policy_algorithms::ppo`], and
//! [`mod@crate::on_policy_algorithms::vpg`] algorithm modules.

/// Advantage values computed per rollout buffer.
pub mod a2c;
/// Proximal Policy Optimization implementation and hook interface.
pub mod ppo;
/// Prototype PPO variant that consumes trajectory batches directly.
pub mod ppo_new;
/// Vanilla Policy Gradient implementation.
pub mod vpg;

use derive_more::Deref;
use r2l_core::{
    buffers::TrajectoryContainer,
    models::{Policy, ValueFunction},
    rng::RNG,
    tensor::R2lTensor,
};
use rand::seq::SliceRandom;

/// Advantage estimates grouped by rollout buffer.
///
/// Each inner vector corresponds to one trajectory container.
#[derive(Deref, Debug)]
pub struct Advantages(pub Vec<Vec<f32>>);

impl Advantages {
    /// Samples advantage values at the provided `(buffer_index, step_index)` pairs.
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        indicies
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
    }

    /// Normalizes each buffer's advantages to zero mean and unit variance.
    pub fn normalize(&mut self) {
        for advantage in self.0.iter_mut() {
            let mean = advantage.iter().sum::<f32>() / advantage.len() as f32;
            let variance =
                advantage.iter().map(|x| (*x - mean).powi(2)).sum::<f32>() / advantage.len() as f32;
            let std = variance.sqrt() + 1e-8;
            for x in advantage.iter_mut() {
                *x = (*x - mean) / std;
            }
        }
    }
}

/// Return targets grouped by rollout buffer.
#[derive(Deref, Debug)]
pub struct Returns(pub Vec<Vec<f32>>);

impl Returns {
    /// Samples return values at the provided `(buffer_index, step_index)` pairs.
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        indicies
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
    }
}

/// Log-probability values grouped by rollout buffer.
#[derive(Deref, Debug)]
pub struct Logps(pub Vec<Vec<f32>>);

impl Logps {
    /// Samples log-probability values at the provided `(buffer_index, step_index)` pairs.
    pub fn sample(&self, indicies: &[(usize, usize)]) -> Vec<f32> {
        indicies
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
    }
}

struct BatchIndexIterator {
    indicies: Vec<(usize, usize)>,
    sample_size: usize,
    current: usize,
}

impl BatchIndexIterator {
    pub fn new<B: TrajectoryContainer>(buffers: &[B], sample_size: usize) -> Self {
        let mut indicies = (0..buffers.len())
            .flat_map(|i| {
                let rb = &buffers[i];
                (0..rb.len()).map(|j| (i, j)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        RNG.with_borrow_mut(|rng| indicies.shuffle(rng));
        Self {
            indicies,
            sample_size,
            current: 0,
        }
    }

    fn iter(&mut self) -> Option<Vec<(usize, usize)>> {
        let total_size = self.indicies.len();
        if self.sample_size + self.current > total_size {
            return None;
        }
        let batch_indicies = &self.indicies[self.current..self.current + self.sample_size];
        self.current += self.sample_size;
        Some(batch_indicies.to_owned())
    }
}

fn logps<T: R2lTensor, B: TrajectoryContainer<Tensor = T>>(
    buffers: &[B],
    policy: &impl Policy<Tensor = T>,
) -> Logps {
    let mut logps = vec![];
    for buffer in buffers {
        let states = buffer.states().cloned().collect::<Vec<_>>();
        let actions = buffer.actions().cloned().collect::<Vec<_>>();
        let logp = policy
            .log_probs(&states, &actions)
            .map(|t| t.to_vec())
            .unwrap();
        logps.push(logp);
    }
    Logps(logps)
}

fn sample<T1: R2lTensor, B: TrajectoryContainer<Tensor = T1>, T2: R2lTensor, L: Fn(&T1) -> T2>(
    buffers: &[B],
    indicies: &[(usize, usize)],
    lifter: L,
) -> (Vec<T2>, Vec<T2>) {
    let mut observations = vec![];
    let mut actions = vec![];
    for (buffer_idx, idx) in indicies {
        let observation = buffers[*buffer_idx].states().nth(*idx).unwrap();
        let action = buffers[*buffer_idx].actions().nth(*idx).unwrap();
        observations.push(lifter(observation));
        actions.push(lifter(action));
    }
    (observations, actions)
}

/// Computes generalized-advantage estimates and returns for one rollout buffer.
pub fn buffer_advantages_and_returns<T1: R2lTensor, T2: R2lTensor, L: Fn(&T1) -> T2>(
    buffer: &impl TrajectoryContainer<Tensor = T1>,
    value_func: &impl ValueFunction<Tensor = T2>,
    gamma: f32,
    lambda: f32,
    lifter: L,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let mut states = buffer.states().map(&lifter).collect::<Vec<_>>();
    states.push(buffer.next_states().last().map(&lifter).unwrap());
    let values_stacked = value_func.values(&states).unwrap();
    let values: Vec<f32> = values_stacked.to_vec();
    let total_steps = buffer.rewards().count();
    let mut advantages: Vec<f32> = vec![0.; total_steps];
    let mut returns: Vec<f32> = vec![0.; total_steps];
    let mut last_gae_lam: f32 = 0.;
    for i in (0..total_steps).rev() {
        let mut dones = buffer
            .terminated()
            .zip(buffer.truncated())
            .map(|(terminated, truncated)| terminated || truncated);
        let next_non_terminal = if dones.nth(i).unwrap() {
            last_gae_lam = 0.;
            0f32
        } else {
            1.
        };
        let delta = buffer.rewards().nth(i).unwrap() + next_non_terminal * gamma * values[i + 1]
            - values[i];
        last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
        advantages[i] = last_gae_lam;
        returns[i] = last_gae_lam + values[i];
    }
    Ok((advantages, returns))
}

/// Computes generalized-advantage estimates and returns for multiple buffers.
pub fn buffers_advantages_and_returns<
    T1: R2lTensor,
    B: TrajectoryContainer<Tensor = T1>,
    T2: R2lTensor,
    L: Fn(&T1) -> T2,
>(
    buffers: &[B],
    value_func: &impl ValueFunction<Tensor = T2>,
    gamma: f32,
    lambda: f32,
    lifter: L,
) -> anyhow::Result<(Advantages, Returns)> {
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];
    for buffer in buffers {
        let (advantages, returns) =
            buffer_advantages_and_returns(buffer, value_func, gamma, lambda, &lifter)?;
        advantage_vec.push(advantages);
        returns_vec.push(returns);
    }
    Ok((Advantages(advantage_vec), Returns(returns_vec)))
}
