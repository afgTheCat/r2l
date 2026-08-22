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
/// Vanilla Policy Gradient implementation.
pub mod vpg;

use derive_more::Deref;
use r2l_core::{
    buffers::TrajectoryBatch,
    error::{Result, TensorError},
    models::{Policy, ValueFunction},
    rng::with_rng,
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
    #[must_use]
    pub fn sample(&self, indices: &[(usize, usize)]) -> Vec<f32> {
        indices
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
    }

    /// Normalizes advantages across all rollout buffers to zero mean and unit variance.
    pub fn normalize(&mut self) {
        let len = self.0.iter().map(Vec::len).sum::<usize>();
        let mean = self.0.iter().flatten().sum::<f32>() / len as f32;
        let variance = self
            .0
            .iter()
            .flatten()
            .map(|x| (*x - mean).powi(2))
            .sum::<f32>()
            / len as f32;
        let std = variance.sqrt() + 1e-8;
        for advantages in &mut self.0 {
            for x in advantages.iter_mut() {
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
    #[must_use]
    pub fn sample(&self, indices: &[(usize, usize)]) -> Vec<f32> {
        indices
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
    #[must_use]
    pub fn sample(&self, indices: &[(usize, usize)]) -> Vec<f32> {
        indices
            .iter()
            .map(|(buff_idx, idx)| self.0[*buff_idx][*idx])
            .collect()
    }
}

fn batch_advantages_and_returns<
    T1: R2lTensor,
    T2: R2lTensor,
    B: TrajectoryBatch<T1>,
    L: Fn(&T1) -> T2,
>(
    batch: &B,
    value_func: &impl ValueFunction<Tensor = T2>,
    gamma: f32,
    lambda: f32,
    lifter: L,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut states = batch.states().iter().map(&lifter).collect::<Vec<_>>();
    let next_state = batch
        .next_states()
        .last()
        .ok_or_else(|| TensorError::EmptyInput {
            operation: "compute advantages and returns".into(),
        })?;
    states.push(lifter(next_state));
    let values_stacked = value_func.values(&states)?;
    let values: Vec<f32> = values_stacked.to_vec()?;
    let total_steps = batch.rewards().len();
    let mut advantages: Vec<f32> = vec![0.; total_steps];
    let mut returns: Vec<f32> = vec![0.; total_steps];
    let mut last_gae_lam: f32 = 0.;

    for i in (0..total_steps).rev() {
        let done = batch.terminated()[i] || batch.truncated()[i];
        let next_non_terminal = if done {
            last_gae_lam = 0.;
            0f32
        } else {
            1.
        };
        let delta = batch.rewards()[i] + next_non_terminal * gamma * values[i + 1] - values[i];
        last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
        advantages[i] = last_gae_lam;
        returns[i] = last_gae_lam + values[i];
    }
    Ok((advantages, returns))
}

/// Computes generalized advantage estimates and return targets for each batch.
///
/// # Errors
///
/// Returns an error if value inference fails.
pub fn batches_advantages_and_returns<
    T1: R2lTensor,
    T2: R2lTensor,
    B: TrajectoryBatch<T1>,
    L: Fn(&T1) -> T2,
>(
    batches: &[B],
    value_func: &impl ValueFunction<Tensor = T2>,
    gamma: f32,
    lambda: f32,
    lifter: L,
) -> Result<(Advantages, Returns)> {
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];
    for batch in batches {
        let (advantages, returns) =
            batch_advantages_and_returns(batch, value_func, gamma, lambda, &lifter)?;
        advantage_vec.push(advantages);
        returns_vec.push(returns);
    }
    Ok((Advantages(advantage_vec), Returns(returns_vec)))
}

/// Samples and converts observations and actions at the supplied batch indices.
pub fn sample<T1: R2lTensor, T2: R2lTensor, B: TrajectoryBatch<T1>, L: Fn(&T1) -> T2>(
    batches: &[B],
    indices: &[(usize, usize)],
    lifter: L,
) -> (Vec<T2>, Vec<T2>) {
    let mut observations = vec![];
    let mut actions = vec![];
    for (batch_idx, idx) in indices {
        observations.push(lifter(&batches[*batch_idx].states()[*idx]));
        actions.push(lifter(&batches[*batch_idx].actions()[*idx]));
    }
    (observations, actions)
}

/// Computes action log-probabilities for every transition in each batch.
///
/// # Errors
///
/// Returns an error if policy evaluation fails.
pub fn logps<T: R2lTensor, B: TrajectoryBatch<T>>(
    batches: &[B],
    policy: &impl Policy<Tensor = T>,
) -> Result<Logps> {
    let mut logps = vec![];
    for batch in batches {
        let logp = policy
            .log_probs(batch.states(), batch.actions())?
            .to_vec()?;
        logps.push(logp);
    }
    Ok(Logps(logps))
}

/// Shuffled minibatch-index cursor spanning multiple trajectory batches.
pub struct ShuffledBatchIndices {
    indices: Vec<(usize, usize)>,
    sample_size: usize,
    current: usize,
}

impl ShuffledBatchIndices {
    /// Creates a shuffled index cursor whose chunks contain at most `sample_size` items.
    pub fn new<T: R2lTensor, B: TrajectoryBatch<T>>(batches: &[B], sample_size: usize) -> Self {
        let mut indices = (0..batches.len())
            .flat_map(|i| {
                let batch = &batches[i];
                (0..batch.len()).map(|j| (i, j)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        with_rng(|rng| indices.shuffle(rng));
        Self {
            indices,
            sample_size,
            current: 0,
        }
    }

    /// Returns the next minibatch of `(batch_index, step_index)` pairs.
    pub fn next_batch(&mut self) -> Option<Vec<(usize, usize)>> {
        let total_size = self.indices.len();
        if self.current >= total_size {
            return None;
        }
        let batch_end = (self.current + self.sample_size).min(total_size);
        let batch_indices = &self.indices[self.current..batch_end];
        self.current = batch_end;
        Some(batch_indices.to_owned())
    }
}
