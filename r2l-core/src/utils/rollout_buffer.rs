use crate::{policies::ValueFunction, rng::RNG};
use candle_core::{Device, Result, Tensor};
use derive_more::Deref;
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
pub struct RolloutBuffer<T: Clone> {
    pub states: Vec<T>,
    pub actions: Vec<T>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
}

impl<T: Clone> Default for RolloutBuffer<T> {
    fn default() -> Self {
        Self {
            states: vec![],
            actions: vec![],
            rewards: vec![],
            dones: vec![],
        }
    }
}

impl<T: Clone> RolloutBuffer<T> {
    // TODO: this should be the last state
    pub fn set_last_state(&mut self, state: T) {
        self.states.push(state.clone());
    }

    pub fn sample_point(&self, index: usize) -> (&T, &T) {
        (&self.states[index], &self.actions[index])
    }
}

impl RolloutBuffer<Tensor> {
    // TODO: I don't know if this should be
    pub fn calculate_advantages_and_returns2(
        &self,
        value_func: &impl ValueFunction,
        gamma: f32,
        lambda: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let states = Tensor::stack(&self.states, 0)?;
        let values: Vec<f32> = value_func.calculate_values(&states)?.to_vec1()?;
        let total_steps = self.rewards.len();
        let mut advantages: Vec<f32> = vec![0.; total_steps];
        let mut returns: Vec<f32> = vec![0.; total_steps];
        let mut last_gae_lam: f32 = 0.;
        for i in (0..total_steps).rev() {
            let next_non_terminal = if self.dones[i] {
                last_gae_lam = 0.;
                0f32
            } else {
                1.
            };
            let delta = self.rewards[i] + next_non_terminal * gamma * values[i + 1] - values[i];
            last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
            advantages[i] = last_gae_lam;
            returns[i] = last_gae_lam + values[i];
        }
        Ok((advantages, returns))
    }
}

pub struct RolloutBatch {
    pub observations: Tensor,
    pub actions: Tensor,
    pub returns: Tensor,
    pub advantages: Tensor,
    pub logp_old: Tensor,
}

#[derive(Deref, Debug)]
pub struct Advantages(Vec<Vec<f32>>);

impl Advantages {
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

#[derive(Deref, Debug)]
pub struct Returns(Vec<Vec<f32>>);

pub fn calculate_advantages_and_returns(
    rollouts: &[RolloutBuffer<Tensor>],
    value_func: &impl ValueFunction,
    gamma: f32,
    lambda: f32,
) -> (Advantages, Returns) {
    let (advantages, returns): (Vec<Vec<f32>>, Vec<Vec<f32>>) = rollouts
        .iter()
        .map(|rollout| {
            rollout
                .calculate_advantages_and_returns2(value_func, gamma, lambda)
                .unwrap() // TODO: get rid of this unwrap
        })
        .unzip();
    (Advantages(advantages), Returns(returns))
}

#[derive(Deref, Debug)]
pub struct Logps(pub Vec<Vec<f32>>);

pub struct RolloutBatchIterator<'a> {
    rollouts: &'a [RolloutBuffer<Tensor>],
    advantages: &'a Advantages,
    returns: &'a Returns,
    logps: &'a Logps,
    indicies: Vec<(usize, usize)>,
    current: usize,
    sample_size: usize,
    device: Device,
}

impl<'a> RolloutBatchIterator<'a> {
    pub fn new(
        rollouts: &'a [RolloutBuffer<Tensor>],
        advantages: &'a Advantages,
        returns: &'a Returns,
        logps: &'a Logps,
        sample_size: usize,
        device: Device,
    ) -> Self {
        let mut indicies = (0..rollouts.len())
            .flat_map(|i| {
                let rb = &rollouts[i];
                (0..rb.rewards.len()).map(|j| (i, j)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        RNG.with_borrow_mut(|rng| indicies.shuffle(rng));
        Self {
            rollouts,
            advantages,
            returns,
            logps,
            indicies,
            current: 0,
            sample_size,
            device,
        }
    }
}

impl<'a> Iterator for RolloutBatchIterator<'a> {
    type Item = RolloutBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let total_episodes = self.indicies.len() - 1;
        if self.current + self.sample_size >= total_episodes {
            return None;
        }
        let batch_indicies = &self.indicies[self.current..self.current + self.sample_size];
        self.current += self.sample_size;
        let (states, actions, advantages, returns, logps) = batch_indicies.iter().fold(
            (vec![], vec![], vec![], vec![], vec![]),
            |(mut states, mut actions, mut advantages, mut returns, mut logps),
             (rollout_idx, idx)| {
                let (state, action) = self.rollouts[*rollout_idx].sample_point(*idx);
                let adv = self.advantages[*rollout_idx][*idx];
                let ret = self.returns[*rollout_idx][*idx];
                let logp = self.logps[*rollout_idx][*idx];
                states.push(state);
                actions.push(action);
                advantages.push(adv);
                returns.push(ret);
                logps.push(logp);
                (states, actions, advantages, returns, logps)
            },
        );
        let states = Tensor::stack(&states, 0).ok()?;
        let actions = Tensor::stack(&actions, 0).ok()?;
        let returns = Tensor::from_slice(&returns, returns.len(), &self.device).ok()?;
        let advantages = Tensor::from_slice(&advantages, advantages.len(), &self.device).ok()?;
        let logp_old = Tensor::from_slice(&logps, logps.len(), &self.device).ok()?;
        Some(RolloutBatch {
            observations: states,
            actions,
            returns,
            advantages,
            logp_old,
        })
    }
}
