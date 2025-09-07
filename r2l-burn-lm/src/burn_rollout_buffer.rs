use anyhow::Result;
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use r2l_core::{
    policies::ValueFunction,
    rng::RNG,
    utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer},
};
use rand::seq::SliceRandom;

pub struct BurnRolloutBuffer<B: Backend>(pub RolloutBuffer<Tensor<B, 1>>);

impl<B: Backend> BurnRolloutBuffer<B> {
    pub fn new(rollout_buffer: RolloutBuffer<Tensor<B, 1>>) -> Self {
        Self(rollout_buffer)
    }

    pub fn calculate_advantages_and_returns(
        &self,
        value_func: &impl ValueFunction<Tensor = Tensor<B, 1>>,
        gamma: f32,
        lambda: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let values = value_func.calculate_values(&self.0.states)?;
        let values: Vec<f32> = values.to_data().to_vec().unwrap();
        let total_steps = self.0.rewards.len();
        let mut advantages: Vec<f32> = vec![0.; total_steps];
        let mut returns: Vec<f32> = vec![0.; total_steps];
        let mut last_gae_lam: f32 = 0.;
        for i in (0..total_steps).rev() {
            let next_non_terminal = if self.0.dones[i] {
                last_gae_lam = 0.;
                0f32
            } else {
                1.
            };
            let delta = self.0.rewards[i] + next_non_terminal * gamma * values[i + 1] - values[i];
            last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
            advantages[i] = last_gae_lam;
            returns[i] = last_gae_lam + values[i];
        }
        Ok((advantages, returns))
    }
}

impl<B: Backend> From<RolloutBuffer<Tensor<B, 1>>> for BurnRolloutBuffer<B> {
    fn from(value: RolloutBuffer<Tensor<B, 1>>) -> Self {
        Self(value)
    }
}

pub struct RolloutBatchIterator<'a, B: Backend> {
    rollouts: &'a [BurnRolloutBuffer<B>],
    advantages: &'a Advantages,
    returns: &'a Returns,
    logps: &'a Logps,
    indicies: Vec<(usize, usize)>,
    current: usize,
    sample_size: usize,
}

impl<'a, B: Backend> RolloutBatchIterator<'a, B> {
    pub fn new(
        rollouts: &'a [BurnRolloutBuffer<B>],
        advantages: &'a Advantages,
        returns: &'a Returns,
        logps: &'a Logps,
        sample_size: usize,
    ) -> Self {
        let mut indicies = (0..rollouts.len())
            .flat_map(|i| {
                let rb = &rollouts[i];
                (0..rb.0.rewards.len()).map(|j| (i, j)).collect::<Vec<_>>()
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
        }
    }
}

pub struct RolloutBatch<B: Backend> {
    pub observations: Vec<Tensor<B, 1>>,
    pub actions: Vec<Tensor<B, 1>>,
    pub returns: Tensor<B, 1>,
    pub advantages: Tensor<B, 1>,
    pub logp_old: Tensor<B, 1>,
}

pub fn calculate_advantages_and_returns<B: Backend>(
    rollouts: &[BurnRolloutBuffer<B>],
    value_func: &impl ValueFunction<Tensor = Tensor<B, 1>>,
    gamma: f32,
    lambda: f32,
) -> (Advantages, Returns) {
    let (advantages, returns): (Vec<Vec<f32>>, Vec<Vec<f32>>) = rollouts
        .iter()
        .map(|rollout| {
            rollout
                .calculate_advantages_and_returns(value_func, gamma, lambda)
                .unwrap() // TODO: get rid of this unwrap
        })
        .unzip();
    (Advantages(advantages), Returns(returns))
}

impl<'a, B: Backend> Iterator for RolloutBatchIterator<'a, B> {
    type Item = RolloutBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        let device = Default::default();
        let total_episodes = self.indicies.len() - 1;
        if self.current + self.sample_size >= total_episodes {
            return None;
        }
        let batch_indicies = &self.indicies[self.current..self.current + self.sample_size];
        self.current += self.sample_size;
        let (observations, actions, advantages, returns, logps) = batch_indicies.iter().fold(
            (vec![], vec![], vec![], vec![], vec![]),
            |(mut states, mut actions, mut advantages, mut returns, mut logps),
             (rollout_idx, idx)| {
                let (state, action) = self.rollouts[*rollout_idx].0.sample_point(*idx);
                let adv = self.advantages[*rollout_idx][*idx];
                let ret = self.returns[*rollout_idx][*idx];
                let logp = self.logps[*rollout_idx][*idx];
                states.push(state.clone());
                actions.push(action.clone());
                advantages.push(adv);
                returns.push(ret);
                logps.push(logp);
                (states, actions, advantages, returns, logps)
            },
        );
        let ret_len = returns.len();
        let returns = Tensor::from_data(TensorData::new(returns, vec![ret_len]), &device);
        let adv = advantages.len();
        let advantages = Tensor::from_data(TensorData::new(advantages, vec![adv]), &device);
        let logp_len = logps.len();
        let logp_old = Tensor::from_data(TensorData::new(logps, vec![logp_len]), &device);
        Some(RolloutBatch {
            observations,
            actions,
            returns,
            advantages,
            logp_old,
        })
    }
}
