use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use r2l_core::{
    rng::RNG,
    utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer},
};
use rand::seq::SliceRandom;
pub struct BurnRolloutBuffer<B: Backend>(pub RolloutBuffer<Tensor<B, 1>>);

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
    pub observations: Tensor<B, 2>,
    pub actions: Tensor<B, 2>,
    pub returns: Tensor<B, 2>,
    pub advantages: Tensor<B, 2>,
    pub logp_old: Tensor<B, 2>,
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
        let (states, actions, advantages, returns, logps) = batch_indicies.iter().fold(
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
        let states = Tensor::stack(states, 0);
        let actions = Tensor::stack(actions, 0);
        let ret_len = returns.len();
        let returns = Tensor::from_data(TensorData::new(returns, vec![ret_len]), &device);
        let adv = advantages.len();
        let advantages = Tensor::from_data(TensorData::new(advantages, vec![adv]), &device);
        let logp_len = logps.len();
        let logp_old = Tensor::from_data(TensorData::new(logps, vec![logp_len]), &device);
        Some(RolloutBatch {
            observations: states,
            actions,
            returns,
            advantages,
            logp_old,
        })
    }
}
