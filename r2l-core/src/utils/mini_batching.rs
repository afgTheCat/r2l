use super::rollout_buffer::{RolloutBatch, RolloutBuffer};
use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;

pub struct RolloutBufferIterator<'a> {
    rollouts: &'a [RolloutBuffer],
    indicies: Vec<(usize, usize)>,
    current: usize,
    sample_size: usize,
    device: Device,
}

impl<'a> RolloutBufferIterator<'a> {
    pub fn new(
        rollouts: &'a [RolloutBuffer],
        indicies: Vec<(usize, usize)>,
        sample_size: usize,
        current: usize,
        device: Device,
    ) -> Self {
        Self {
            rollouts,
            indicies,
            sample_size,
            current,
            device,
        }
    }
}

impl<'a> Iterator for RolloutBufferIterator<'a> {
    type Item = RolloutBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let total_episodes = self.indicies.len() - 1;
        if self.current + self.sample_size < total_episodes {
            let batch_indicies = &self.indicies[self.current..self.current + self.sample_size];
            self.current += self.sample_size;
            let (states, actions, advantages, returns, logps) = batch_indicies.iter().fold(
                (vec![], vec![], vec![], vec![], vec![]),
                |(mut states, mut actions, mut advantages, mut returns, mut logps),
                 (rollout_idx, idx)| {
                    let (state, action, adv, ret, logp) =
                        self.rollouts[*rollout_idx].sample_point(*idx);
                    states.push(state);
                    actions.push(action);
                    advantages.push(adv.unwrap());
                    returns.push(ret.unwrap());
                    logps.push(logp);
                    (states, actions, advantages, returns, logps)
                },
            );
            let states = Tensor::stack(&states, 0).ok()?;
            let actions = Tensor::stack(&actions, 0).ok()?;
            let returns = Tensor::from_slice(&returns, returns.len(), &self.device).ok()?;
            let advantages =
                Tensor::from_slice(&advantages, advantages.len(), &self.device).ok()?;
            let logp_old = Tensor::from_slice(&logps, logps.len(), &self.device).ok()?;
            Some(RolloutBatch {
                observations: states,
                actions,
                returns,
                advantages,
                logp_old,
            })
        } else {
            None
        }
    }
}

pub fn create_rollout_buffer_iterator(
    rollouts: &[RolloutBuffer],
    sample_size: usize,
    device: Device,
) -> RolloutBufferIterator {
    let mut indicies = (0..rollouts.len())
        .flat_map(|i| {
            let rb = &rollouts[i];
            (0..rb.rewards.len()).map(|j| (i, j)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    indicies.shuffle(&mut rand::rng());
    RolloutBufferIterator::new(rollouts, indicies, sample_size, 0, device)
}
