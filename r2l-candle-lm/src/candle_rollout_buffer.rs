use bincode::{
    Decode, Encode,
    error::{DecodeError, EncodeError},
};
use candle_core::{Device, Result, Tensor};
use r2l_core::{
    policies::ValueFunction,
    rng::RNG,
    utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer},
};
use rand::seq::SliceRandom;

pub struct CandleRolloutBuffer(pub RolloutBuffer<Tensor>);

impl From<RolloutBuffer<Tensor>> for CandleRolloutBuffer {
    fn from(value: RolloutBuffer<Tensor>) -> Self {
        Self(value)
    }
}

impl Encode for CandleRolloutBuffer {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        let writer_config = bincode::config::standard();
        let states = self
            .0
            .states
            .iter()
            .map(|t| t.to_vec1::<f32>())
            .collect::<Result<Vec<_>>>()
            .map_err(|err| EncodeError::OtherString(err.to_string()))?;
        bincode::encode_into_writer(&states, &mut encoder.writer(), writer_config)?;
        let actions = self
            .0
            .actions
            .iter()
            .map(|t| t.to_vec1::<f32>())
            .collect::<Result<Vec<_>>>()
            .map_err(|err| EncodeError::OtherString(err.to_string()))?;
        bincode::encode_into_writer(&actions, &mut encoder.writer(), writer_config)?;
        bincode::encode_into_writer(&self.0.rewards, &mut encoder.writer(), writer_config)?;
        bincode::encode_into_writer(&self.0.dones, &mut encoder.writer(), writer_config)?;
        Ok(())
    }
}

// TODO: we need to add the device into the context here to be able to use Tensors on the GPU
impl<C> Decode<C> for CandleRolloutBuffer {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        let states_raw: Vec<Vec<f32>> = Vec::decode(decoder)?;
        let states = states_raw
            .into_iter()
            .map(|v| Tensor::from_slice(&v, v.len(), &Device::Cpu))
            .collect::<Result<Vec<_>>>()
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let actions_raw: Vec<Vec<f32>> = Vec::decode(decoder)?;
        let actions = actions_raw
            .into_iter()
            .map(|v| Tensor::from_slice(&v, v.len(), &Device::Cpu))
            .collect::<Result<Vec<_>>>()
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let rewards: Vec<f32> = Vec::decode(decoder)?;
        let dones: Vec<bool> = Vec::decode(decoder)?;
        Ok(Self(RolloutBuffer {
            states,
            actions,
            rewards,
            dones,
        }))
    }
}

// TODO: readd this
// impl<'de, C> BorrowDecode<'de, C> for CandleRolloutBuffer {
//     fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = C>>(
//         decoder: &mut D,
//     ) -> std::result::Result<Self, bincode::error::DecodeError> {
//         RolloutBuffer::decode(decoder)
//     }
// }

impl CandleRolloutBuffer {
    // TODO: I don't know if this should be
    pub fn calculate_advantages_and_returns2(
        &self,
        value_func: &impl ValueFunction<Tensor = Tensor>,
        gamma: f32,
        lambda: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let states = Tensor::stack(&self.0.states, 0)?;
        let values_stacked = value_func.calculate_values(&states).unwrap();
        let values: Vec<f32> = values_stacked.to_vec1()?;
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

pub struct RolloutBatch {
    pub observations: Tensor,
    pub actions: Tensor,
    pub returns: Tensor,
    pub advantages: Tensor,
    pub logp_old: Tensor,
}

pub fn calculate_advantages_and_returns(
    rollouts: &[CandleRolloutBuffer],
    value_func: &impl ValueFunction<Tensor = Tensor>,
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

pub struct RolloutBatchIterator<'a> {
    rollouts: &'a [CandleRolloutBuffer],
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
        rollouts: &'a [CandleRolloutBuffer],
        advantages: &'a Advantages,
        returns: &'a Returns,
        logps: &'a Logps,
        sample_size: usize,
        device: Device,
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
                let (state, action) = self.rollouts[*rollout_idx].0.sample_point(*idx);
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
