use crate::{env::Env, numeric::Buffer, policies::ValueFunction, rng::RNG};
use bincode::{
    BorrowDecode, Decode, Encode,
    error::{DecodeError, EncodeError},
};
use candle_core::{Device, Result, Tensor};
use derive_more::Deref;
use rand::{Rng, seq::SliceRandom};

#[derive(Debug, Default, Clone)]
pub struct RolloutBuffer {
    pub states: Vec<Tensor>,
    pub actions: Vec<Tensor>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub last_state: Option<Tensor>,
}

impl Encode for RolloutBuffer {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        let writer_config = bincode::config::standard();
        let states = self
            .states
            .iter()
            .map(|t| t.to_vec1::<f32>())
            .collect::<Result<Vec<_>>>()
            .map_err(|err| EncodeError::OtherString(err.to_string()))?;
        bincode::encode_into_writer(&states, &mut encoder.writer(), writer_config)?;
        let actions = self
            .actions
            .iter()
            .map(|t| t.to_vec1::<f32>())
            .collect::<Result<Vec<_>>>()
            .map_err(|err| EncodeError::OtherString(err.to_string()))?;
        bincode::encode_into_writer(&actions, &mut encoder.writer(), writer_config)?;
        bincode::encode_into_writer(&self.rewards, &mut encoder.writer(), writer_config)?;
        bincode::encode_into_writer(&self.dones, &mut encoder.writer(), writer_config)?;
        // bincode::encode_into_writer(&self.logps, &mut encoder.writer(), writer_config)?;
        match &self.last_state {
            None => 0u32.encode(encoder)?,
            Some(last_state) => {
                1u32.encode(encoder)?;
                let last_state = last_state
                    .to_vec1::<f32>()
                    .map_err(|err| EncodeError::OtherString(err.to_string()))?;
                bincode::encode_into_writer(&last_state, &mut encoder.writer(), writer_config)?;
            }
        };
        Ok(())
    }
}

// TODO: we need to add the device into the context here to be able to use Tensors on the GPU
impl<C> Decode<C> for RolloutBuffer {
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
        // let logps: Vec<f32> = Vec::decode(decoder)?;
        let last_state_type = u32::decode(decoder)?;
        let last_state: Option<Tensor> = match last_state_type {
            0 => None,
            1 => {
                let last_state: Vec<f32> = Vec::decode(decoder)?;
                let last_state = Tensor::from_slice(&last_state, last_state.len(), &Device::Cpu)
                    .map_err(|err| DecodeError::OtherString(err.to_string()))?;
                Some(last_state)
            }
            _ => unreachable!(),
        };
        Ok(Self {
            states,
            actions,
            rewards,
            dones,
            last_state,
        })
    }
}

impl<'de, C> BorrowDecode<'de, C> for RolloutBuffer {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = C>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        RolloutBuffer::decode(decoder)
    }
}

impl RolloutBuffer {
    pub fn push_step(&mut self, state: Tensor, action: Tensor, reward: f32, done: bool) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.dones.push(done);
    }

    // TODO: we should get rid of the resetting probably and reset here
    pub fn set_states(&mut self, states: Vec<(Tensor, Tensor, f32, bool)>, last_state: Tensor) {
        for (state, action, reward, done) in states {
            self.push_step(state, action, reward, done);
        }
        self.set_last_state(last_state);
    }

    // TODO: this should be the last state
    pub fn set_last_state(&mut self, state: Tensor) {
        self.states.push(state.clone());
        self.last_state = Some(state);
    }

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

    pub fn sample_point(&self, index: usize) -> (&Tensor, &Tensor) {
        (&self.states[index], &self.actions[index])
    }

    pub fn reset(
        &mut self,
        env: &mut impl Env<Tensor = Buffer>,
        device: &Device,
    ) -> Result<Tensor> {
        let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.dones.clear();
        if let Some(last_state) = self.last_state.take() {
            Ok(last_state)
        } else {
            Ok(env.reset(seed).to_candle_tensor(device))
        }
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
    rollouts: &[RolloutBuffer],
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
    rollouts: &'a [RolloutBuffer],
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
        rollouts: &'a [RolloutBuffer],
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
