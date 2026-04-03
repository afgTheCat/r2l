pub mod burn_agents;
pub mod candle_agents;
pub mod ppo;
pub mod ppo2;

use r2l_core::distributions::Policy;
use r2l_core::policies::ValueFunction;
use r2l_core::tensor::R2lTensor;
use r2l_core::{rng::RNG, utils::rollout_buffer::Logps};
use r2l_core::{
    sampler::buffer::TrajectoryContainer,
    utils::rollout_buffer::{Advantages, Returns},
};
use rand::seq::SliceRandom;

pub enum HookResult {
    Continue,
    Break,
}

#[macro_export]
macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            $crate::HookResult::Continue => {}
            $crate::HookResult::Break => return Ok(()),
        }
    };
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
        if self.sample_size + self.current >= total_size {
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

pub fn buffer_advantages_and_returns<T1: R2lTensor, T2: R2lTensor, L: Fn(&T1) -> T2>(
    buffer: &impl TrajectoryContainer<Tensor = T1>,
    value_func: &impl ValueFunction<Tensor = T2>,
    gamma: f32,
    lambda: f32,
    lifter: L,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let mut states = buffer.states().map(&lifter).collect::<Vec<_>>();
    states.push(buffer.next_states().last().map(&lifter).unwrap());
    let values_stacked = value_func.calculate_values(&states).unwrap();
    let values: Vec<f32> = values_stacked.to_vec();
    let total_steps = buffer.rewards().count();
    let mut advantages: Vec<f32> = vec![0.; total_steps];
    let mut returns: Vec<f32> = vec![0.; total_steps];
    let mut last_gae_lam: f32 = 0.;

    for i in (0..total_steps).rev() {
        let mut dones = buffer
            .terminated()
            .zip(buffer.trancuated())
            .map(|(terminated, trancuated)| terminated || trancuated);
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
