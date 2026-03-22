use crate::burn_agents::ppo::HookResult;
use burn::{
    prelude::Backend,
    tensor::{Tensor as BurnTensor, backend::AutodiffBackend},
};
use r2l_burn_lm::{
    learning_module::{BurnPolicy, ParalellActorCriticLM, PolicyValuesLosses},
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
};
use r2l_core::distributions::Policy;
use r2l_core::policies::{LearningModule, ValueFunction};
use r2l_core::rng::RNG;
use r2l_core::utils::rollout_buffer::{Advantages, Logps, Returns};
use r2l_core::{agents::Agent5, sampler5::buffer::TrajectoryContainer};
use rand::seq::SliceRandom;
use std::ops::Deref;

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub struct PPOBatchData<B: Backend> {
    pub logp: Logp<B>,
    pub values_pred: ValuesPred<B>,
    pub logp_diff: LogpDiff<B>,
    pub ratio: BurnTensor<B, 1>,
}

pub trait BurnPPOHooksTrait<B: AutodiffBackend, D: BurnPolicy<B>> {
    fn before_learning_hook<T: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        _agent: &mut BurnPPOCore<B, D>,
        _rollout_buffers: &[T],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<T: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        _agent: &mut BurnPPOCore<B, D>,
        _rollout_buffers: &[T],
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _agent: &mut BurnPPOCore<B, D>,
        _policy_loss: &mut PolicyLoss<B>,
        _value_loss: &mut ValueLoss<B>,
        _data: &PPOBatchData<B>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct BurnPPOCore<B: AutodiffBackend, D: BurnPolicy<B>> {
    pub lm: ParalellActorCriticLM<B, D>,
    pub clip_range: f32,
    pub sample_size: usize,
    pub gamma: f32,
    pub lambda: f32,
}

pub struct BurnPPO<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnPPOHooksTrait<B, D>> {
    pub core: BurnPPOCore<B, D>,
    pub hooks: H,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> BurnPPOCore<B, D> {
    pub fn new(
        lm: ParalellActorCriticLM<B, D>,
        clip_range: f32,
        sample_size: usize,
        gamma: f32,
        lambda: f32,
    ) -> Self {
        Self {
            lm,
            clip_range,
            sample_size,
            gamma,
            lambda,
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnPPOHooksTrait<B, D>> BurnPPO<B, D, H> {
    pub fn new(core: BurnPPOCore<B, D>, hooks: H) -> Self {
        Self { core, hooks }
    }

    fn batching_loop<C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        buffers: &[C],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new::<B, C>(buffers, self.core.sample_size);
        let ppo = &mut self.core;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample::<B, C>(buffers, &indices);
            let advantages = advantages.sample(&indices);
            let advantages = BurnTensor::from_data(advantages.as_slice(), &Default::default());
            let logp_old = logps.sample(&indices);
            let logp_old = BurnTensor::from_data(logp_old.as_slice(), &Default::default());
            let returns = returns.sample(&indices);
            let returns = BurnTensor::from_data(returns.as_slice(), &Default::default());
            let logp = Logp(
                ppo.lm
                    .model
                    .distr
                    .log_probs(&observations, &actions)?,
            );
            let values_pred = ValuesPred(ppo.lm.calculate_values(&observations)?);
            let value_diff = returns.clone() - values_pred.deref().clone();
            let mut value_loss = ValueLoss((value_diff.clone() * value_diff).mean());
            let logp_diff = LogpDiff(logp.deref().clone() - logp_old);
            let ratio = logp_diff.clone().exp();
            let clip_adv = ratio
                .clone()
                .clamp(1. - ppo.clip_range, 1. + ppo.clip_range)
                * advantages.clone();
            let mut policy_loss =
                PolicyLoss((-(ratio.clone() * advantages).min_pair(clip_adv)).mean());
            let ppo_data = PPOBatchData {
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            let hook_result =
                self.hooks
                    .batch_hook(ppo, &mut policy_loss, &mut value_loss, &ppo_data)?;
            ppo.lm.update(PolicyValuesLosses {
                policy_loss: policy_loss.0,
                value_loss: value_loss.0,
            })?;
            match hook_result {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
        }
    }

    fn learning_loop<C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        buffers: &[C],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> anyhow::Result<()> {
        loop {
            self.batching_loop(buffers, &advantages, &logps, &returns)?;
            process_hook_result!(self.hooks.rollout_hook(&mut self.core, buffers));
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnPPOHooksTrait<B, D>> Agent5
    for BurnPPO<B, D, H>
{
    type Tensor = BurnTensor<B::InnerBackend, 1>;
    type Policy = D::InnerModule;

    fn policy(&self) -> Self::Policy {
        self.core.lm.model.distr.valid()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns::<B, C>(
            buffers,
            &self.core.lm,
            self.core.gamma,
            self.core.lambda,
        )?;
        process_hook_result!(self.hooks.before_learning_hook(
            &mut self.core,
            buffers,
            &mut advantages,
            &mut returns
        ));
        let logps = logps::<B, C>(buffers, &self.policy())?;
        self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}

fn uplift_tensor<const N: usize, B: AutodiffBackend>(
    tensor: BurnTensor<B::InnerBackend, N>,
) -> BurnTensor<B, N> {
    BurnTensor::from_data(tensor.into_data(), &Default::default())
}

fn buffer_advantages_and_returns<
    B: AutodiffBackend,
    C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>,
>(
    buffer: &C,
    value_func: &impl ValueFunction<Tensor = BurnTensor<B, 1>>,
    gamma: f32,
    lambda: f32,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let mut states = buffer.states().cloned().collect::<Vec<_>>();
    states.push(buffer.next_states().last().unwrap().clone());
    let states = states.into_iter().map(uplift_tensor::<1, B>).collect::<Vec<_>>();
    let values = value_func.calculate_values(&states)?;
    let values: Vec<f32> = values.to_data().to_vec().unwrap();
    let total_steps = buffer.rewards().count();
    let mut advantages: Vec<f32> = vec![0.; total_steps];
    let mut returns: Vec<f32> = vec![0.; total_steps];
    let mut last_gae_lam: f32 = 0.;

    for i in (0..total_steps).rev() {
        let mut dones = buffer.dones();
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

fn buffers_advantages_and_returns<
    B: AutodiffBackend,
    C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>,
>(
    buffers: &[C],
    value_func: &impl ValueFunction<Tensor = BurnTensor<B, 1>>,
    gamma: f32,
    lambda: f32,
) -> anyhow::Result<(Advantages, Returns)> {
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];
    for buffer in buffers {
        let (advantages, returns) =
            buffer_advantages_and_returns::<B, C>(buffer, value_func, gamma, lambda)?;
        advantage_vec.push(advantages);
        returns_vec.push(returns);
    }
    Ok((Advantages(advantage_vec), Returns(returns_vec)))
}

struct BatchIndexIterator {
    indices: Vec<(usize, usize)>,
    sample_size: usize,
    current: usize,
}

impl BatchIndexIterator {
    fn new<B: AutodiffBackend, C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        buffers: &[C],
        sample_size: usize,
    ) -> Self {
        let mut indices = (0..buffers.len())
            .flat_map(|i| {
                let buffer = &buffers[i];
                (0..buffer.rewards().count()).map(|j| (i, j)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        RNG.with_borrow_mut(|rng| indices.shuffle(rng));
        Self {
            indices,
            sample_size,
            current: 0,
        }
    }

    fn iter(&mut self) -> Option<Vec<(usize, usize)>> {
        let total_size = self.indices.len();
        if self.sample_size + self.current >= total_size {
            return None;
        }
        let batch_indices = &self.indices[self.current..self.current + self.sample_size];
        self.current += self.sample_size;
        Some(batch_indices.to_owned())
    }
}

fn logps<B: AutodiffBackend, C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
    buffers: &[C],
    policy: &impl Policy<Tensor = BurnTensor<B::InnerBackend, 1>>,
) -> anyhow::Result<Logps> {
    let mut logps = vec![];
    for buffer in buffers {
        let states = buffer.states().cloned().collect::<Vec<_>>();
        let actions = buffer.actions().cloned().collect::<Vec<_>>();
        let logp = policy.log_probs(&states, &actions)?;
        logps.push(logp.to_data().to_vec().unwrap());
    }
    Ok(Logps(logps))
}

fn sample<B: AutodiffBackend, C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
    buffers: &[C],
    indices: &[(usize, usize)],
) -> (Vec<BurnTensor<B, 1>>, Vec<BurnTensor<B, 1>>) {
    let mut observations = vec![];
    let mut actions = vec![];
    for (buffer_idx, idx) in indices {
        let observation = buffers[*buffer_idx].states().nth(*idx).unwrap().clone();
        let action = buffers[*buffer_idx].actions().nth(*idx).unwrap().clone();
        observations.push(uplift_tensor::<1, B>(observation));
        actions.push(uplift_tensor::<1, B>(action));
    }
    (observations, actions)
}
