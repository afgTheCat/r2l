use anyhow::Result;
use candle_core::{Device, Tensor};
use r2l_candle_lm::{
    candle_rollout_buffer::{
        CandleRolloutBuffer, RolloutBatch, RolloutBatchIterator, calculate_advantages_and_returns,
    },
    learning_module::{LearningModuleKind, PolicyValuesLosses},
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
};
use r2l_core::{
    agents::{Agent, Agent2},
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
    rng::RNG,
    sampler2::Buffer,
    utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer},
};
use rand::seq::SliceRandom;
use std::ops::Deref;

pub enum HookResult {
    Continue,
    Break,
}

pub struct PPOBatchData {
    pub logp: Logp,
    pub values_pred: ValuesPred,
    pub logp_diff: LogpDiff,
    pub ratio: Tensor,
}

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub trait PPOLearningModule:
    LearningModule<Losses = PolicyValuesLosses> + ValueFunction<Tensor = Tensor>
{
}

impl PPOLearningModule for LearningModuleKind {}

// TODO: we could just pass the agent inside. Also we should probably set the rollout buffer inside
// the agent to some buffer, so that we also don't really need that as a parameter. Much cleaner
// design to be honest
pub trait PPOHooksTrait<A> {
    fn before_learning_hook(
        &mut self,
        agent: &mut A,
        rollout_buffers: &mut Vec<CandleRolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook(
        &mut self,
        agent: &mut A,
        rollout_buffers: &Vec<CandleRolloutBuffer>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        agent: &mut A,
        rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub trait PPOHooksTrait2<B: Buffer<Tensor = Tensor>, A> {
    fn before_learning_hook(
        &mut self,
        agent: &mut A,
        buffers: &[B],
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook(&mut self, buffers: &[B], agent: &mut A) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        agent: &mut A,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct EmptyPPO3Hooks;

impl<A> PPOHooksTrait<A> for EmptyPPO3Hooks {}

pub struct CandlePPOCore<P: Policy, LM: PPOLearningModule> {
    pub policy: P,
    pub learning_module: LM,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

// This could be something more generic
pub struct CandlePPO<D: Policy, LM: PPOLearningModule> {
    pub ppo: CandlePPOCore<D, LM>,
    pub hooks: Box<dyn PPOHooksTrait<CandlePPOCore<D, LM>>>,
}

impl<D: Policy<Tensor = Tensor>, LM: PPOLearningModule> CandlePPO<D, LM> {
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        let ppo = &mut self.ppo;
        loop {
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let logp = Logp(ppo.policy.log_probs(&batch.observations, &batch.actions)?);
            let values_pred =
                ValuesPred(ppo.learning_module.calculate_values(&batch.observations)?);
            let mut value_loss = ValueLoss(batch.returns.sub(&values_pred)?.sqr()?.mean_all()?);
            let logp_diff = LogpDiff((logp.deref() - &batch.logp_old)?);
            let ratio = logp_diff.exp()?;
            let clip_adv = (ratio.clamp(1. - ppo.clip_range, 1. + ppo.clip_range)?
                * batch.advantages.clone())?;
            let mut policy_loss = PolicyLoss(
                Tensor::minimum(&(&ratio * &batch.advantages)?, &clip_adv)?
                    .neg()?
                    .mean_all()?,
            );
            let ppo_data = PPOBatchData {
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            let hook_result =
                self.hooks
                    .batch_hook(ppo, &batch, &mut policy_loss, &mut value_loss, &ppo_data)?;
            ppo.learning_module.update(PolicyValuesLosses {
                policy_loss,
                value_loss,
            })?;
            match hook_result {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
        }
    }

    // TODO: rename this to learning loop
    fn learning_loop(
        &mut self,
        rollouts: Vec<CandleRolloutBuffer>,
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> Result<()> {
        loop {
            let mut batch_iter = RolloutBatchIterator::new(
                &rollouts,
                &advantages,
                &returns,
                &logps,
                self.ppo.sample_size,
                self.ppo.device.clone(),
            );
            self.batching_loop(&mut batch_iter)?;
            let rollout_hook_res = self.hooks.rollout_hook(&mut self.ppo, &rollouts);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<D: Policy<Tensor = Tensor> + Clone, LM: PPOLearningModule> Agent for CandlePPO<D, LM> {
    type Policy = D;

    fn policy(&self) -> Self::Policy {
        self.ppo.policy.clone()
    }

    fn learn(&mut self, rollouts: Vec<RolloutBuffer<Tensor>>) -> Result<()> {
        let mut rollouts: Vec<CandleRolloutBuffer> = rollouts
            .into_iter()
            .map(CandleRolloutBuffer::from)
            .collect();
        let (mut advantages, mut returns) = calculate_advantages_and_returns(
            &rollouts,
            &self.ppo.learning_module,
            self.ppo.gamma,
            self.ppo.lambda,
        );
        let before_learning_hook_res = self.hooks.before_learning_hook(
            &mut self.ppo,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        );
        process_hook_result!(before_learning_hook_res);
        let logps = Logps(
            rollouts
                .iter()
                .map(|roll| {
                    let states = &roll.0.states[0..roll.0.states.len() - 1];
                    let actions = &roll.0.actions;
                    self.policy()
                        .log_probs(states, actions)
                        .map(|t| t.squeeze(0).unwrap().to_vec1().unwrap())
                })
                .collect::<Result<Vec<Vec<f32>>>>()?,
        );
        self.learning_loop(rollouts, advantages, returns, logps)?;
        Ok(())
    }
}

fn calculate_advantages_and_returns2<B: Buffer>(
    buffers: &[B],
    value_func: &impl ValueFunction<Tensor = Tensor>,
    gamma: f32,
    lambda: f32,
) -> Result<(Advantages, Returns)>
where
    Tensor: From<<B as Buffer>::Tensor>,
{
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];
    for buff in buffers {
        let total_steps = buff.total_steps();
        let all_states = buff
            .all_states()
            .into_iter()
            .map(|s| s.clone().into())
            .collect::<Vec<_>>();
        let values_stacked = value_func.calculate_values(&all_states)?;
        let values: Vec<f32> = values_stacked.to_vec1()?;
        let mut advantages: Vec<f32> = vec![0.; total_steps];
        let mut returns: Vec<f32> = vec![0.; total_steps];
        let mut last_gae_lam: f32 = 0.;
        for i in (0..total_steps).rev() {
            let next_non_terminal = if buff.dones()[i] {
                last_gae_lam = 0.;
                0f32
            } else {
                1.
            };
            let delta = buff.rewards()[i] + next_non_terminal * gamma * values[i + 1] - values[i];
            last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
            advantages[i] = last_gae_lam;
            returns[i] = last_gae_lam + values[i];
        }
        advantage_vec.push(advantages);
        returns_vec.push(returns);
    }
    Ok((Advantages(advantage_vec), Returns(returns_vec)))
}

pub struct Buffers<'a, B: Buffer>(&'a [B]);

impl<'a, B: Buffer<Tensor = Tensor>> Buffers<'a, B> {
    fn sample(&self, indicies: &[(usize, usize)]) -> (Vec<Tensor>, Vec<Tensor>) {
        let mut observations = vec![];
        let mut actions = vec![];
        for (buffer_idx, idx) in indicies {
            let observation = &self.0[*buffer_idx].all_states()[*idx];
            let action = &self.0[*buffer_idx].actions()[*idx];
            observations.push(observation.clone());
            actions.push(action.clone());
        }
        (observations, actions)
    }
}

pub struct CandlePPO2<D: Policy, B: Buffer<Tensor = Tensor>, LM: PPOLearningModule> {
    pub ppo: CandlePPOCore<D, LM>,
    pub hooks: Box<dyn PPOHooksTrait2<B, CandlePPOCore<D, LM>>>,
}

struct BatchIndexIterator {
    indicies: Vec<(usize, usize)>,
    sample_size: usize,
    current: usize,
}

impl BatchIndexIterator {
    pub fn new<B: Buffer>(buffers: &[B], sample_size: usize) -> Self {
        let mut indicies = (0..buffers.len())
            .flat_map(|i| {
                let rb = &buffers[i];
                (0..rb.total_steps()).map(|j| (i, j)).collect::<Vec<_>>()
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
        Some(batch_indicies.to_owned())
    }
}

impl<D: Policy<Tensor = Tensor>, B: Buffer<Tensor = Tensor>, LM: PPOLearningModule>
    CandlePPO2<D, B, LM>
{
    fn batching_loop(
        &mut self,
        buffers: Buffers<B>,
        mut index_iterator: BatchIndexIterator,
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> Result<()> {
        let ppo = &mut self.ppo;
        loop {
            let Some(indicies) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = buffers.sample(&indicies);
            let advantages = advantages.sample(&indicies);
            let advantages = Tensor::from_slice(&advantages, advantages.len(), &ppo.device)?;
            let logp_old = logps.sample(&indicies);
            let logp_old = Tensor::from_slice(&logp_old, logp_old.len(), &ppo.device)?;
            let returns = returns.sample(&indicies);
            let returns = Tensor::from_slice(&returns, returns.len(), &ppo.device)?;

            let logp = Logp(ppo.policy.log_probs(&observations, &actions)?);
            let values_pred = ValuesPred(ppo.learning_module.calculate_values(&observations)?);
            let mut value_loss = ValueLoss(returns.sub(&values_pred)?.sqr()?.mean_all()?);
            let logp_diff = LogpDiff((logp.deref() - &logp_old)?);
            let ratio = logp_diff.exp()?;
            let clip_adv =
                (ratio.clamp(1. - ppo.clip_range, 1. + ppo.clip_range)? * advantages.clone())?;
            let mut policy_loss = PolicyLoss(
                Tensor::minimum(&(&ratio * &advantages)?, &clip_adv)?
                    .neg()?
                    .mean_all()?,
            );
            let ppo_data = PPOBatchData {
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            let hook_result =
                self.hooks
                    .batch_hook(ppo, &mut policy_loss, &mut value_loss, &ppo_data)?;
            ppo.learning_module.update(PolicyValuesLosses {
                policy_loss,
                value_loss,
            })?;
            match hook_result {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
        }
    }

    fn learning_loop(
        &mut self,
        buffers: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> Result<()> {
        loop {
            let index_iterator = BatchIndexIterator::new(buffers, self.ppo.sample_size);
            self.batching_loop(
                Buffers(buffers),
                index_iterator,
                &advantages,
                &logps,
                &returns,
            )?;
            let rollout_hook_res = self.hooks.rollout_hook(buffers, &mut self.ppo);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<P: Policy<Tensor = Tensor> + Clone, LM: PPOLearningModule, B2: Buffer<Tensor = Tensor>> Agent2
    for CandlePPO2<P, B2, LM>
{
    type Policy = P;

    fn policy2(&self) -> Self::Policy {
        self.ppo.policy.clone()
    }

    fn learn2<B: Buffer>(&mut self, buffers: &[B]) -> Result<()>
    where
        <B as Buffer>::Tensor: Into<<Self::Policy as Policy>::Tensor>,
    {
        // let (mut advantages, mut returns) = calculate_advantages_and_returns2(
        //     buffers,
        //     &self.ppo.learning_module,
        //     self.ppo.gamma,
        //     self.ppo.lambda,
        // )?;
        // let before_learning_hook_res =
        //     self.hooks
        //         .before_learning_hook(&mut self.ppo, buffers, &mut advantages, &mut returns);
        // process_hook_result!(before_learning_hook_res);
        //
        // let mut logps: Vec<Vec<f32>> = vec![];
        // for buff in buffers {
        //     let total_steps = buff.total_steps();
        //     let states: Vec<Tensor> = buff.all_states()[0..total_steps - 1]
        //         .into_iter()
        //         .map(|t| t.clone().into())
        //         .collect();
        //     let actions = buff.actions();
        //     logps.push(
        //         self.policy2()
        //             .log_probs(&states, actions)
        //             .map(|t| t.squeeze(0).unwrap().to_vec1().unwrap())
        //             .unwrap(),
        //     );
        // }
        // let logps = Logps(logps);
        //
        // self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}
