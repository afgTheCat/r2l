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
    sampler2::Buffer,
    utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer},
};
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

pub struct EmptyPPO3Hooks;

impl<A> PPOHooksTrait<A> for EmptyPPO3Hooks {}

pub struct CandlePPOCore<D: Policy, LM: PPOLearningModule> {
    pub distribution: D,
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
            let logp = Logp(
                ppo.distribution
                    .log_probs(&batch.observations, &batch.actions)?,
            );
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

    fn learning_loop2<B: Buffer<Tensor = Tensor>>(
        &mut self,
        buffers: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) {
        loop {
            // TODO: batch
        }
    }
}

impl<D: Policy<Tensor = Tensor> + Clone, LM: PPOLearningModule> Agent for CandlePPO<D, LM> {
    type Policy = D;

    fn policy(&self) -> Self::Policy {
        self.ppo.distribution.clone()
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

fn calculate_advantages_and_returns2<B: Buffer<Tensor = Tensor>>(
    buffers: &[B],
    value_func: &impl ValueFunction<Tensor = Tensor>,
    gamma: f32,
    lambda: f32,
) -> Result<(Advantages, Returns)> {
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];
    for buff in buffers {
        let total_steps = buff.total_steps();
        let values_stacked = value_func.calculate_values(buff.all_states())?;
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

impl<D: Policy<Tensor = Tensor> + Clone, LM: PPOLearningModule, B: Buffer<Tensor = Tensor>>
    Agent2<B> for CandlePPO<D, LM>
{
    type Policy = D;

    fn policy2(&self) -> Self::Policy {
        self.ppo.distribution.clone()
    }

    fn learn2(&mut self, buffers: Vec<B>) -> Result<()> {
        let (mut advantages, mut returns) = calculate_advantages_and_returns2(
            &buffers,
            &self.ppo.learning_module,
            self.ppo.gamma,
            self.ppo.lambda,
        )?;

        // let before_learning_hook_res = self.hooks.before_learning_hook(
        //     &mut self.ppo,
        //     &mut rollouts,
        //     &mut advantages,
        //     &mut returns,
        // );
        // process_hook_result!(before_learning_hook_res);

        let mut logps: Vec<Vec<f32>> = vec![];
        for buff in buffers {
            let total_steps = buff.total_steps();
            let states = &buff.all_states()[0..total_steps - 1];
            let actions = buff.actions();
            logps.push(
                self.policy()
                    .log_probs(states, actions)
                    .map(|t| t.squeeze(0).unwrap().to_vec1().unwrap())
                    .unwrap(),
            );
        }
        let logps = Logps(logps);

        // let logps = Logps(
        //     rollouts
        //         .iter()
        //         .map(|roll| {
        //             let states = &roll.0.states[0..roll.0.states.len() - 1];
        //             let actions = &roll.0.actions;
        //             self.policy()
        //                 .log_probs(states, actions)
        //                 .map(|t| t.squeeze(0).unwrap().to_vec1().unwrap())
        //         })
        //         .collect::<Result<Vec<Vec<f32>>>>()?,
        // );

        // self.learning_loop(rollouts, advantages, returns, logps)?;

        Ok(())
    }
}
