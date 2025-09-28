use crate::candle_agents::ModuleWithValueFunction;
use anyhow::Result;
use candle_core::{Device, Tensor as CandleTensor};
use r2l_candle_lm::{
    candle_rollout_buffer::{
        CandleRolloutBuffer, RolloutBatch, RolloutBatchIterator, calculate_advantages_and_returns,
    },
    learning_module2::PolicyValuesLosses,
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
};
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
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
    pub ratio: CandleTensor,
}

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

// TODO: we could just pass the agent inside. Also we should probably set the rollout buffer inside
// the agent to some buffer, so that we also don't really need that as a parameter. Much cleaner
// design to be honest
pub trait PPOHooksTrait<M: ModuleWithValueFunction> {
    fn before_learning_hook(
        &mut self,
        _agent: &mut CandlePPOCore<M>,
        _rollout_buffers: &mut Vec<CandleRolloutBuffer>,
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook(
        &mut self,
        _agent: &mut CandlePPOCore<M>,
        _rollout_buffers: &Vec<CandleRolloutBuffer>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _agent: &mut CandlePPOCore<M>,
        _rollout_batch: &RolloutBatch,
        _policy_loss: &mut PolicyLoss,
        _value_loss: &mut ValueLoss,
        _data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct EmptyPPO3Hooks;

impl<M: ModuleWithValueFunction> PPOHooksTrait<M> for EmptyPPO3Hooks {}

pub struct CandlePPOCore<M: ModuleWithValueFunction> {
    pub module: M,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

// This could be something more generic
pub struct CandlePPO<M: ModuleWithValueFunction> {
    pub ppo: CandlePPOCore<M>,
    pub hooks: Box<dyn PPOHooksTrait<M>>,
}

impl<M: ModuleWithValueFunction> CandlePPO<M> {
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        let ppo = &mut self.ppo;
        loop {
            // println!("batching loop");
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let logp = Logp(
                ppo.module
                    .get_policy_ref()
                    .log_probs(&batch.observations, &batch.actions)?,
            );
            let values_pred = ValuesPred(
                ppo.module
                    .value_func()
                    .calculate_values(&batch.observations)?,
            );
            let mut value_loss = ValueLoss(batch.returns.sub(&values_pred)?.sqr()?.mean_all()?);
            let logp_diff = LogpDiff((logp.deref() - &batch.logp_old)?);
            let ratio = logp_diff.exp()?;
            let clip_adv = (ratio.clamp(1. - ppo.clip_range, 1. + ppo.clip_range)?
                * batch.advantages.clone())?;
            let mut policy_loss = PolicyLoss(
                CandleTensor::minimum(&(&ratio * &batch.advantages)?, &clip_adv)?
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
            ppo.module.learning_module().update(PolicyValuesLosses {
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

impl<M: ModuleWithValueFunction> Agent for CandlePPO<M> {
    type Policy = <M as ModuleWithValueFunction>::P;

    fn policy(&self) -> Self::Policy {
        self.ppo.module.get_inference_policy()
    }

    fn learn(&mut self, rollouts: Vec<RolloutBuffer<CandleTensor>>) -> Result<()> {
        let mut rollouts: Vec<CandleRolloutBuffer> = rollouts
            .into_iter()
            .map(CandleRolloutBuffer::from)
            .collect();
        let (mut advantages, mut returns) = calculate_advantages_and_returns(
            &rollouts,
            self.ppo.module.value_func(),
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
