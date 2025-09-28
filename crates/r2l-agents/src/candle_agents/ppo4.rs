use crate::candle_agents::ModuleWithValueFunction;
use crate::candle_agents::ppo::{HookResult, PPOBatchData};
use anyhow::Result;
use candle_core::{Device, Tensor as CandleTensor};
use r2l_candle_lm::{
    learning_module2::PolicyValuesLosses,
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
};
use r2l_core::agents::Agent4;
use r2l_core::sampler3::buffer_stack::BufferStack3;
use r2l_core::{
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
    utils::rollout_buffer::{Advantages, Logps, Returns},
};
use std::marker::PhantomData;
use std::ops::Deref;

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub trait PPOHooksTrait4<M: ModuleWithValueFunction> {
    fn before_learning_hook(
        &mut self,
        agent: &mut CandlePPOCore4<M>,
        buffers: &BufferStack3<CandleTensor>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook(
        &mut self,
        buffers: &BufferStack3<CandleTensor>,
        agent: &mut CandlePPOCore4<M>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        agent: &mut CandlePPOCore4<M>,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct PPO4DefaultHooks<M: ModuleWithValueFunction> {
    _lm: PhantomData<M>,
}

impl<M: ModuleWithValueFunction> PPO4DefaultHooks<M> {
    pub fn new() -> Self {
        Self { _lm: PhantomData }
    }
}

impl<M: ModuleWithValueFunction> PPOHooksTrait4<M> for PPO4DefaultHooks<M> {}

pub struct CandlePPOCore4<M: ModuleWithValueFunction> {
    pub module: M,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

pub struct CandlePPO4<M: ModuleWithValueFunction, H: PPOHooksTrait4<M>> {
    pub ppo: CandlePPOCore4<M>,
    pub hooks: H,
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait4<M>> CandlePPO4<M, H> {
    fn batching_loop(
        &mut self,
        buffers: &BufferStack3<CandleTensor>,
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> Result<()> {
        let mut index_iterator = buffers.index_iterator(self.ppo.sample_size);
        let ppo = &mut self.ppo;
        loop {
            let Some(indicies) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = buffers.sample(&indicies);
            let advantages = advantages.sample(&indicies);
            let advantages = CandleTensor::from_slice(&advantages, advantages.len(), &ppo.device)?;
            let logp_old = logps.sample(&indicies);
            let logp_old = CandleTensor::from_slice(&logp_old, logp_old.len(), &ppo.device)?;
            let returns = returns.sample(&indicies);
            let returns = CandleTensor::from_slice(&returns, returns.len(), &ppo.device)?;
            let logp = Logp(
                ppo.module
                    .get_policy_ref()
                    .log_probs(&observations, &actions)?,
            );
            let values_pred = ValuesPred(ppo.module.value_func().calculate_values(&observations)?);
            let mut value_loss = ValueLoss(returns.sub(&values_pred)?.sqr()?.mean_all()?);
            let logp_diff = LogpDiff((logp.deref() - &logp_old)?);
            let ratio = logp_diff.exp()?;
            let clip_adv =
                (ratio.clamp(1. - ppo.clip_range, 1. + ppo.clip_range)? * advantages.clone())?;
            let mut policy_loss = PolicyLoss(
                CandleTensor::minimum(&(&ratio * &advantages)?, &clip_adv)?
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

    fn learning_loop(
        &mut self,
        buffers: BufferStack3<CandleTensor>,
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> Result<()> {
        loop {
            self.batching_loop(&buffers, &advantages, &logps, &returns)?;
            let rollout_hook_res = self.hooks.rollout_hook(&buffers, &mut self.ppo);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait4<M>> Agent4 for CandlePPO4<M, H> {
    type Policy = <M as ModuleWithValueFunction>::P;

    fn policy3(&self) -> Self::Policy {
        self.ppo.module.get_inference_policy()
    }

    fn learn3(&mut self, buffers: BufferStack3<<Self::Policy as Policy>::Tensor>) -> Result<()> {
        let (mut advantages, mut returns) = buffers.advantages_and_returns(
            self.ppo.module.value_func(),
            self.ppo.gamma,
            self.ppo.lambda,
        );
        let before_learning_hook_res =
            self.hooks
                .before_learning_hook(&mut self.ppo, &buffers, &mut advantages, &mut returns);
        process_hook_result!(before_learning_hook_res);
        let logps = buffers.logps(&self.policy3());
        self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}
