use crate::candle_agents::ModuleWithValueFunction;
use crate::candle_agents::ppo::{HookResult, PPOBatchData};
use anyhow::Result;
use candle_core::{Device, Tensor as CandleTensor};
use r2l_candle_lm::{
    learning_module2::PolicyValuesLosses,
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
};
use r2l_core::agents::Agent3;
use r2l_core::sampler3::buffers::{BatchIndexIterator, BufferStack};
use r2l_core::tensor::R2lTensor;
use r2l_core::{
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
    sampler2::Buffer,
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

pub trait PPOHooksTrait3<M: ModuleWithValueFunction> {
    fn before_learning_hook<BT: R2lTensor, B: Buffer<Tensor = BT>>(
        &mut self,
        agent: &mut CandlePPOCore3<M>,
        buffers: &BufferStack<B>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<HookResult>
    where
        CandleTensor: From<BT>,
    {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<BT: R2lTensor, B: Buffer<Tensor = BT>>(
        &mut self,
        buffers: &BufferStack<B>,
        agent: &mut CandlePPOCore3<M>,
    ) -> candle_core::Result<HookResult>
    where
        CandleTensor: From<BT>,
    {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        agent: &mut CandlePPOCore3<M>,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct PPO3DefaultHooks<M: ModuleWithValueFunction> {
    _lm: PhantomData<M>,
}

impl<M: ModuleWithValueFunction> PPO3DefaultHooks<M> {
    pub fn new() -> Self {
        Self { _lm: PhantomData }
    }
}

impl<M: ModuleWithValueFunction> PPOHooksTrait3<M> for PPO3DefaultHooks<M> {}

pub struct CandlePPOCore3<M: ModuleWithValueFunction> {
    pub module: M,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

pub struct CandlePPO3<M: ModuleWithValueFunction, H: PPOHooksTrait3<M>> {
    pub ppo: CandlePPOCore3<M>,
    pub hooks: H,
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait3<M>> CandlePPO3<M, H> {
    fn batching_loop<B: Buffer>(
        &mut self,
        buffers: &BufferStack<B>,
        mut index_iterator: BatchIndexIterator,
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> Result<()>
    where
        CandleTensor: From<<B as Buffer>::Tensor>,
    {
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

    fn learning_loop<B: Buffer>(
        &mut self,
        buffers: BufferStack<B>,
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> Result<()>
    where
        CandleTensor: From<<B as Buffer>::Tensor>,
    {
        loop {
            let index_iterator = buffers.index_iterator(self.ppo.sample_size);
            self.batching_loop(&buffers, index_iterator, &advantages, &logps, &returns)?;
            let rollout_hook_res = self.hooks.rollout_hook(&buffers, &mut self.ppo);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait3<M>> Agent3 for CandlePPO3<M, H> {
    type Policy = <M as ModuleWithValueFunction>::P;

    fn policy3(&self) -> Self::Policy {
        self.ppo.module.get_inference_policy()
    }

    fn learn3<B: Buffer>(&mut self, buffers: BufferStack<B>) -> Result<()>
    where
        CandleTensor: From<<B as Buffer>::Tensor>,
    {
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
