use crate::buffers_advantages_and_returns;
use crate::candle_agents::ModuleWithValueFunction;
use crate::{BatchIndexIterator, HookResult, logps, sample};
use anyhow::Result;
use candle_core::Tensor as CandleTensor;
use candle_core::{Device, Error};
use r2l_candle_lm::learning_module2::PolicyValuesLosses;
use r2l_candle_lm::tensors::{Logp, LogpDiff, PolicyLoss};
use r2l_candle_lm::tensors::{ValueLoss, ValuesPred};
use r2l_core::distributions::Policy;
use r2l_core::policies::LearningModule;
use r2l_core::policies::ValueFunction;
use r2l_core::rng::RNG;
use r2l_core::tensor::R2lTensor;
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

pub struct PPOBatchData {
    pub logp: Logp,
    pub values_pred: ValuesPred,
    pub logp_diff: LogpDiff,
    pub ratio: CandleTensor,
}

pub trait PPOHooksTrait5<M: ModuleWithValueFunction> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        _agent: &mut CandlePPOCore5<M>,
        _buffers: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        _buffers: &[B],
        _agent: &mut CandlePPOCore5<M>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _agent: &mut CandlePPOCore5<M>,
        _policy_loss: &mut PolicyLoss,
        _value_loss: &mut ValueLoss,
        _data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct CandlePPOCore5<M: ModuleWithValueFunction> {
    pub module: M,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

pub struct CandlePPO5<M: ModuleWithValueFunction, H: PPOHooksTrait5<M>> {
    pub ppo: CandlePPOCore5<M>,
    pub hooks: H,
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait5<M>> CandlePPO5<M, H> {
    pub fn new(ppo: CandlePPOCore5<M>, hooks: H) -> Self {
        Self { ppo, hooks }
    }
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait5<M>> CandlePPO5<M, H> {
    fn batching_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.ppo.sample_size);
        let ppo = &mut self.ppo;
        loop {
            let Some(indicies) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indicies);
            let advantages = advantages.sample(&indicies);
            let advantages = CandleTensor::from_slice(&advantages, advantages.len(), &ppo.device)?;
            let logp_old = logps.sample(&indicies);
            let logp_old = CandleTensor::from_slice(&logp_old, logp_old.len(), &ppo.device)?;
            let returns = returns.sample(&indicies);
            let returns = CandleTensor::from_slice(&returns, returns.len(), &ppo.device)?;
            let logp = Logp(
                ppo.module
                    .get_policy_ref()
                    .log_probs(&observations, &actions)
                    .map_err(Error::wrap)?,
            );
            let values_pred = ValuesPred(
                ppo.module
                    .value_func()
                    .calculate_values(&observations)
                    .map_err(Error::wrap)?,
            );
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
            ppo.module
                .learning_module()
                .update(PolicyValuesLosses {
                    policy_loss,
                    value_loss,
                })
                .map_err(Error::wrap)?;
            match hook_result {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
        }
    }

    fn learning_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> Result<()> {
        loop {
            self.batching_loop(buffers, &advantages, &logps, &returns)?;
            let rollout_hook_res = self.hooks.rollout_hook(buffers, &mut self.ppo);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait5<M>> Agent5 for CandlePPO5<M, H> {
    type Tensor = CandleTensor;

    type Policy = <M as ModuleWithValueFunction>::P;

    fn policy(&self) -> Self::Policy {
        self.ppo.module.get_inference_policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns(
            buffers,
            self.ppo.module.value_func(),
            self.ppo.gamma,
            self.ppo.lambda,
        )?;
        process_hook_result!(self.hooks.before_learning_hook(
            &mut self.ppo,
            buffers,
            &mut advantages,
            &mut returns
        ));
        let logps = logps(buffers, &self.policy());
        self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}
