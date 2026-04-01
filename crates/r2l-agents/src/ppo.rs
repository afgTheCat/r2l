use r2l_burn_lm::BurnModuleWithValueFunction;
use r2l_candle_lm::{CandleModuleWithValueFunction, learning_module::PolicyValuesLosses};
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    policies::{ModuleWithValueFunction, ValueFunction},
    sampler::buffer::TrajectoryContainer,
    tensor::R2lTensor,
    utils::rollout_buffer::{Advantages, Logps, Returns},
};
use std::marker::PhantomData;

use crate::{
    BatchIndexIterator, HookResult, buffers_advantages_and_returns,
    candle_agents::ModuleWithValueFunction as MW, logps, sample,
};
use candle_core::Tensor as CandleTensor;

pub struct PPOBatchData<T: R2lTensor> {
    pub logp: T,
    pub values_pred: T,
    pub logp_diff: T,
    pub ratio: T,
}

struct PPOCore<M: ModuleWithValueFunction> {
    pub module: M,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

// So this could be a unified implementation. One PPOHooks trait, nice
pub trait PPOHooks<M: ModuleWithValueFunction> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _agent: &mut PPOCore<M>,
        _buffers: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _buffers: &[B],
        _agent: &mut PPOCore<M>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _agent: &mut PPOCore<M>,
        _losses: &mut M::Losses,
        _data: &PPOBatchData<M::Tensor>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

struct PPO<M: ModuleWithValueFunction, H: PPOHooks<M>> {
    core: PPOCore<M>,
    hooks: H,
}

// Something like this might just work
struct CandlePPO<M: CandleModuleWithValueFunction, H: PPOHooks<M>> {
    ppo: PPO<M, H>,
    device: candle_core::Device,
}

impl<M: CandleModuleWithValueFunction, H: PPOHooks<M>> CandlePPO<M, H> {
    fn batch_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.ppo.core.sample_size);
        let ppo = &mut self.ppo.core;
        loop {
            let Some(indicies) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indicies, |t| t.clone());
            let advantages = advantages.sample(&indicies);
            let advantages = CandleTensor::from_slice(&advantages, advantages.len(), &self.device)?;
            let logp_old = logps.sample(&indicies);
            let logp_old = CandleTensor::from_slice(&logp_old, logp_old.len(), &self.device)?;
            let returns = returns.sample(&indicies);
            let returns = CandleTensor::from_slice(&returns, returns.len(), &self.device)?;
            let logp = ppo.module.get_policy().log_probs(&observations, &actions)?;
            let values_pred = ppo.module.value_func().calculate_values(&observations)?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean_all()?;
            let logp_diff = (&logp - &logp_old)?;
            let ratio = logp_diff.exp()?;
            let clip_adv =
                (ratio.clamp(1. - ppo.clip_range, 1. + ppo.clip_range)? * advantages.clone())?;
            let policy_loss = CandleTensor::minimum(&(&ratio * &advantages)?, &clip_adv)?
                .neg()?
                .mean_all()?;
            let mut losses = PolicyValuesLosses::new(policy_loss, value_loss);
            let ppo_data = PPOBatchData {
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            match self.ppo.hooks.batch_hook(ppo, &mut losses, &ppo_data)? {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
            ppo.module.update(losses)?;
        }
    }

    fn learning_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> anyhow::Result<()> {
        loop {
            self.batch_loop(buffers, &advantages, &logps, &returns)?;
            let rollout_hook_res = self.ppo.hooks.rollout_hook(buffers, &mut self.ppo.core);
            crate::process_hook_result!(rollout_hook_res);
        }
    }
}

impl<M: CandleModuleWithValueFunction, H: PPOHooks<M>> Agent for CandlePPO<M, H> {
    type Tensor = M::InferenceTensor;

    type Policy = M::InferencePolicy;

    fn policy(&self) -> Self::Policy {
        self.ppo.core.module.get_inference_policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns(
            buffers,
            self.ppo.core.module.value_func(),
            self.ppo.core.gamma,
            self.ppo.core.lambda,
            |t| t.clone(),
        )?;
        crate::process_hook_result!(self.ppo.hooks.before_learning_hook(
            &mut self.ppo.core,
            buffers,
            &mut advantages,
            &mut returns
        ));
        let logps = logps(buffers, &self.policy());
        self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}

struct BurnPPO<
    B: burn::tensor::backend::AutodiffBackend,
    M: BurnModuleWithValueFunction<B>,
    H: PPOHooks<M>,
> {
    ppo: PPO<M, H>,
    _phantom: PhantomData<B>,
}

impl<B: burn::tensor::backend::AutodiffBackend, M: BurnModuleWithValueFunction<B>, H: PPOHooks<M>>
    Agent for BurnPPO<B, M, H>
{
    type Tensor = M::InferenceTensor;

    type Policy = M::InferencePolicy;

    fn policy(&self) -> Self::Policy {
        self.ppo.core.module.get_inference_policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        todo!()
    }
}
