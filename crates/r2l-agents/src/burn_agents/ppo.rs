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
use r2l_core::{agents::Agent, sampler::buffer::TrajectoryContainer};
use rand::seq::SliceRandom;
use std::ops::Deref;

use crate::{
    BatchIndexIterator, HookResult, buffers_advantages_and_returns, burn_agents::uplift_tensor,
    logps, sample,
};

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
        let mut index_iterator = BatchIndexIterator::new(buffers, self.core.sample_size);
        let ppo = &mut self.core;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indices, uplift_tensor);
            let advantages = advantages.sample(&indices);
            let advantages = BurnTensor::from_data(advantages.as_slice(), &Default::default());
            let logp_old = logps.sample(&indices);
            let logp_old = BurnTensor::from_data(logp_old.as_slice(), &Default::default());
            let returns = returns.sample(&indices);
            let returns = BurnTensor::from_data(returns.as_slice(), &Default::default());
            let logp = Logp(ppo.lm.model.distr.log_probs(&observations, &actions)?);
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
            crate::process_hook_result!(self.hooks.rollout_hook(&mut self.core, buffers));
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnPPOHooksTrait<B, D>> Agent for BurnPPO<B, D, H> {
    type Tensor = BurnTensor<B::InnerBackend, 1>;
    type Policy = D::InnerModule;

    fn policy(&self) -> Self::Policy {
        self.core.lm.model.distr.valid()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns(
            buffers,
            &self.core.lm,
            self.core.gamma,
            self.core.lambda,
            uplift_tensor,
        )?;
        crate::process_hook_result!(self.hooks.before_learning_hook(
            &mut self.core,
            buffers,
            &mut advantages,
            &mut returns
        ));
        let logps = logps(buffers, &self.policy());
        self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}
