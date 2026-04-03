pub mod burn_ppo;
pub mod candle_ppo;

use burn::tensor::backend::{AutodiffBackend, Backend};
use r2l_burn_lm::learning_module::PolicyValuesLosses as BurnLosses;
use r2l_candle_lm::learning_module::PolicyValuesLosses as CandleLosses;
use r2l_core::{
    distributions::Policy,
    policies::{ModuleWithValueFunction, ValueFunction},
    sampler::buffer::TrajectoryContainer,
    tensor::R2lTensor,
    utils::rollout_buffer::{Advantages, Returns},
};

use crate::HookResult;

pub struct PPOBatchData<T: R2lTensor> {
    pub logp: T,
    pub values_pred: T,
    pub logp_diff: T,
    pub ratio: T,
}

impl PPOBatchData<candle_core::Tensor> {
    pub fn clip_fraction(&self, clip_range: f32) -> candle_core::Result<f32> {
        (&self.ratio - 1.)?
            .abs()?
            .gt(clip_range)?
            .to_dtype(candle_core::DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()
    }

    pub fn approx_kl(&self) -> candle_core::Result<f32> {
        let ratio = self.ratio.detach();
        let log_ratio = self.logp_diff.detach();
        ratio
            .sub(&candle_core::Tensor::ones_like(&ratio)?)?
            .sub(&log_ratio)?
            .mean_all()?
            .to_scalar::<f32>()
    }
}

impl<B: Backend> PPOBatchData<burn::Tensor<B, 1>> {
    pub fn clip_fraction(&self, clip_range: f32) -> f32 {
        let ratio: Vec<f32> = self.ratio.to_data().to_vec().unwrap();
        ratio
            .iter()
            .filter(|value| (**value - 1.).abs() > clip_range)
            .count() as f32
            / ratio.len() as f32
    }

    pub fn approx_kl(&self) -> f32 {
        let ratio: Vec<f32> = self.ratio.to_data().to_vec().unwrap();
        let log_ratio: Vec<f32> = self.logp_diff.to_data().to_vec().unwrap();
        ratio
            .iter()
            .zip(log_ratio.iter())
            .map(|(ratio, log_ratio)| (ratio - 1.) - log_ratio)
            .sum::<f32>()
            / ratio.len() as f32
    }
}

pub struct PPOParams<M: ModuleWithValueFunction> {
    pub module: M,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

pub trait PPOTensorOps: R2lTensor {
    fn calculate_logp_diff(logp: &Self, logp_old: &Self) -> anyhow::Result<Self>;
    fn calculate_ratio(logp_diff: &Self) -> anyhow::Result<Self>;
    fn calculate_policy_loss(
        ratio: &Self,
        advantages: &Self,
        clip_range: f32,
    ) -> anyhow::Result<Self>;
    fn calculate_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self>;
}

pub trait PPOLosses<T> {
    fn ppo_losses(policy_loss: T, value_loss: T) -> Self;
}

pub trait PPOModule: ModuleWithValueFunction {
    fn ppo_batch_data_and_losses(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
        advantages: &Self::Tensor,
        logp_old: &Self::Tensor,
        returns: &Self::Tensor,
        clip_range: f32,
    ) -> anyhow::Result<(Self::Losses, PPOBatchData<Self::Tensor>)>;
}

impl<M> PPOModule for M
where
    M: ModuleWithValueFunction,
    M::Policy: Policy<Tensor = M::Tensor>,
    M::ValueFunction: ValueFunction<Tensor = M::Tensor>,
    M::Tensor: PPOTensorOps,
    M::Losses: PPOLosses<M::Tensor>,
{
    fn ppo_batch_data_and_losses(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
        advantages: &Self::Tensor,
        logp_old: &Self::Tensor,
        returns: &Self::Tensor,
        clip_range: f32,
    ) -> anyhow::Result<(Self::Losses, PPOBatchData<Self::Tensor>)> {
        let logp = self.get_policy().log_probs(observations, actions)?;
        let values_pred = self.value_func().calculate_values(observations)?;
        let value_loss = M::Tensor::calculate_value_loss(returns, &values_pred)?;
        let logp_diff = M::Tensor::calculate_logp_diff(&logp, logp_old)?;
        let ratio = M::Tensor::calculate_ratio(&logp_diff)?;
        let policy_loss = M::Tensor::calculate_policy_loss(&ratio, advantages, clip_range)?;
        let losses = M::Losses::ppo_losses(policy_loss, value_loss);
        let ppo_data = PPOBatchData {
            logp,
            values_pred,
            logp_diff,
            ratio,
        };
        Ok((losses, ppo_data))
    }
}

impl PPOTensorOps for candle_core::Tensor {
    fn calculate_logp_diff(logp: &Self, logp_old: &Self) -> anyhow::Result<Self> {
        Ok((logp - logp_old)?)
    }

    fn calculate_ratio(logp_diff: &Self) -> anyhow::Result<Self> {
        Ok(logp_diff.exp()?)
    }

    fn calculate_policy_loss(
        ratio: &Self,
        advantages: &Self,
        clip_range: f32,
    ) -> anyhow::Result<Self> {
        let clip_adv = (ratio.clamp(1. - clip_range, 1. + clip_range)? * advantages.clone())?;
        Ok(candle_core::Tensor::minimum(&(ratio * advantages)?, &clip_adv)?
            .neg()?
            .mean_all()?)
    }

    fn calculate_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self> {
        Ok(returns.sub(values_pred)?.sqr()?.mean_all()?)
    }
}

impl<B: AutodiffBackend> PPOTensorOps for burn::Tensor<B, 1> {
    fn calculate_logp_diff(logp: &Self, logp_old: &Self) -> anyhow::Result<Self> {
        Ok(logp.clone() - logp_old.clone())
    }

    fn calculate_ratio(logp_diff: &Self) -> anyhow::Result<Self> {
        Ok(logp_diff.clone().exp())
    }

    fn calculate_policy_loss(
        ratio: &Self,
        advantages: &Self,
        clip_range: f32,
    ) -> anyhow::Result<Self> {
        let clip_adv = ratio
            .clone()
            .clamp(1. - clip_range, 1. + clip_range)
            * advantages.clone();
        Ok((-(ratio.clone() * advantages.clone()).min_pair(clip_adv)).mean())
    }

    fn calculate_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self> {
        let value_diff = returns.clone() - values_pred.clone();
        Ok((value_diff.clone() * value_diff).mean())
    }
}

impl PPOLosses<candle_core::Tensor> for CandleLosses {
    fn ppo_losses(policy_loss: candle_core::Tensor, value_loss: candle_core::Tensor) -> Self {
        CandleLosses::new(policy_loss, value_loss)
    }
}

impl<B: AutodiffBackend> PPOLosses<burn::Tensor<B, 1>> for BurnLosses<B> {
    fn ppo_losses(policy_loss: burn::Tensor<B, 1>, value_loss: burn::Tensor<B, 1>) -> Self {
        BurnLosses::new(policy_loss, value_loss)
    }
}

// So this could be a unified implementation. One PPOHooks trait, nice
pub trait PPOHooksTrait<M: ModuleWithValueFunction> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _agent: &mut PPOParams<M>,
        _buffers: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _buffers: &[B],
        _agent: &mut PPOParams<M>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _agent: &mut PPOParams<M>,
        _losses: &mut M::Losses,
        _data: &PPOBatchData<M::Tensor>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}
