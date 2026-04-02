pub mod burn_ppo;
pub mod candle_ppo;

use burn::prelude::Backend;
use r2l_core::{
    policies::ModuleWithValueFunction,
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
