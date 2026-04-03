use crate::tensor::{R2lBuffer, R2lTensor, R2lTensorOp};
use burn::{
    prelude::Backend,
    tensor::{Tensor as BurnTensor, TensorData, backend::AutodiffBackend},
};

impl<B: Backend> From<R2lBuffer> for BurnTensor<B, 1> {
    fn from(value: R2lBuffer) -> Self {
        let device = Default::default();
        let tensor_data = TensorData::new(value.data, value.shape.clone());
        BurnTensor::from_data(tensor_data, &device)
    }
}

impl<B: Backend> From<BurnTensor<B, 1>> for R2lBuffer {
    fn from(value: BurnTensor<B, 1>) -> Self {
        let data = value.to_data().to_vec().unwrap();
        let shape = vec![data.len()];
        Self { data, shape }
    }
}

impl<B: Backend, const D: usize> R2lTensor for BurnTensor<B, D> {
    fn to_vec(&self) -> Vec<f32> {
        self.to_data().to_vec().unwrap()
    }
}

impl<B: AutodiffBackend> R2lTensorOp for burn::Tensor<B, 1> {
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
        let clip_adv = ratio.clone().clamp(1. - clip_range, 1. + clip_range) * advantages.clone();
        Ok((-(ratio.clone() * advantages.clone()).min_pair(clip_adv)).mean())
    }

    fn calculate_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self> {
        let value_diff = returns.clone() - values_pred.clone();
        Ok((value_diff.clone() * value_diff).mean())
    }
}
