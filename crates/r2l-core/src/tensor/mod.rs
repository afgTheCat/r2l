#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

pub trait R2lTensor: Clone + Send + Sync + Debug + 'static {
    fn to_vec(&self) -> Vec<f32>;
}

// TODO: make this more generic
pub trait R2lTensorOp: R2lTensor {
    fn calculate_logp_diff(logp: &Self, logp_old: &Self) -> anyhow::Result<Self>;
    fn calculate_ratio(logp_diff: &Self) -> anyhow::Result<Self>;
    fn calculate_policy_loss(
        ratio: &Self,
        advantages: &Self,
        clip_range: f32,
    ) -> anyhow::Result<Self>;
    fn calculate_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self>;
}

// TODO: let's start using the DType at one point
// Also this should be called something else
// Name is also in progress for now
#[derive(Debug, Clone)]
pub struct R2lBuffer {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl R2lBuffer {
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn to_data(self) -> Vec<f32> {
        self.data
    }
}

impl R2lTensor for R2lBuffer {
    fn to_vec(&self) -> Vec<f32> {
        todo!()
    }
}
