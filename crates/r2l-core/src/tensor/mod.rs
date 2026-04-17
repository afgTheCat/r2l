#[cfg(feature = "burn")]
mod burn_tensor;

#[cfg(feature = "candle")]
mod candle_tensor;

use std::fmt::Debug;

pub trait R2lTensor: Clone + Send + Sync + Debug + 'static {
    fn to_vec(&self) -> Vec<f32>;
}

pub trait R2lTensorMath: R2lTensor {
    fn add(&self, other: &Self) -> anyhow::Result<Self>;
    fn sub(&self, other: &Self) -> anyhow::Result<Self>;
    fn mul(&self, other: &Self) -> anyhow::Result<Self>;
    fn exp(&self) -> anyhow::Result<Self>;
    fn clamp(&self, min: f32, max: f32) -> anyhow::Result<Self>;
    fn minimum(&self, other: &Self) -> anyhow::Result<Self>;
    fn neg(&self) -> anyhow::Result<Self>;
    fn mean(&self) -> anyhow::Result<Self>;
    fn sqr(&self) -> anyhow::Result<Self>;
}

#[derive(Debug, Clone)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn from_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        Self { data, shape }
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert!(shape.iter().product::<usize>() == data.len());
        Self { data, shape }
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
}

impl R2lTensor for TensorData {
    fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
}
