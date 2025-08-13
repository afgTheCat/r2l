use crate::env::Observation;
use burn::tensor::Tensor;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
}

#[derive(Debug, Clone)]
pub struct Buffer<T = f32> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    // maybe redundant
    pub dtype: DType,
}

impl<T> Observation for Buffer<T> {
    fn to_tensor<B: burn::prelude::Backend>(&self) -> Tensor<B, 1> {
        todo!()
    }

    fn from_tensor<B: burn::prelude::Backend>(t: Tensor<B, 1>) -> Self {
        todo!()
    }
}
