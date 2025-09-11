// TODO: all the external facing traits should be framework agnostic

use burn::{
    prelude::Backend,
    tensor::{Tensor as BurnTensor, TensorData},
};
use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor, WithDType};

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

impl DType {
    pub fn to_candle_dtype(&self) -> CandleDType {
        match self {
            DType::U8 => CandleDType::U8,
            DType::U32 => CandleDType::U32,
            DType::I64 => CandleDType::I64,
            DType::BF16 => CandleDType::BF16,
            DType::F16 => CandleDType::F16,
            DType::F32 => CandleDType::F32,
            DType::F64 => CandleDType::F64,
        }
    }

    pub fn from_candle_dtype(dtype: CandleDType) -> DType {
        match dtype {
            CandleDType::U8 => DType::U8,
            CandleDType::U32 => DType::U32,
            CandleDType::I64 => DType::I64,
            CandleDType::BF16 => DType::BF16,
            CandleDType::F16 => DType::F16,
            CandleDType::F32 => DType::F32,
            CandleDType::F64 => DType::F64,
        }
    }
}

// TODO: let's start using the DType at one point
// Also this should be called something else
// Name is also in progress for now
#[derive(Debug, Clone)]
pub struct Buffer<T: WithDType = f32> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl<B: Backend> From<Buffer> for BurnTensor<B, 1> {
    fn from(value: Buffer) -> Self {
        let device = Default::default();
        let tensor_data = TensorData::new(value.to_vec1::<f32>(), value.shape.clone());
        BurnTensor::from_data(tensor_data, &device)
    }
}

impl<B: Backend> From<BurnTensor<B, 1>> for Buffer {
    fn from(value: BurnTensor<B, 1>) -> Self {
        let data = value.to_data().to_vec().unwrap();
        let shape = vec![data.len()];
        Self {
            data,
            shape,
            dtype: DType::F32,
        }
    }
}

impl From<Buffer> for CandleTensor {
    fn from(val: Buffer) -> Self {
        val.to_candle_tensor(&Device::Cpu)
    }
}

impl From<CandleTensor> for Buffer {
    fn from(value: CandleTensor) -> Self {
        Buffer::from_candle_tensor(&value)
    }
}

impl<T: WithDType> Buffer<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    pub fn from_vec(data: Vec<T>, dtype: DType) -> Self {
        let shape = vec![data.len()];
        Self { data, shape, dtype }
    }

    pub fn to_candle_tensor(&self, device: &Device) -> CandleTensor {
        let Buffer { data, shape, .. } = self.clone();
        CandleTensor::from_vec(data, shape, device).unwrap()
    }

    // TODO: burn tensor
    pub fn from_candle_tensor(t: &CandleTensor) -> Self {
        let data = t.to_vec1().unwrap(); // TODO: handle stuff that is not 1d
        let shape = t.shape();
        let dtype = t.dtype();
        Self {
            data,
            shape: shape.clone().into_dims(),
            dtype: DType::from_candle_dtype(dtype),
        }
    }

    // TODO: implement this normally or something
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let t = self.to_candle_tensor(&Device::Cpu);
        let min_t = min.to_candle_tensor(&Device::Cpu);
        let max_t = max.to_candle_tensor(&Device::Cpu);
        Self::from_candle_tensor(&t.clamp(&min_t, &max_t).unwrap())
    }

    pub fn to_vec1<D: WithDType>(&self) -> Vec<D> {
        self.to_candle_tensor(&Device::Cpu).to_vec1().unwrap()
    }
}
