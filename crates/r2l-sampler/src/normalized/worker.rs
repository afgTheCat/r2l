use std::marker::PhantomData;

use r2l_core::{buffers::Memory, models::Actor, tensor::R2lTensor};

pub enum WorkerCommand<T: R2lTensor> {
    Step(PhantomData<T>),
    SetPolicy(Box<dyn Actor<Tensor = T>>),
}

pub enum WorkerResult<T: R2lTensor> {
    Stepped(Memory<T>),
    PolicySet,
}
