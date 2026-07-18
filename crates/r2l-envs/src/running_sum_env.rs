use anyhow::Result;
use r2l_core::{env::Env, tensor::TensorData};

pub struct RunningSumEnv {}

impl Env for RunningSumEnv {
    type Tensor = TensorData;

    fn reset(&mut self, seed: u64) -> Result<Self::Tensor> {
        todo!()
    }

    fn step(&mut self, action: Self::Tensor) -> Result<r2l_core::env::Snapshot<Self::Tensor>> {
        todo!()
    }

    fn env_description(&self) -> r2l_core::prelude::EnvDescription<Self::Tensor> {
        todo!()
    }
}
