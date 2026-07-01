use std::marker::PhantomData;

use anyhow::Result;

use crate::{models::Actor, tensor::R2lTensor};

#[derive(Debug, Clone)]
pub struct ActorWrapper<A: Actor + Clone, T: R2lTensor> {
    actor: A,
    env: PhantomData<T>,
}

impl<D: Actor + Clone, T: R2lTensor> ActorWrapper<D, T> {
    pub fn new(actor: D) -> Self {
        Self {
            actor,
            env: PhantomData,
        }
    }
}

impl<D: Actor + Clone, T: R2lTensor> Actor for ActorWrapper<D, T> {
    type Tensor = T;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.actor.action(D::Tensor::convert(&observation))?;
        Ok(T::convert(&action))
    }
}
