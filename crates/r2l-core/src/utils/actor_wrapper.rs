use std::marker::PhantomData;

use anyhow::Result;

use crate::{distributions::Actor, tensor::R2lTensor};

#[derive(Debug, Clone)]
pub struct ActorWrapper<A: Actor + Clone, T: R2lTensor> {
    policy: A,
    env: PhantomData<T>,
}

impl<D: Actor + Clone, T: R2lTensor> ActorWrapper<D, T> {
    pub fn new(policy: D) -> Self {
        Self {
            policy,
            env: PhantomData,
        }
    }
}

impl<D: Actor + Clone, T: R2lTensor> Actor for ActorWrapper<D, T>
where
    T: From<D::Tensor>,
    T: Into<D::Tensor>,
{
    type Tensor = T;

    fn get_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.policy.get_action(observation.into())?;
        Ok(action.into())
    }
}
