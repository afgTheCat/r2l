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
    type State = D::State;

    fn action(
        &self,
        observation: Self::Tensor,
        state: Option<Self::State>,
    ) -> Result<(Self::Tensor, Self::State)> {
        let (action, state) = self.actor.action(D::Tensor::convert(&observation), state)?;
        Ok((T::convert(&action), state))
    }
}
