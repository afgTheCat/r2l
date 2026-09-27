use std::marker::PhantomData;

use crate::{error::Result, models::Actor, tensor::R2lTensor};

/// Converts environment observations to policy rows and flattens returned actions.
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
        let action = self.actor.action(D::Tensor::from_vec_and_shape(
            observation.to_vec()?,
            [1, observation.size()],
        )?)?;
        Ok(T::from_vec_and_shape(action.to_vec()?, [action.size()])?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.actor.mode_action(D::Tensor::from_vec_and_shape(
            observation.to_vec()?,
            [1, observation.size()],
        )?)?;
        Ok(T::from_vec_and_shape(action.to_vec()?, [action.size()])?)
    }
}

/// Adapts observation and action shapes while always selecting the modal action.
#[derive(Debug, Clone)]
pub struct ModeActorWrapper<A: Actor + Clone, T: R2lTensor> {
    actor: A,
    env: PhantomData<T>,
}

impl<D: Actor + Clone, T: R2lTensor> ModeActorWrapper<D, T> {
    /// Wraps an actor for deterministic evaluation.
    pub fn new(actor: D) -> Self {
        Self {
            actor,
            env: PhantomData,
        }
    }
}

impl<D: Actor + Clone, T: R2lTensor> Actor for ModeActorWrapper<D, T> {
    type Tensor = T;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        self.mode_action(observation)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let action = self.actor.mode_action(D::Tensor::from_vec_and_shape(
            observation.to_vec()?,
            [1, observation.size()],
        )?)?;
        Ok(T::from_vec_and_shape(action.to_vec()?, [action.size()])?)
    }
}
