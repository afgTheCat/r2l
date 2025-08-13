pub mod paralell_actor_critic;

use burn::{prelude::Backend, tensor::Tensor};

use crate::{
    distributions::Distribution,
    env::{Action, Observation},
};

pub trait Policy<B: Backend, O: Observation, A: Action> {
    type Dist: Distribution<O, A>;

    // retrieves the underlying distribution
    fn distribution(&self) -> &Self::Dist;

    // updates the policy according to the underlying thing
    fn update(&mut self, policy_loss: Tensor<B, 2>, value_loss: Tensor<B, 2>);
}
