pub mod paralell_actor_critic;

use burn::{prelude::Backend, tensor::Tensor};

use crate::distributions::Distribution;

pub trait Policy<B: Backend> {
    type Dist: Distribution<B>;

    // retrieves the underlying distribution
    fn distribution(&self) -> &Self::Dist;

    // updates the policy according to the underlying thing
    fn update(&mut self, policy_loss: Tensor<B, 2>, value_loss: Tensor<B, 2>);
}
