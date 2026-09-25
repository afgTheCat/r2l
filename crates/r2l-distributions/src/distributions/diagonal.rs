use std::f32::consts::PI;

use r2l_core::{
    Shape,
    error::{Error, Result},
    models::Actor,
    rng::with_rng,
    tensor::R2lTensor,
};
use rand_distr::{Distribution, StandardNormal};

use crate::{Network, Policy2};

/// Independent Gaussian actions with network-predicted means and shared log standard deviations.
/// The supplied `[1, actions]` log-standard-deviation tensor remains differentiable.
/// Its registration with an optimizer and serialization are the caller's responsibility.
#[derive(Debug, Clone)]
pub struct DiagGaussian<N: Network> {
    mean: N,
    log_std: N::Tensor,
}

impl<N: Network> DiagGaussian<N> {
    /// Builds a Gaussian policy using an externally managed log-standard-deviation tensor.
    ///
    /// # Arguments
    /// * `mean` - Network producing a nonempty vector of action means.
    /// * `log_std` - Shared log standard deviations of shape `[1, actions]`.
    ///
    /// # Errors
    /// Returns an error for incompatible network or parameter dimensions.
    pub fn new(mean: N, log_std: N::Tensor) -> Result<Self> {
        let width = super::output_width(&mean)?;
        if log_std.to_shape().dims() != [1, width] {
            return Err(Error::invalid_parameter(
                "log_std shape",
                format!("[1, {width}]"),
                format!("{:?}", log_std.to_shape()),
            ));
        }
        Ok(Self { mean, log_std })
    }
}

impl<N: Network> Actor for DiagGaussian<N> {
    type Tensor = N::Tensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        let mean = self.mean.forward(observation)?;
        let noise: Vec<f32> = with_rng(|rng| {
            (0..mean.size())
                .map(|_| StandardNormal.sample(rng))
                .collect()
        });
        let noise = Self::Tensor::from_vec_like(noise, mean.to_shape(), &mean)?;
        Ok(mean.add(&self.log_std.exp()?.mul(&noise)?)?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        self.mean.forward(observation)
    }
}

impl<N: Network> Policy2 for DiagGaussian<N> {
    fn action_shape(&self) -> Shape {
        self.mean.output_shape()
    }

    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        let mean = self.mean.forward(observations)?;
        let shape = mean.to_shape();
        super::action_batch(&actions, shape[0], shape[1])?;
        let log_std = self.log_std.broadcast_as(&shape)?;
        let inverse_variance = log_std.mul_scalar(-2.)?.exp()?;
        Ok(actions
            .sub(&mean)?
            .sqr()?
            .mul(&inverse_variance)?
            .mul_scalar(-0.5)?
            .sub(&log_std)?
            .add_scalar(-0.5 * (2. * PI).ln())?
            .sum_dim(1)?)
    }

    fn entropy(&self, _observations: Self::Tensor) -> Result<Self::Tensor> {
        // Independent of observations; sum over action dimensions, with no batch multiplier.
        Ok(self
            .log_std
            .add_scalar(0.5 + 0.5 * (2. * PI).ln())?
            .sum_dim(1)?
            .mean()?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(Some(self.log_std.exp()?.mean()?.to_vec()?[0]))
    }
}
