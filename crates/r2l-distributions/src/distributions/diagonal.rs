use std::f32::consts::PI;

use r2l_core::{
    Shape,
    error::{Error, Result},
    models::Actor,
    rng::with_rng,
    tensor::R2lTensor,
};
use rand_distr::{Distribution, StandardNormal};

use super::TensorParameter;
use super::sde::{SdeConfig, StateDependentNoise};
use crate::{Network, Policy};

/// Gaussian actions with network-predicted means and learned exploration scales.
/// Log standard deviations have shape `[1, actions]`, or `[features, actions]`
/// with state-dependent exploration. Pass a Burn `Param` for
/// optimizer-managed parameters, or a tensor when parameters are managed externally.
/// For Candle, obtain the tensor from the policy's `VarMap`-backed `VarBuilder`.
#[derive(Debug, Clone)]
pub struct DiagGaussian<N: Network, P: TensorParameter<N::Tensor> = <N as Network>::Tensor> {
    pub(super) mean: N,
    pub(super) log_std: P,
    pub(super) sde: Option<StateDependentNoise<N::Tensor>>,
}

impl<N: Network, P: TensorParameter<N::Tensor>> DiagGaussian<N, P> {
    /// Builds a Gaussian policy from a mean network and log-standard-deviation parameter.
    ///
    /// # Arguments
    /// * `mean` - Network producing a nonempty vector of action means.
    /// * `log_std` - Shared log standard deviations of shape `[1, actions]`. Use
    ///   `burn::module::Param::from_tensor` to include them in Burn optimizer updates.
    ///   For Candle, use a `VarBuilder` backed by the policy optimizer's `VarMap`.
    ///
    /// # Errors
    /// Returns an error for incompatible network or parameter dimensions.
    pub fn new(mean: N, log_std: P) -> Result<Self> {
        let width = super::output_width(&mean)?;
        if log_std.value().to_shape().dims() != [1, width] {
            return Err(Error::invalid_parameter(
                "log_std shape",
                format!("[1, {width}]"),
                format!("{:?}", log_std.value().to_shape()),
            ));
        }
        Ok(Self {
            mean,
            log_std,
            sde: None,
        })
    }

    /// Builds a Gaussian with state-dependent exploration from the network's hidden features.
    ///
    /// `log_std` must have shape `[features, actions]`. Features are detached in
    /// the variance calculation, matching SB3's default `learn_features=false`.
    ///
    /// # Errors
    /// Returns an error if features are unavailable or parameter dimensions do not match.
    pub fn with_sde(mean: N, log_std: P, config: SdeConfig) -> Result<Self> {
        let width = super::output_width(&mean)?;
        let features = mean
            .feature_size()
            .filter(|size| *size > 0)
            .ok_or_else(|| {
                Error::invalid_parameter("gSDE network", "exposed hidden features", "unavailable")
            })?;
        if log_std.value().to_shape().dims() != [features, width] {
            return Err(Error::invalid_parameter(
                "log_std shape",
                format!("[{features}, {width}]"),
                format!("{:?}", log_std.value().to_shape()),
            ));
        }
        Ok(Self {
            mean,
            log_std,
            sde: Some(StateDependentNoise::new(config)),
        })
    }

    fn sde_variance(&self, features: &N::Tensor) -> Result<N::Tensor> {
        Ok(features
            .detach()
            .sqr()?
            .matmul(&self.log_std.value().mul_scalar(2.)?.exp()?)?
            .add_scalar(1e-6)?)
    }
}

impl<N: Network, P: TensorParameter<N::Tensor>> Actor for DiagGaussian<N, P> {
    type Tensor = N::Tensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        if let Some(sde) = &self.sde {
            let (mean, features) = self.mean.forward_with_features(observation)?;
            return Ok(mean.add(&sde.sample(&features.detach(), &self.log_std.value())?)?);
        }
        let mean = self.mean.forward(observation)?;
        let noise: Vec<f32> = with_rng(|rng| {
            (0..mean.size())
                .map(|_| StandardNormal.sample(rng))
                .collect()
        });
        let noise = Self::Tensor::from_vec_like(noise, mean.to_shape(), &mean)?;
        Ok(mean.add(&self.log_std.value().exp()?.mul(&noise)?)?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        self.mean.forward(observation)
    }
}

impl<N: Network, P: TensorParameter<N::Tensor>> Policy for DiagGaussian<N, P> {
    fn action_shape(&self) -> Shape {
        self.mean.output_shape()
    }

    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        if self.sde.is_some() {
            let (mean, features) = self.mean.forward_with_features(observations)?;
            let shape = mean.to_shape();
            super::action_batch(&actions, shape[0], shape[1])?;
            let log_variance = self.sde_variance(&features)?.log()?;
            return Ok(actions
                .sub(&mean)?
                .sqr()?
                .mul(&log_variance.neg()?.exp()?)?
                .add(&log_variance)?
                .add_scalar((2. * PI).ln())?
                .mul_scalar(-0.5)?
                .sum_dim(1)?);
        }
        let mean = self.mean.forward(observations)?;
        let shape = mean.to_shape();
        super::action_batch(&actions, shape[0], shape[1])?;
        let log_std = self.log_std.value().broadcast_as(&shape)?;
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

    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        if self.sde.is_some() {
            let (_, features) = self.mean.forward_with_features(observations)?;
            return Ok(self
                .sde_variance(&features)?
                .log()?
                .add_scalar(1. + (2. * PI).ln())?
                .mul_scalar(0.5)?
                .sum_dim(1)?
                .mean()?);
        }
        // Independent of observations; sum over action dimensions, with no batch multiplier.
        Ok(self
            .log_std
            .value()
            .add_scalar(0.5 + 0.5 * (2. * PI).ln())?
            .sum_dim(1)?
            .mean()?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(Some(self.log_std.value().exp()?.mean()?.to_vec()?[0]))
    }
}
