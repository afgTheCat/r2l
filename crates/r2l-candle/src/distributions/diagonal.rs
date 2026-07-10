use std::collections::HashMap;
use std::f32;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use r2l_core::models::{ActivationFunction, Actor, Policy, PolicyMetadata};
use safetensors::serialize as st_serialize;

use crate::sequential::{Sequential, build_sequential, network_shape};

// TODO: we may want to resample the noise better than it is now
/// Diagonal-Gaussian Candle policy for continuous action spaces.
///
/// This policy predicts the mean of a Gaussian action distribution and keeps a
/// learned diagonal log-standard-deviation parameter. It implements the
/// `r2l-core` [`Actor`] and [`Policy`] traits.
#[derive(Debug, Clone)]
pub struct DiagGaussianDistribution {
    noise: Tensor,
    mu_net: Sequential,
    log_std: Tensor,
    device: Device,
}

impl DiagGaussianDistribution {
    /// Builds a diagonal-Gaussian policy network.
    pub fn build(
        observation_size: usize,
        layers: &[usize],
        vb: &VarBuilder,
        log_std: Tensor,
        prefix: &str,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let mu_net = build_sequential(observation_size, layers, vb, prefix, activation)?;
        let noise = Tensor::randn(0f32, 1., log_std.shape(), log_std.device()).unwrap();
        let device = vb.device().clone();
        Ok(Self {
            log_std,
            mu_net,
            noise,
            device,
        })
    }

    pub(crate) fn from_parts(
        tensors: HashMap<String, Tensor>,
        device: Device,
        metadata: PolicyMetadata,
    ) -> Self {
        let (observation_size, layers) = network_shape(&tensors, "policy");
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let action_size = *layers.last().unwrap();
        let log_std = vb.get(action_size, "policy.log_std").unwrap();
        Self::build(
            observation_size,
            &layers,
            &vb,
            log_std,
            "policy",
            metadata.activation,
        )
        .unwrap()
    }

    /// Returns the Candle device used by this policy.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Returns the flattened observation size expected by this policy.
    pub fn observation_size(&self) -> usize {
        self.mu_net.input_size()
    }
}

impl Actor for DiagGaussianDistribution {
    type Tensor = Tensor;

    fn action(&self, observation: Tensor) -> Result<Tensor> {
        assert!(
            observation.rank() == 1,
            "Observation should be a flattened tensor"
        );
        let observation = observation.unsqueeze(0)?;
        let mu = self
            .mu_net
            .forward(&observation.unsqueeze(0)?)?
            .squeeze(0)?;
        let std = self.log_std.exp()?.unsqueeze(0)?;
        let noise = Tensor::randn(0f32, 1., self.log_std.shape(), self.log_std.device())?;
        let action = (mu + std.mul(&noise.unsqueeze(0)?)?)?.squeeze(0)?.detach();
        Ok(action)
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        let metadata = PolicyMetadata {
            activation: self.mu_net.activation(),
        }
        .to_safetensors_metadata();
        let mut tensors = self.mu_net.named_tensors("policy");
        tensors.push(("policy.log_std".to_string(), self.log_std.clone()));
        st_serialize(tensors, Some(metadata)).ok()
    }
}

impl Policy for DiagGaussianDistribution {
    fn log_probs(&self, states: &[Tensor], actions: &[Tensor]) -> Result<Tensor> {
        let states = Tensor::stack(states, 0)?;
        let actions = Tensor::stack(actions, 0)?;
        let mu = self.mu_net.forward(&states)?;
        let std = self.log_std.exp()?.broadcast_as(mu.shape())?;
        let var = std.sqr()?;
        let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
        let log_sqrt_2pi = Tensor::full(log_sqrt_2pi, mu.shape(), mu.device())?;
        let log_probs = ((((actions - &mu)?.sqr()? / (2. * var)?)?.neg()?
            - &self.log_std.broadcast_as(mu.shape())?)?
            - log_sqrt_2pi)?;
        let log_probs = log_probs.sum(1)?;
        Ok(log_probs)
    }

    fn entropy(&self, _states: &[Tensor]) -> Result<Tensor> {
        let log_2pi_plus_1_div_2 = Tensor::full(
            0.5 * ((2. * f32::consts::PI).ln() + 1.),
            self.log_std.shape(),
            self.log_std.device(),
        )?;
        let entropy = log_2pi_plus_1_div_2.add(&self.log_std)?.sum_all()?;
        Ok(entropy)
    }

    fn std(&self) -> Result<f32> {
        let std = self.log_std.exp()?.mean_all()?.to_scalar::<f32>()?;
        Ok(std)
    }

    fn resample_noise(&mut self) -> Result<()> {
        self.noise = Tensor::randn(0f32, 1., self.noise.shape(), self.noise.device())?;
        Ok(())
    }
}
