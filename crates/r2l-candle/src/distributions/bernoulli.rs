use anyhow::{Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder, ops::sigmoid};
use r2l_core::{
    models::{ActivationFunction, Actor, Policy},
    rng::with_rng,
};
use rand::RngExt;

use crate::sequential::{Sequential, build_sequential};

/// Bernoulli Candle policy for Gymnasium `MultiBinary` action spaces.
#[derive(Clone, Debug)]
pub struct BernoulliDistribution {
    logits: Sequential,
    action_size: usize,
    device: Device,
}

impl BernoulliDistribution {
    /// Builds a Bernoulli policy network.
    pub fn build(
        observation_size: usize,
        action_size: usize,
        hidden_layers: &[usize],
        vb: &VarBuilder,
        device: Device,
        prefix: &str,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let layers = &[hidden_layers, &[action_size]].concat();
        let logits = build_sequential(observation_size, layers, vb, prefix, activation)?;
        Ok(Self {
            logits,
            action_size,
            device,
        })
    }

    /// Returns the Candle device used by this policy.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Returns the flattened observation size expected by this policy.
    pub fn observation_size(&self) -> usize {
        self.logits.input_size()
    }
}

impl Actor for BernoulliDistribution {
    type Tensor = Tensor;

    fn action(&self, observation: Tensor) -> Result<Tensor> {
        assert!(
            observation.rank() == 1,
            "Observation should be a flattened tensor"
        );
        let observation = observation.unsqueeze(0)?;
        let logits = self.logits.forward(&observation)?;
        let probs: Vec<f32> = sigmoid(&logits.squeeze(0)?)?.to_vec1()?;
        let actions = probs
            .into_iter()
            .map(|prob| {
                if with_rng(|rng| rng.random::<f32>()) < prob {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        Ok(Tensor::from_vec(actions, self.action_size, &self.device)?.detach())
    }
}

impl Policy for BernoulliDistribution {
    fn log_probs(&self, states: &[Tensor], actions: &[Tensor]) -> Result<Tensor> {
        let states = Tensor::stack(states, 0)?;
        let actions = Tensor::stack(actions, 0)?;
        let logits = self.logits.forward(&states)?;
        let probs = sigmoid(&logits)?.clamp(1e-6, 1. - 1e-6)?;
        let log_probs =
            (actions.mul(&probs.log()?)? + (1. - &actions)?.mul(&(1. - &probs)?.log()?)?)?;
        Ok(log_probs.sum(1)?)
    }

    fn entropy(&self, states: &[Tensor]) -> Result<Tensor> {
        let states = Tensor::stack(states, 0)?;
        let logits = self.logits.forward(&states)?;
        let probs = sigmoid(&logits)?.clamp(1e-6, 1. - 1e-6)?;
        let entropy_per_bit =
            (probs.mul(&probs.log()?)? + (1. - &probs)?.mul(&(1. - &probs)?.log()?)?)?;
        Ok(entropy_per_bit.neg()?.sum(1)?.mean_all()?)
    }

    fn std(&self) -> Result<f32> {
        bail!("standard deviation is not defined for Bernoulli distributions")
    }
}
