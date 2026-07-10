use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use r2l_core::models::{ActivationFunction, Actor, Policy};

use crate::sequential::{Sequential, build_sequential};

/// Multi-categorical Candle policy for Gymnasium `MultiDiscrete` action spaces.
#[derive(Clone, Debug)]
pub struct MultiCategoricalDistribution {
    nvec: Vec<usize>,
    logits: Sequential,
    device: Device,
}

impl MultiCategoricalDistribution {
    /// Builds a policy network whose output is split according to `nvec`.
    pub fn build(
        observation_size: usize,
        nvec: Vec<usize>,
        hidden_layers: &[usize],
        vb: &VarBuilder,
        device: Device,
        prefix: &str,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let logits_size = nvec.iter().sum();
        let layers = &[hidden_layers, &[logits_size]].concat();
        let logits = build_sequential(observation_size, layers, vb, prefix, activation)?;
        Ok(Self {
            nvec,
            logits,
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

    /// Returns the number of discrete action dimensions.
    pub fn action_size(&self) -> usize {
        self.nvec.len()
    }
}

impl Actor for MultiCategoricalDistribution {
    type Tensor = Tensor;

    fn action(&self, _observation: Tensor) -> Result<Tensor> {
        todo!();
    }
}

impl Policy for MultiCategoricalDistribution {
    fn log_probs(&self, _states: &[Tensor], _actions: &[Tensor]) -> Result<Tensor> {
        todo!();
    }

    fn entropy(&self, _states: &[Tensor]) -> Result<Tensor> {
        todo!();
    }

    fn std(&self) -> Result<f32> {
        todo!();
    }
}
