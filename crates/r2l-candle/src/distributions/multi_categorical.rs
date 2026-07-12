use anyhow::{Result, bail};
use candle_core::{DType, Device, Error, Tensor};
use candle_nn::{
    Module, VarBuilder,
    ops::{log_softmax, softmax},
};
use r2l_core::{
    env::action_ranges,
    models::{ActivationFunction, Actor, Policy},
    rng::with_rng,
};
use rand::distr::Distribution as RandDistribution;
use rand::distr::weighted::WeightedIndex;

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
}

impl Actor for MultiCategoricalDistribution {
    type Tensor = Tensor;

    fn action(&self, observation: Tensor) -> Result<Tensor> {
        assert!(
            observation.rank() == 1,
            "Observation should be a flattened tensor"
        );
        let observation = observation.unsqueeze(0)?;
        let logits = self.logits.forward(&observation)?;
        let logits = logits.squeeze(0)?;
        let mut actions = Vec::new();
        for (offset, choices) in action_ranges(&self.nvec) {
            let probs: Vec<f32> = softmax(&logits.narrow(0, offset, choices)?, 0)?.to_vec1()?;
            let distribution = WeightedIndex::new(&probs).map_err(Error::wrap)?;
            let action = with_rng(|rng| distribution.sample(rng));
            actions.push(action as f32);
        }
        Ok(Tensor::from_vec(actions, self.nvec.len(), &self.device)?.detach())
    }
}

impl Policy for MultiCategoricalDistribution {
    fn log_probs(&self, states: &[Tensor], actions: &[Tensor]) -> Result<Tensor> {
        let states = Tensor::stack(states, 0)?;
        let actions = Tensor::stack(actions, 0)?;
        let logits = self.logits.forward(&states)?;
        let mut selected_log_probs = Vec::new();
        for (action_idx, (offset, choices)) in action_ranges(&self.nvec).enumerate() {
            let logits = logits.narrow(1, offset, choices)?;
            let log_probs = log_softmax(&logits, 1)?;
            let action = actions.narrow(1, action_idx, 1)?.to_dtype(DType::U32)?;
            selected_log_probs.push(log_probs.gather(&action, 1)?.squeeze(1)?);
        }
        Ok(Tensor::stack(&selected_log_probs, 0)?.sum(0)?)
    }

    fn entropy(&self, states: &[Tensor]) -> Result<Tensor> {
        let states = Tensor::stack(states, 0)?;
        let logits = self.logits.forward(&states)?;
        let mut entropies = Vec::new();
        for (offset, choices) in action_ranges(&self.nvec) {
            let logits = logits.narrow(1, offset, choices)?;
            let probs = softmax(&logits, 1)?;
            let log_probs = log_softmax(&logits, 1)?;
            entropies.push(probs.mul(&log_probs)?.neg()?.sum(1)?);
        }
        let entropy_per_state = Tensor::stack(&entropies, 0)?.sum(0)?;
        Ok(entropy_per_state.mean_all()?)
    }

    fn std(&self) -> Result<f32> {
        bail!("standard deviation is not defined for multi-categorical distributions")
    }
}
