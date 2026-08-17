use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder, ops::sigmoid};
use r2l_core::{
    error::{Error, Result},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    rng::with_rng,
};
use safetensors::serialize as st_serialize;

use crate::sequential::{Sequential, build_sequential};

/// Independent Bernoulli policy for Gymnasium `MultiBinary` action spaces.
///
/// This is equivalent to a multi-categorical policy with two categories per
/// action component, but uses one logit per component instead of two.
#[derive(Clone, Debug)]
pub struct MultiBernoulliDistribution {
    logits: Sequential,
    action_size: usize,
    device: Device,
}

impl MultiBernoulliDistribution {
    /// Builds a multi-Bernoulli policy network.
    ///
    /// # Errors
    ///
    /// Returns an error if the network parameters cannot be initialized.
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
    #[must_use]
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Returns the flattened observation size expected by this policy.
    #[must_use]
    pub fn observation_size(&self) -> usize {
        self.logits.input_size()
    }

    pub(crate) fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.logits.named_tensors(prefix)
    }
}

impl Actor for MultiBernoulliDistribution {
    type Tensor = Tensor;

    fn action(&self, observation: Tensor) -> Result<Tensor> {
        let observation = observation.unsqueeze(0)?;
        let logits = self.logits.forward(&observation)?;
        let probs: Vec<f32> = sigmoid(&logits.squeeze(0)?)?.to_vec1()?;
        let actions = probs
            .into_iter()
            .map(|prob| {
                if with_rng(rand::RngExt::random::<f32>) < prob {
                    1.
                } else {
                    0.
                }
            })
            .collect();
        Ok(Tensor::from_vec(actions, self.action_size, &self.device)?.detach())
    }

    fn mode_action(&self, observation: Tensor) -> Result<Tensor> {
        let logits = self.logits.forward(&observation.unsqueeze(0)?)?;
        let probs: Vec<f32> = sigmoid(&logits.squeeze(0)?)?.to_vec1()?;
        let actions = probs
            .into_iter()
            .map(|probability| f32::from(probability >= 0.5))
            .collect();
        Ok(Tensor::from_vec(actions, self.action_size, &self.device)?.detach())
    }
}

impl ToSafetensors for MultiBernoulliDistribution {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        st_serialize(self.named_tensors("policy"), None).map_err(Error::wrap)
    }
}

impl Policy for MultiBernoulliDistribution {
    fn log_probs(&self, states: &[Tensor], actions: &[Tensor]) -> Result<Tensor> {
        debug_assert!(!states.is_empty());
        debug_assert_eq!(states.len(), actions.len());
        let states = Tensor::stack(states, 0)?;
        let actions = Tensor::stack(actions, 0)?;
        let logits = self.logits.forward(&states)?;
        let probs = sigmoid(&logits)?.clamp(1e-6, 1. - 1e-6)?;
        let log_probs =
            (actions.mul(&probs.log()?)? + (1. - &actions)?.mul(&(1. - &probs)?.log()?)?)?;
        Ok(log_probs.sum(1)?)
    }

    fn entropy(&self, states: &[Tensor]) -> Result<Tensor> {
        debug_assert!(!states.is_empty());
        let states = Tensor::stack(states, 0)?;
        let logits = self.logits.forward(&states)?;
        let probs = sigmoid(&logits)?.clamp(1e-6, 1. - 1e-6)?;
        let entropy_per_bit =
            (probs.mul(&probs.log()?)? + (1. - &probs)?.mul(&(1. - &probs)?.log()?)?)?;
        Ok(entropy_per_bit.neg()?.sum(1)?.mean_all()?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
