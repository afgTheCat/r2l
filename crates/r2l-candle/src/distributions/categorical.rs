use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::log_softmax;
use candle_nn::{Module, ops::softmax};
use itertools::Itertools;
use r2l_core::{
    error::{Error, InvalidParameterError, Result, TensorError},
    models::{ActivationFunction, Actor, Policy, ToSafetensors},
    rng::with_rng,
};
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;
use safetensors::serialize as st_serialize;

use crate::sequential::{Sequential, build_sequential};

/// Categorical Candle policy for discrete action spaces.
///
/// This policy produces category indices sampled from logits predicted by a
/// feed-forward network and implements the `r2l-core` [`Actor`] and [`Policy`]
/// traits.
#[derive(Clone, Debug)]
pub struct CategoricalDistribution {
    logits: Sequential,
    device: Device,
}

impl CategoricalDistribution {
    /// Builds a categorical policy network.
    ///
    /// # Errors
    ///
    /// Returns an error if the network parameters cannot be initialized.
    pub fn build(
        observation_size: usize,
        action_size: usize,
        layers: &[usize],
        vb: &VarBuilder,
        device: Device,
        prefix: &str,
        activation: ActivationFunction,
    ) -> Result<Self> {
        if layers.last().copied() != Some(action_size) {
            return Err(Error::InvalidParameter(Box::new(
                InvalidParameterError::InvalidValue {
                    name: "layers".into(),
                    expected: format!("a final layer of size {action_size}"),
                    value: format!("{layers:?}"),
                },
            )));
        }
        let logits = build_sequential(observation_size, layers, vb, prefix, activation)?;
        Ok(Self { logits, device })
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

impl Actor for CategoricalDistribution {
    type Tensor = Tensor;

    fn action(&self, observation: Tensor) -> Result<Tensor> {
        let observation = observation.unsqueeze(0)?;
        let logits = self.logits.forward(&observation)?;
        let action_probs: Vec<f32> = softmax(&logits, 1)?.squeeze(0)?.to_vec1()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = with_rng(|rng| distribution.sample(rng));
        Ok(Tensor::from_vec(vec![action as f32], 1, &self.device)?.detach())
    }

    fn mode_action(&self, observation: Tensor) -> Result<Tensor> {
        let logits = self.logits.forward(&observation.unsqueeze(0)?)?;
        let logits: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let action = logits
            .iter()
            .position_max_by(|a, b| a.total_cmp(b))
            .ok_or_else(|| TensorError::EmptyInput {
                operation: "select categorical modal action".into(),
            })?;
        Ok(Tensor::from_vec(vec![action as f32], 1, &self.device)?.detach())
    }
}

impl ToSafetensors for CategoricalDistribution {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        st_serialize(self.logits.named_tensors("policy"), None).map_err(Error::wrap)
    }
}

impl Policy for CategoricalDistribution {
    fn log_probs(&self, states: &[Tensor], actions: &[Tensor]) -> Result<Tensor> {
        debug_assert!(!states.is_empty());
        debug_assert_eq!(states.len(), actions.len());
        let states = Tensor::stack(states, 0)?;
        let actions = Tensor::stack(actions, 0)?;
        let logits = self.logits.forward(&states)?;
        let log_probs = log_softmax(&logits, 1)?;
        Ok(log_probs
            .gather(&actions.to_dtype(DType::U32)?, 1)?
            .squeeze(1)?)
    }

    fn entropy(&self, states: &[Tensor]) -> Result<Tensor> {
        debug_assert!(!states.is_empty());
        let states = Tensor::stack(states, 0)?;
        let logits = self.logits.forward(&states)?;
        let probs = softmax(&logits, 1)?;
        let log_probs = log_softmax(&logits, 1)?;
        let entropy_per_state = probs.mul(&log_probs)?.neg()?.sum(1)?;
        let entropy = entropy_per_state.mean_all()?;
        Ok(entropy)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
