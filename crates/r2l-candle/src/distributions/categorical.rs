use std::collections::HashMap;

use anyhow::{Result, bail};
use candle_core::{DType, Device, Error, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::log_softmax;
use candle_nn::{Module, ops::softmax};
use r2l_core::{
    models::{ActivationFunction, Actor, Policy, PolicyMetadata},
    rng::with_rng,
};
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;
use safetensors::serialize as st_serialize;

use crate::sequential::{Sequential, build_sequential, network_shape};

/// Categorical Candle policy for discrete action spaces.
///
/// This policy produces one-hot actions sampled from logits predicted by a
/// feed-forward network and implements the `r2l-core` [`Actor`] and [`Policy`]
/// traits.
#[derive(Clone, Debug)]
pub struct CategoricalDistribution {
    action_size: usize,
    logits: Sequential,
    device: Device,
}

impl CategoricalDistribution {
    /// Builds a categorical policy network.
    pub fn build(
        observation_size: usize,
        action_size: usize,
        layers: &[usize],
        vb: &VarBuilder,
        device: Device,
        prefix: &str,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let logits = build_sequential(observation_size, layers, vb, prefix, activation)?;
        Ok(Self {
            action_size,
            logits,
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
        Self::build(
            observation_size,
            action_size,
            &layers,
            &vb,
            device,
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
        self.logits.input_size()
    }
}

impl Actor for CategoricalDistribution {
    type Tensor = Tensor;
    type State = ();

    fn action(&self, observation: Tensor, _state: Option<()>) -> Result<(Tensor, ())> {
        assert!(
            observation.rank() == 1,
            "Observation should be a flattened tensor"
        );
        let observation = observation.unsqueeze(0)?;
        let logits = self.logits.forward(&observation)?;
        let action_probs: Vec<f32> = softmax(&logits, 1)?.squeeze(0)?.to_vec1()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = with_rng(|rng| distribution.sample(rng));
        // TODO: there is a one_hot function in candle. Should we use it?
        let mut action_mask: Vec<f32> = vec![0.0; self.action_size];
        action_mask[action] = 1.;
        let action = Tensor::from_vec(action_mask, self.action_size, &self.device)?.detach();
        Ok((action, ()))
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        let metadata = PolicyMetadata {
            activation: self.logits.activation(),
        }
        .to_safetensors_metadata();
        st_serialize(self.logits.named_tensors("policy"), Some(metadata)).ok()
    }
}

impl Policy for CategoricalDistribution {
    fn log_probs(&self, states: &[Tensor], actions: &[Tensor]) -> Result<Tensor> {
        let states = Tensor::stack(states, 0)?;
        let actions = Tensor::stack(actions, 0)?;
        let logits = self.logits.forward(&states)?;
        let log_probs = log_softmax(&logits, 1)?;
        let log_probs = actions.mul(&log_probs)?.sum(1)?;
        Ok(log_probs)
    }

    fn entropy(&self, states: &[Tensor]) -> Result<Tensor> {
        let states = Tensor::stack(states, 0)?;
        let logits = self.logits.forward(&states)?;
        let probs = softmax(&logits, 1)?;
        let log_probs = log_softmax(&logits, 1)?;
        let entropy_per_state = probs.mul(&log_probs)?.neg()?.sum(1)?;
        let entropy = entropy_per_state.mean_all()?;
        Ok(entropy)
    }

    fn std(&self) -> Result<f32> {
        bail!("standard deviation is not defined for categorical distributions")
    }
}
