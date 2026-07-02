use anyhow::{Result, bail};
use candle_core::{DType, Device, Error, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::log_softmax;
use candle_nn::{Module, ops::softmax};
use r2l_core::models::{Actor, Policy};
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

use crate::sequential::{ThreadSafeSequential, build_sequential, load_tensors_with_layer_sizes};

/// Categorical Candle policy for discrete action spaces.
///
/// This policy produces one-hot actions sampled from logits predicted by a
/// feed-forward network and implements the `r2l-core` [`Actor`] and [`Policy`]
/// traits.
#[derive(Clone, Debug)]
pub struct CategoricalDistribution {
    action_size: usize,
    logits: ThreadSafeSequential,
    device: Device,
    prefix: String,
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
    ) -> Result<Self> {
        let logits = build_sequential(observation_size, layers, vb, prefix)?;
        Ok(Self {
            action_size,
            logits,
            device,
            prefix: prefix.to_string(),
        })
    }

    /// Builds a categorical policy from safetensors bytes using a custom tensor prefix.
    pub fn from_bytes_with_prefix(bytes: &[u8], device: Device, prefix: &str) -> Self {
        let (tensors, observation_size, layers) =
            load_tensors_with_layer_sizes(bytes, &device, prefix);
        let action_size = *layers.last().unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        Self::build(observation_size, action_size, &layers, &vb, device, prefix).unwrap()
    }

    /// Returns the Candle device used by this policy.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Returns the flattened observation size expected by this policy.
    pub fn observation_size(&self) -> usize {
        let observation_size = self.logits.layer(0).and_then(|s| s.input_size());
        match observation_size {
            Some(observation_size) => observation_size,
            None => panic!("Invalid observation_size"),
        }
    }
}

impl Actor for CategoricalDistribution {
    type Tensor = Tensor;

    fn action(&self, observation: Tensor) -> Result<Tensor> {
        assert!(
            observation.rank() == 1,
            "Observation should be a flattened tensor"
        );
        let observation = observation.unsqueeze(0)?;
        let logits = self.logits.forward(&observation)?;
        let action_probs: Vec<f32> = softmax(&logits, 1)?.squeeze(0)?.to_vec1()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = distribution.sample(&mut rand::rng());
        // TODO: there is a one_hot function in candle. Should we use it?
        let mut action_mask: Vec<f32> = vec![0.0; self.action_size];
        action_mask[action] = 1.;
        let action = Tensor::from_vec(action_mask, self.action_size, &self.device)?.detach();
        Ok(action)
    }

    fn try_serialize(&self) -> Option<Vec<u8>> {
        safetensors::serialize(self.logits.named_tensors(&self.prefix), None).ok()
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
