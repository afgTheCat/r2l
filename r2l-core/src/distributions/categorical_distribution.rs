use super::{Distribution, ThreadSafeSequential};
use crate::utils::build_sequential::build_sequential;
use candle_core::{Device, Error, Result, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::log_softmax;
use candle_nn::{Module, ops::softmax};
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

pub struct CategoricalDistribution {
    action_size: usize,
    logits: ThreadSafeSequential,
    device: Device,
}

impl CategoricalDistribution {
    pub fn new(action_size: usize, logits: ThreadSafeSequential, device: Device) -> Self {
        Self {
            action_size,
            logits,
            device,
        }
    }

    pub fn build(
        input_dim: usize,
        action_size: usize,
        layers: &[usize],
        vb: &VarBuilder,
        device: Device,
        prefix: &str,
    ) -> Result<Self> {
        let (logits, _) = build_sequential(input_dim, layers, vb, prefix)?;
        Ok(Self {
            action_size,
            logits: ThreadSafeSequential(logits),
            device,
        })
    }
}

impl Distribution for CategoricalDistribution {
    fn get_action(&self, observation: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.logits.forward(observation)?;
        let action_probs: Vec<f32> = softmax(&logits, 1)?.squeeze(0)?.to_vec1()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = distribution.sample(&mut rand::rng());
        let mut action_mask: Vec<f32> = vec![0.0; self.action_size];
        action_mask[action] = 1.;
        let action = Tensor::from_vec(action_mask, self.action_size, &self.device)?.detach();
        let logp = self.log_probs(observation, &action)?.detach();
        Ok((action, logp))
    }

    fn log_probs(&self, states: &Tensor, actions: &Tensor) -> Result<Tensor> {
        let log_probs = log_softmax(&self.logits.forward(states)?, 1)?;
        actions.mul(&log_probs)?.sum(1)
    }

    fn std(&self) -> Result<f32> {
        todo!()
    }

    fn entropy(&self) -> Result<Tensor> {
        todo!()
    }
}
