use crate::thread_safe_sequential::{ThreadSafeSequential, build_sequential};
use anyhow::Result;
use candle_core::{Device, Error, Tensor as CandleTensor};
use candle_nn::VarBuilder;
use candle_nn::ops::log_softmax;
use candle_nn::{Module, ops::softmax};
use r2l_core::distributions::Policy;
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

#[derive(Clone, Debug)]
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
        let logits = build_sequential(input_dim, layers, vb, prefix)?;
        Ok(Self {
            action_size,
            logits,
            device,
        })
    }
}

impl Policy for CategoricalDistribution {
    type Tensor = CandleTensor;

    fn get_action(&self, observation: CandleTensor) -> Result<CandleTensor> {
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
        let action = CandleTensor::from_vec(action_mask, self.action_size, &self.device)?.detach();
        Ok(action)
    }

    fn log_probs(&self, states: &[CandleTensor], actions: &[CandleTensor]) -> Result<CandleTensor> {
        let states = CandleTensor::stack(&states, 0)?;
        let actions = CandleTensor::stack(&actions, 0)?;
        let logits = self.logits.forward(&states)?;
        let log_probs = log_softmax(&logits, 1)?;
        let log_probs = actions.mul(&log_probs)?.sum(1)?;
        Ok(log_probs)
    }

    fn std(&self) -> Result<f32> {
        todo!()
    }

    fn entropy(&self) -> Result<CandleTensor> {
        todo!()
    }
}
