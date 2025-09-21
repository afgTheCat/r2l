use crate::sequential::Sequential;
use burn::{
    module::Module,
    prelude::Backend,
    tensor::{Tensor as BurnTensor, TensorData, activation::softmax},
};
use r2l_core::distributions::Policy;
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

#[derive(Debug, Module)]
pub struct CategoricalDistribution<B: Backend> {
    logits: Sequential<B>,
    action_size: usize,
}

impl<B: Backend> CategoricalDistribution<B> {
    pub fn build(logits_layers: &[usize]) -> Self {
        let action_size = *logits_layers.last().unwrap();
        let logits: Sequential<B> = Sequential::build(logits_layers);
        Self {
            logits,
            action_size,
        }
    }
}

impl<B: Backend> Policy for CategoricalDistribution<B> {
    type Tensor = BurnTensor<B, 1>;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        let device: <B as Backend>::Device = Default::default();
        let observation: BurnTensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation);
        let action_probs: Vec<f32> = softmax(logits, 1).to_data().to_vec().unwrap();
        let distribution = WeightedIndex::new(&action_probs).unwrap();
        let action = distribution.sample(&mut rand::rng());
        let mut action_mask: Vec<f32> = vec![0.0; self.action_size];
        action_mask[action] = 1.;
        Ok(BurnTensor::from_data(
            TensorData::new(action_mask, vec![self.action_size]),
            &device,
        ))
    }

    // FIXME: check the other fixme comment for DiagGaussian
    fn log_probs(
        &self,
        states: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> anyhow::Result<Self::Tensor> {
        let states: BurnTensor<B, 2> = BurnTensor::stack(states.to_vec(), 0);
        let actions: BurnTensor<B, 2> = BurnTensor::stack(actions.to_vec(), 0);
        let logits = self.logits.forward(states);
        let log_probs = softmax(logits, 1);
        let log_probs = (actions * log_probs).sum_dim(1);
        Ok(log_probs.squeeze(1))
    }

    fn std(&self) -> anyhow::Result<f32> {
        todo!()
    }

    fn entropy(&self) -> anyhow::Result<Self::Tensor> {
        todo!()
    }

    fn resample_noise(&mut self) -> anyhow::Result<()> {
        todo!()
    }
}
