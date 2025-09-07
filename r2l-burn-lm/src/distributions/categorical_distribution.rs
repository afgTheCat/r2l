use crate::sequential::Sequential;
use burn::{
    module::Module,
    prelude::Backend,
    tensor::{Tensor, TensorData, activation::softmax},
};
use r2l_core::distributions::Distribution;
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

#[derive(Debug, Module)]
pub struct CategoricalDistribution<B: Backend> {
    logits: Sequential<B>,
    action_size: usize,
}

impl<B: Backend> Distribution for CategoricalDistribution<B> {
    type Tensor = Tensor<B, 1>;

    fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
        let device: <B as Backend>::Device = Default::default();
        let observation: Tensor<B, 2> = observation.unsqueeze();
        let logits = self.logits.forward(observation);
        let action_probs: Vec<f32> = softmax(logits, 1).to_data().to_vec().unwrap();
        let distribution = WeightedIndex::new(&action_probs).unwrap();
        let action = distribution.sample(&mut rand::rng());
        let mut action_mask: Vec<f32> = vec![0.0; self.action_size];
        action_mask[action] = 1.;
        Ok(Tensor::from_data(
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
        todo!()
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
