use crate::distributions::{
    categorical_distribution::CategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution,
};
use burn::{Tensor as BurnTensor, module::Module, prelude::Backend};
use r2l_core::distributions::Policy;
pub mod categorical_distribution;
pub mod diagonal_distribution;

#[derive(Debug)]
pub enum DistributionKind<B: Backend> {
    Categorical(CategoricalDistribution<B>),
    Diag(DiagGaussianDistribution<B>),
}

// impl<B: Backend> Policy for CategoricalDistribution<B> {
//     type Tensor = BurnTensor<B, 1>;
//
//     fn get_action(&self, observation: Self::Tensor) -> anyhow::Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn log_probs(
//         &self,
//         observations: &[Self::Tensor],
//         actions: &[Self::Tensor],
//     ) -> anyhow::Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn std(&self) -> anyhow::Result<f32> {
//         todo!()
//     }
//
//     fn entropy(&self) -> anyhow::Result<Self::Tensor> {
//         todo!()
//     }
//
//     fn resample_noise(&mut self) -> anyhow::Result<()> {
//         todo!()
//     }
// }
