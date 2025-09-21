use crate::distributions::{
    categorical_distribution::CategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution,
};
use burn::prelude::Backend;
pub mod categorical_distribution;
pub mod diagonal_distribution;

pub enum DistributionKind<B: Backend> {
    Categorical(CategoricalDistribution<B>),
    Diag(DiagGaussianDistribution<B>),
}
