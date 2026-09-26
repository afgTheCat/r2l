//! Policies, networks and learners for Burn and Candle over shared tensor contracts.

pub mod distributions;
pub mod learning_modules;
pub mod networks;

pub use distributions::{
    bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical, policy::DistributionKind,
};
pub use networks::Network;
pub use r2l_core::models::{Policy, ValueFunction};
pub use r2l_core::on_policy::learning_module::OnPolicyLearner;
