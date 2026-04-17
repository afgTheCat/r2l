// TODO: do we even need a folder for this?
pub mod categorical_distribution;
pub mod diagonal_distribution;

use std::{f32, fmt::Debug};

use anyhow::Result;
use candle_core::Tensor as CandleTensor;
use candle_nn::VarBuilder;
use categorical_distribution::CategoricalDistribution;
use diagonal_distribution::DiagGaussianDistribution;
use r2l_core::{
    env::ActionSpaceType,
    models::{Actor, Policy},
};

#[derive(Debug, Clone)]
pub enum CandlePolicyKind {
    Categorical(CategoricalDistribution),
    DiagGaussian(DiagGaussianDistribution),
}

impl CandlePolicyKind {
    pub fn categorical(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
    ) -> Result<Self> {
        let layers = &[&hidden_layers[..], &[action_size]].concat();
        let distr = CategoricalDistribution::build(
            observation_size,
            action_size,
            layers,
            policy_varbuilder,
            policy_varbuilder.device().clone(),
            "policy",
        )?;
        Ok(Self::Categorical(distr))
    }

    pub fn diag_gaussian(
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
    ) -> Result<Self> {
        let layers = &[&hidden_layers[..], &[action_size]].concat();
        let log_std = policy_varbuilder.get(action_size, "log_std")?;
        let distr = DiagGaussianDistribution::build(
            observation_size,
            layers,
            policy_varbuilder,
            log_std,
            "policy",
        )?;
        Ok(Self::DiagGaussian(distr))
    }

    pub fn build(
        action_space: ActionSpaceType,
        policy_varbuilder: &VarBuilder,
        hidden_layers: &[usize],
        action_size: usize,
        observation_size: usize,
    ) -> Result<Self> {
        match action_space {
            ActionSpaceType::Discrete => Self::categorical(
                policy_varbuilder,
                hidden_layers,
                action_size,
                observation_size,
            ),
            ActionSpaceType::Continuous => Self::diag_gaussian(
                policy_varbuilder,
                hidden_layers,
                action_size,
                observation_size,
            ),
        }
    }
}

impl Actor for CandlePolicyKind {
    type Tensor = CandleTensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.action(observation),
            Self::DiagGaussian(diag) => diag.action(observation),
        }
    }
}

impl Policy for CandlePolicyKind {
    fn log_probs(&self, states: &[Self::Tensor], actions: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.log_probs(states, actions),
            Self::DiagGaussian(diag) => diag.log_probs(states, actions),
        }
    }

    fn entropy(&self, states: &[Self::Tensor]) -> Result<Self::Tensor> {
        match self {
            Self::Categorical(cat) => cat.entropy(states),
            Self::DiagGaussian(diag) => diag.entropy(states),
        }
    }

    fn std(&self) -> Result<f32> {
        match self {
            Self::Categorical(cat) => cat.std(),
            Self::DiagGaussian(diag) => diag.std(),
        }
    }

    fn resample_noise(&mut self) -> Result<()> {
        match self {
            Self::Categorical(cat) => cat.resample_noise(),
            Self::DiagGaussian(diag) => diag.resample_noise(),
        }
    }
}
