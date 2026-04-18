use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use candle_nn::VarBuilder;
use r2l_burn::distributions::{
    PolicyKind, categorical_distribution::CategoricalDistribution as BurnCategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution as BurnDiagGaussianDistribution,
};
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::env::ActionSpaceType;

pub enum DistributionType {
    Dynamic,
    CategoricalDistribution,
    DiagGaussianDistribution,
}

pub struct PolicyBuilder {
    pub hidden_layers: Vec<usize>,
    pub distribution_type: DistributionType,
}

impl PolicyBuilder {
    pub fn build_burn<B: AutodiffBackend>(
        &self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> Result<PolicyKind<B>> {
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        let policy_layers = &[&[observation_size][..], &layers[..]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => Ok(PolicyKind::Diag(
                BurnDiagGaussianDistribution::build(policy_layers),
            )),
            DistributionType::CategoricalDistribution => Ok(PolicyKind::Categorical(
                BurnCategoricalDistribution::build(policy_layers),
            )),
            DistributionType::Dynamic => match action_space {
                ActionSpaceType::Discrete => Ok(PolicyKind::Categorical(
                    BurnCategoricalDistribution::build(policy_layers),
                )),
                ActionSpaceType::Continuous => Ok(PolicyKind::Diag(
                    BurnDiagGaussianDistribution::build(policy_layers),
                )),
            },
        }
    }

    pub fn build_candle(
        &self,
        policy_varbuilder: &VarBuilder,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> Result<CandlePolicyKind> {
        match self.distribution_type {
            DistributionType::CategoricalDistribution => CandlePolicyKind::categorical(
                policy_varbuilder,
                &self.hidden_layers,
                action_size,
                observation_size,
            ),
            DistributionType::DiagGaussianDistribution => CandlePolicyKind::diag_gaussian(
                policy_varbuilder,
                &self.hidden_layers,
                action_size,
                observation_size,
            ),
            DistributionType::Dynamic => CandlePolicyKind::build(
                action_space,
                policy_varbuilder,
                &self.hidden_layers,
                action_size,
                observation_size,
            ),
        }
    }
}
