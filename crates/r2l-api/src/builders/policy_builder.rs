use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use candle_core::Device;
use candle_nn::VarBuilder;
use r2l_burn::distributions::{
    BurnPolicyKind,
    categorical_distribution::CategoricalDistribution as BurnCategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution as BurnDiagGaussianDistribution,
};
use r2l_candle::distributions::{
    CandlePolicyKind,
    categorical_distribution::CategoricalDistribution as CandleCategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution as CandleDiagGaussianDistribution,
};

#[derive(Debug, Clone, Copy)]
pub enum ActionSpaceType {
    Discrete,
    Continous,
}

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
    ) -> Result<BurnPolicyKind<B>> {
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        let policy_layers = &[&[observation_size][..], &layers[..]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => Ok(BurnPolicyKind::Diag(
                BurnDiagGaussianDistribution::build(policy_layers),
            )),
            DistributionType::CategoricalDistribution => Ok(BurnPolicyKind::Categorical(
                BurnCategoricalDistribution::build(policy_layers),
            )),
            DistributionType::Dynamic => match action_space {
                ActionSpaceType::Discrete => Ok(BurnPolicyKind::Categorical(
                    BurnCategoricalDistribution::build(policy_layers),
                )),
                ActionSpaceType::Continous => Ok(BurnPolicyKind::Diag(
                    BurnDiagGaussianDistribution::build(policy_layers),
                )),
            },
        }
    }

    pub fn build_candle(
        &self,
        policy_varbuilder: &VarBuilder,
        device: &Device,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> Result<CandlePolicyKind> {
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => {
                let log_std = policy_varbuilder.get(action_size, "log_std")?;
                Ok(CandlePolicyKind::DiagGaussian(
                    CandleDiagGaussianDistribution::build(
                        observation_size,
                        layers,
                        policy_varbuilder,
                        log_std,
                        "policy",
                    )?,
                ))
            }
            DistributionType::CategoricalDistribution => Ok(CandlePolicyKind::Categorical(
                CandleCategoricalDistribution::build(
                    observation_size,
                    action_size,
                    layers,
                    policy_varbuilder,
                    device.clone(),
                    "policy",
                )?,
            )),
            DistributionType::Dynamic => match action_space {
                ActionSpaceType::Discrete => Ok(CandlePolicyKind::Categorical(
                    CandleCategoricalDistribution::build(
                        observation_size,
                        action_size,
                        layers,
                        policy_varbuilder,
                        device.clone(),
                        "policy",
                    )?,
                )),
                ActionSpaceType::Continous => {
                    let log_std = policy_varbuilder.get(action_size, "log_std")?;
                    Ok(CandlePolicyKind::DiagGaussian(
                        CandleDiagGaussianDistribution::build(
                            observation_size,
                            layers,
                            policy_varbuilder,
                            log_std,
                            "policy",
                        )?,
                    ))
                }
            },
        }
    }
}
