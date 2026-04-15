use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use candle_core::Device;
use candle_nn::VarBuilder;
use r2l_burn::distributions::{
    BurnDistributionKind,
    categorical_distribution::CategoricalDistribution as BurnCategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution as BurnDiagGaussianDistribution,
};
use r2l_candle::distributions::{
    CandleDistributionKind,
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

pub struct PolicyDistributionBuilder {
    pub hidden_layers: Vec<usize>,
    pub distribution_type: DistributionType,
}

impl PolicyDistributionBuilder {
    pub fn build_burn<B: AutodiffBackend>(
        &self,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> Result<BurnDistributionKind<B>> {
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        let policy_layers = &[&[observation_size][..], &layers[..]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => Ok(BurnDistributionKind::Diag(
                BurnDiagGaussianDistribution::build(policy_layers),
            )),
            DistributionType::CategoricalDistribution => Ok(BurnDistributionKind::Categorical(
                BurnCategoricalDistribution::build(policy_layers),
            )),
            DistributionType::Dynamic => match action_space {
                ActionSpaceType::Discrete => Ok(BurnDistributionKind::Categorical(
                    BurnCategoricalDistribution::build(policy_layers),
                )),
                ActionSpaceType::Continous => Ok(BurnDistributionKind::Diag(
                    BurnDiagGaussianDistribution::build(policy_layers),
                )),
            },
        }
    }

    pub fn build_candle(
        &self,
        distr_varbuilder: &VarBuilder,
        device: &Device,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> Result<CandleDistributionKind> {
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => {
                let log_std = distr_varbuilder.get(action_size, "log_std")?;
                Ok(CandleDistributionKind::DiagGaussian(
                    CandleDiagGaussianDistribution::build(
                        observation_size,
                        layers,
                        distr_varbuilder,
                        log_std,
                        "policy",
                    )?,
                ))
            }
            DistributionType::CategoricalDistribution => Ok(CandleDistributionKind::Categorical(
                CandleCategoricalDistribution::build(
                    observation_size,
                    action_size,
                    layers,
                    distr_varbuilder,
                    device.clone(),
                    "policy",
                )?,
            )),
            DistributionType::Dynamic => match action_space {
                ActionSpaceType::Discrete => Ok(CandleDistributionKind::Categorical(
                    CandleCategoricalDistribution::build(
                        observation_size,
                        action_size,
                        layers,
                        distr_varbuilder,
                        device.clone(),
                        "policy",
                    )?,
                )),
                ActionSpaceType::Continous => {
                    let log_std = distr_varbuilder.get(action_size, "log_std")?;
                    Ok(CandleDistributionKind::DiagGaussian(
                        CandleDiagGaussianDistribution::build(
                            observation_size,
                            layers,
                            distr_varbuilder,
                            log_std,
                            "policy",
                        )?,
                    ))
                }
            },
        }
    }
}
