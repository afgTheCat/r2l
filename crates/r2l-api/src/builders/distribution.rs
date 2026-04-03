use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;
use r2l_candle_lm::distributions::{
    DistributionKind, categorical_distribution::CategoricalDistribution,
    diagonal_distribution::DiagGaussianDistribution,
};
use r2l_core::env::{EnvironmentDescription, Space};

pub enum DistributionType {
    Dynamic,
    CategoricalDistribution,
    DiagGaussianDistribution,
}

#[derive(Debug, Clone, Copy)]
pub enum ActionSpaceType {
    Discrete,
    Continous,
}

pub struct DistributionBuilder {
    pub hidden_layers: Vec<usize>,
    pub distribution_type: DistributionType,
}

impl DistributionBuilder {
    pub fn build(
        &self,
        distr_varbuilder: &VarBuilder,
        device: &Device,
        observation_size: usize,
        action_size: usize,
        action_space: ActionSpaceType,
    ) -> Result<DistributionKind> {
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => {
                let log_std = distr_varbuilder.get(action_size, "log_std")?;
                Ok(DistributionKind::DiagGaussian(
                    DiagGaussianDistribution::build(
                        observation_size,
                        layers,
                        distr_varbuilder,
                        log_std,
                        "policy",
                    )?,
                ))
            }
            DistributionType::CategoricalDistribution => Ok(DistributionKind::Categorical(
                CategoricalDistribution::build(
                    observation_size,
                    action_size,
                    layers,
                    distr_varbuilder,
                    device.clone(),
                    "policy",
                )?,
            )),
            DistributionType::Dynamic => match action_space {
                ActionSpaceType::Discrete => Ok(DistributionKind::Categorical(
                    CategoricalDistribution::build(
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
                    Ok(DistributionKind::DiagGaussian(
                        DiagGaussianDistribution::build(
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

    pub fn build_with_env<T>(
        &mut self,
        distribution_varbuilder: &VarBuilder,
        device: &Device,
        env_description: &EnvironmentDescription<T>,
    ) -> Result<DistributionKind> {
        let observation_size = env_description.observation_size();
        let action_size = env_description.action_size();
        let action_space_type = match env_description.action_space {
            Space::Continous { .. } => ActionSpaceType::Continous,
            Space::Discrete(..) => ActionSpaceType::Discrete,
        };
        self.build(
            distribution_varbuilder,
            device,
            observation_size,
            action_size,
            action_space_type,
        )
    }
}
