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
    pub action_size: Option<usize>,
    pub observation_size: Option<usize>,
    pub action_space_type: Option<ActionSpaceType>,
}

impl DistributionBuilder {
    pub fn build(
        &self,
        distribution_varbuilder: &VarBuilder,
        device: &Device,
    ) -> Result<DistributionKind> {
        let observation_size = self.observation_size.unwrap();
        let action_size = self.observation_size.unwrap();
        let action_space = self.action_space_type.unwrap();
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => {
                let log_std = distribution_varbuilder.get(action_size, "log_std")?;
                Ok(DistributionKind::DiagGaussian(
                    DiagGaussianDistribution::build(
                        observation_size,
                        layers,
                        distribution_varbuilder,
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
                    distribution_varbuilder,
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
                        distribution_varbuilder,
                        device.clone(),
                        "policy",
                    )?,
                )),
                ActionSpaceType::Continous => {
                    let log_std = distribution_varbuilder.get(action_size, "log_std")?;
                    Ok(DistributionKind::DiagGaussian(
                        DiagGaussianDistribution::build(
                            observation_size,
                            layers,
                            distribution_varbuilder,
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
        self.observation_size = Some(env_description.action_size());
        self.observation_size = Some(env_description.observation_size());
        self.action_space_type = match env_description.action_space {
            Space::Continous { .. } => Some(ActionSpaceType::Continous),
            Space::Discrete(..) => Some(ActionSpaceType::Discrete),
        };
        self.build(distribution_varbuilder, device)
    }
}
