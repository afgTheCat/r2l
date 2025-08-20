use candle_core::{Device, Result};
use candle_nn::VarBuilder;
use r2l_core::{
    distributions::{
        DistributionKind, categorical_distribution::CategoricalDistribution,
        diagonal_distribution::DiagGaussianDistribution,
    },
    env::{EnvironmentDescription, Space},
};

pub enum DistributionType {
    Dynamic,
    CategoricalDistribution,
    DiagGaussianDistribution,
}

pub struct DistributionBuilder {
    pub hidden_layers: Vec<usize>,
    pub distribution_type: DistributionType,
}

impl DistributionBuilder {
    pub fn build(
        &self,
        distribution_varbuilder: &VarBuilder,
        device: &Device,
        env_description: &EnvironmentDescription,
    ) -> Result<DistributionKind> {
        let action_size = env_description.action_size();
        let observation_size = env_description.observation_size();
        let layers = &[&self.hidden_layers[..], &[action_size]].concat();
        match self.distribution_type {
            DistributionType::DiagGaussianDistribution => {
                let log_std = distribution_varbuilder.get(action_size, "log_std")?;
                Ok(DistributionKind::DiagGaussian(
                    DiagGaussianDistribution::build(
                        observation_size,
                        layers,
                        &distribution_varbuilder,
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
                    &distribution_varbuilder,
                    device.clone(),
                    "policy",
                )?,
            )),
            DistributionType::Dynamic => match env_description.action_space {
                Space::Discrete(..) => Ok(DistributionKind::Categorical(
                    CategoricalDistribution::build(
                        observation_size,
                        action_size,
                        layers,
                        &distribution_varbuilder,
                        device.clone(),
                        "policy",
                    )?,
                )),
                Space::Continous { .. } => {
                    let log_std = distribution_varbuilder.get(action_size, "log_std")?;
                    Ok(DistributionKind::DiagGaussian(
                        DiagGaussianDistribution::build(
                            observation_size,
                            layers,
                            &distribution_varbuilder,
                            log_std,
                            "policy",
                        )?,
                    ))
                }
            },
        }
    }
}
