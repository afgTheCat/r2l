use burn::prelude::Backend;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use r2l_burn::distributions::BurnPolicyKind;
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    env::{EnvDescription, Space},
    error::Result,
    models::ActivationFunction,
    networks::{MlpConfig, NetworkConfig},
    tensor::{R2lTensor, VecTensor},
};
use serde::{Deserialize, Serialize};

/// Backend-independent configuration for an inference policy.
///
/// Training learner builders and inference runners both use this
/// configuration to construct the same policy architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PolicyBuilder {
    pub(crate) observation_space: Space<VecTensor>,
    pub(crate) action_space: Space<VecTensor>,
    pub(crate) network: NetworkConfig,
    pub(crate) log_std_init: f32,
}

impl PolicyBuilder {
    pub(crate) fn new<T: R2lTensor>(env_description: &EnvDescription<T>) -> Result<Self> {
        Ok(Self {
            observation_space: env_description.observation_space.convert()?,
            action_space: env_description.action_space.convert()?,
            network: NetworkConfig::Mlp(MlpConfig {
                hidden_layers: vec![64, 64],
                activation: ActivationFunction::default(),
            }),
            log_std_init: 0.0,
        })
    }

    pub(crate) fn build_candle_with_varmap<T: R2lTensor>(
        &self,
        device: &Device,
    ) -> Result<(CandlePolicyKind, VarMap)> {
        let varmap = VarMap::new();
        let var_builder = r2l_candle::seeded_var_builder(&varmap, DType::F32, device);
        let policy = self.build_candle::<T>(&var_builder)?;
        Ok((policy, varmap))
    }

    pub(crate) fn build_candle<T: R2lTensor>(
        &self,
        var_builder: &VarBuilder<'_>,
    ) -> Result<CandlePolicyKind> {
        let NetworkConfig::Mlp(config) = &self.network else {
            todo!("Candle CNN policy construction")
        };
        CandlePolicyKind::build(
            self.action_space.convert::<T>()?,
            var_builder,
            &config.hidden_layers,
            self.observation_space.size(),
            config.activation,
            self.log_std_init,
        )
    }

    /// Builds a Burn policy for backend `B`.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy configuration is invalid or unsupported.
    pub(crate) fn build_burn<B: Backend, T: R2lTensor>(&self) -> Result<BurnPolicyKind<B>> {
        BurnPolicyKind::build_with_network(
            self.action_space.convert::<T>()?,
            &self.observation_space.observation_shape(),
            &self.network,
            self.log_std_init,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_policy_builder() -> PolicyBuilder {
        PolicyBuilder::new(&EnvDescription::<VecTensor>::new(
            Space::Box {
                min: None,
                max: None,
                shape: vec![3, 8, 8],
            },
            Space::Discrete(2),
        ))
        .unwrap()
    }

    #[test]
    fn policy_config_preserves_observation_shape() {
        let builder = image_policy_builder();
        assert_eq!(builder.observation_space.observation_shape(), vec![3, 8, 8]);
        assert_eq!(builder.observation_space.size(), 192);

        let serialized = yaml_serde::to_string(&builder).unwrap();
        let restored: PolicyBuilder = yaml_serde::from_str(&serialized).unwrap();
        assert_eq!(
            restored.observation_space.observation_shape(),
            builder.observation_space.observation_shape()
        );
        assert_eq!(
            restored.observation_space.size(),
            builder.observation_space.size()
        );
    }
}
