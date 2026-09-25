use burn::prelude::Backend;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use r2l_core::{
    env::{EnvDescription, Space},
    error::Result,
    models::ActivationFunction,
    networks::{MlpConfig, NetworkConfig},
    tensor::{R2lTensor, VecTensor},
};
use r2l_distributions::learning_modules::burn_lm::BurnDistributionKind;
use r2l_distributions::learning_modules::candle_lm::CandleDistributionKind;
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
    ) -> Result<(CandleDistributionKind, VarMap)> {
        let varmap = VarMap::new();
        let var_builder =
            r2l_distributions::networks::seeded_var_builder(&varmap, DType::F32, device);
        let policy = self.build_candle::<T>(&var_builder)?;
        Ok((policy, varmap))
    }

    pub(crate) fn build_candle<T: R2lTensor>(
        &self,
        var_builder: &VarBuilder<'_>,
    ) -> Result<CandleDistributionKind> {
        let network = super::networks::NetworkBuilder::new(self.network.clone());
        CandleDistributionKind::from_space(
            self.action_space.convert::<T>()?,
            &mut |prefix, width| {
                network.build_candle(
                    &self.observation_space.observation_shape(),
                    width,
                    &var_builder.pp(prefix),
                )
            },
            &mut |prefix, width| {
                Ok(var_builder.pp(prefix).get_with_hints(
                    (1, width),
                    "log_std",
                    candle_nn::Init::Const(f64::from(self.log_std_init)),
                )?)
            },
        )
    }

    /// Builds a Burn policy for backend `B`.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy configuration is invalid or unsupported.
    pub(crate) fn build_burn<B: Backend, T: R2lTensor>(&self) -> Result<BurnDistributionKind<B>> {
        let network = super::networks::NetworkBuilder::new(self.network.clone());
        BurnDistributionKind::from_space(
            self.action_space.convert::<T>()?,
            &mut |_, width| {
                network.build_burn::<B>(
                    &self.observation_space.observation_shape(),
                    width,
                    &Default::default(),
                )
            },
            &mut |_, width| {
                Ok(burn::module::Param::from_tensor(burn::Tensor::full(
                    [1, width],
                    self.log_std_init,
                    &Default::default(),
                )))
            },
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
                shape: vec![3, 8, 8].into(),
            },
            Space::Discrete(2),
        ))
        .unwrap()
    }

    #[test]
    fn policy_config_preserves_observation_shape() {
        let builder = image_policy_builder();
        assert_eq!(
            builder.observation_space.observation_shape().dims(),
            &[3, 8, 8]
        );
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
