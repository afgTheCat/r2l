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
use r2l_distributions::{DiagGaussian, Network, distributions::sde::SdeConfig};
use serde::{Deserialize, Serialize};

/// Backend-independent configuration for an inference policy.
///
/// Training learner builders and inference runners both use this
/// configuration to construct the same policy architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PolicyBuilder {
    pub(crate) network: NetworkConfig,
    pub(crate) log_std_init: f32,
    pub(crate) sde: SdeConfig,
}

impl Default for PolicyBuilder {
    fn default() -> Self {
        Self {
            network: NetworkConfig::Mlp(MlpConfig {
                hidden_layers: vec![64, 64],
                activation: ActivationFunction::default(),
            }),
            log_std_init: 0.0,
            sde: SdeConfig::Disabled,
        }
    }
}

impl PolicyBuilder {
    pub(crate) fn build_candle_with_varmap<T: R2lTensor>(
        &self,
        description: &EnvDescription<T>,
        device: &Device,
    ) -> Result<(CandleDistributionKind, VarMap)> {
        let varmap = VarMap::new();
        let var_builder =
            r2l_distributions::networks::candle::seeded_var_builder(&varmap, DType::F32, device);
        let policy = self.build_candle(description, &var_builder)?;
        Ok((policy, varmap))
    }

    pub(crate) fn build_candle<T: R2lTensor>(
        &self,
        description: &EnvDescription<T>,
        var_builder: &VarBuilder<'_>,
    ) -> Result<CandleDistributionKind> {
        let network = super::networks::NetworkBuilder::new(self.network.clone());
        if let config @ SdeConfig::Enabled { .. } = self.sde {
            let width = Self::sde_action_size(&description.action_space)?;
            let mean = network.build_candle(
                &description.observation_space.observation_shape(),
                width,
                &var_builder.pp("policy"),
            )?;
            let features = mean
                .feature_size()
                .expect("built-in networks expose features");
            let log_std = var_builder.pp("policy").get_with_hints(
                (features, width),
                "log_std",
                candle_nn::Init::Const(f64::from(self.log_std_init)),
            )?;
            return Ok(CandleDistributionKind::DiagGaussian(
                DiagGaussian::with_sde(mean, log_std, config)?,
            ));
        }
        CandleDistributionKind::from_space(
            description.action_space.convert::<VecTensor>()?,
            &mut |prefix, width| {
                network.build_candle(
                    &description.observation_space.observation_shape(),
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

    /// Builds a Burn policy on the given device.
    ///
    /// # Arguments
    ///
    /// * `description` - Observation and action spaces used to size the policy.
    /// * `device` - Device on which network parameters and Gaussian log deviations are initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy configuration is invalid or unsupported.
    pub(crate) fn build_burn<T: R2lTensor>(
        &self,
        description: &EnvDescription<T>,
        device: &burn::tensor::Device,
    ) -> Result<BurnDistributionKind> {
        let network = super::networks::NetworkBuilder::new(self.network.clone());
        if let config @ SdeConfig::Enabled { .. } = self.sde {
            let width = Self::sde_action_size(&description.action_space)?;
            let mean = network.build_burn(
                &description.observation_space.observation_shape(),
                width,
                device,
            )?;
            let features = mean
                .feature_size()
                .expect("built-in networks expose features");
            let log_std = burn::module::Param::from_tensor(burn::Tensor::full(
                [features, width],
                self.log_std_init,
                device,
            ));
            return Ok(BurnDistributionKind::DiagGaussian(DiagGaussian::with_sde(
                mean, log_std, config,
            )?));
        }
        BurnDistributionKind::from_space(
            description.action_space.convert::<VecTensor>()?,
            &mut |_, width| {
                network.build_burn(
                    &description.observation_space.observation_shape(),
                    width,
                    device,
                )
            },
            &mut |_, width| {
                Ok(burn::module::Param::from_tensor(burn::Tensor::full(
                    [1, width],
                    self.log_std_init,
                    device,
                )))
            },
        )
    }

    fn sde_action_size<T: R2lTensor>(action_space: &Space<T>) -> Result<usize> {
        match action_space {
            Space::Box { shape, .. } => Ok(shape.num_elements()),
            _ => Err(r2l_core::error::Error::invalid_parameter(
                "gSDE action space",
                "a continuous Box space",
                format!("{action_space:?}"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabling_sde_allows_discrete_actions_on_both_backends() -> Result<()> {
        let description = EnvDescription::<VecTensor>::new(
            Space::Box {
                min: None,
                max: None,
                shape: r2l_core::Shape::from([4]),
            },
            Space::Discrete(2),
        );
        let mut builder = PolicyBuilder::default();
        assert!(matches!(builder.sde, SdeConfig::Disabled));
        builder.sde = SdeConfig::Enabled {
            resample_every: None,
        };
        assert!(
            builder
                .build_candle_with_varmap(&description, &Device::Cpu)
                .is_err()
        );
        assert!(
            builder
                .build_burn(&description, &Default::default())
                .is_err()
        );

        builder.sde = SdeConfig::Disabled;
        assert!(
            builder
                .build_candle_with_varmap(&description, &Device::Cpu)
                .is_ok()
        );
        assert!(
            builder
                .build_burn(&description, &Default::default())
                .is_ok()
        );
        Ok(())
    }
}
