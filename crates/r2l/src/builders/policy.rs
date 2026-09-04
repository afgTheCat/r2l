use burn::prelude::Backend;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use r2l_burn::distributions::BurnPolicyKind;
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{
    env::{EnvDescription, Space},
    error::Result,
    models::ActivationFunction,
    tensor::{R2lTensor, VecTensor},
};
use serde::{Deserialize, Serialize};

/// Backend-independent configuration for an inference policy.
///
/// Training learner builders and inference runners both use this
/// configuration to construct the same policy architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PolicyBuilder {
    pub(crate) observation_size: usize,
    pub(crate) action_space: Space<VecTensor>,
    pub(crate) hidden_layers: Vec<usize>,
    pub(crate) activation_function: ActivationFunction,
    pub(crate) log_std_init: f32,
}

impl PolicyBuilder {
    pub(crate) fn new<T: R2lTensor>(env_description: &EnvDescription<T>) -> Result<Self> {
        Ok(Self {
            observation_size: env_description.observation_size(),
            action_space: env_description.action_space.convert()?,
            hidden_layers: vec![64, 64],
            activation_function: ActivationFunction::default(),
            log_std_init: 0.0,
        })
    }

    pub(crate) fn build_candle_with_varmap<T: R2lTensor>(
        &self,
        device: &Device,
    ) -> Result<(CandlePolicyKind, VarMap)> {
        let varmap = VarMap::new();
        let var_builder = r2l_candle::seeded_var_builder(&varmap, DType::F32, device);
        let policy = CandlePolicyKind::build(
            self.action_space.convert::<T>()?,
            &var_builder,
            &self.hidden_layers,
            self.observation_size,
            self.activation_function,
            self.log_std_init,
        )?;
        Ok((policy, varmap))
    }

    /// Builds a Burn policy for backend `B`.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy configuration is invalid or unsupported.
    pub(crate) fn build_burn<B: Backend, T: R2lTensor>(&self) -> Result<BurnPolicyKind<B>> {
        let action_space = self.action_space.convert::<T>()?;
        let action_size = action_space.size();
        let policy_layers = [
            &[self.observation_size][..],
            &self.hidden_layers,
            &[action_size],
        ]
        .concat();
        BurnPolicyKind::build(
            action_space,
            &policy_layers,
            self.activation_function,
            self.log_std_init,
        )
    }
}
