use burn::prelude::Backend;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use r2l_burn::distributions::BurnPolicyKind;
use r2l_candle::distributions::CandlePolicyKind;
use r2l_core::{env::Space, models::ActivationFunction, tensor::R2lTensor};
use serde::{Deserialize, Serialize};

/// Backend-independent configuration for an inference policy.
///
/// Training learner builders and inference runners both use this
/// configuration to construct the same policy architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PolicyBuilder {
    pub(crate) hidden_layers: Vec<usize>,
    pub(crate) activation_function: ActivationFunction,
    pub(crate) log_std_init: f32,
}

impl Default for PolicyBuilder {
    fn default() -> Self {
        Self {
            hidden_layers: vec![64, 64],
            activation_function: ActivationFunction::default(),
            log_std_init: 0.0,
        }
    }
}

impl PolicyBuilder {
    pub(crate) fn build_candle_with_varmap<T: R2lTensor>(
        &self,
        observation_size: usize,
        action_space: Space<T>,
        device: &Device,
    ) -> r2l_core::error::Result<(CandlePolicyKind, VarMap)> {
        let varmap = VarMap::new();
        let var_builder = r2l_candle::seeded_var_builder(&varmap, DType::F32, device);
        let policy = CandlePolicyKind::build(
            action_space,
            &var_builder,
            &self.hidden_layers,
            observation_size,
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
    pub(crate) fn build_burn<B: Backend, T: R2lTensor>(
        &self,
        observation_size: usize,
        action_space: Space<T>,
    ) -> r2l_core::error::Result<BurnPolicyKind<B>> {
        let action_size = action_space.size();
        let policy_layers = [&[observation_size][..], &self.hidden_layers, &[action_size]].concat();
        BurnPolicyKind::build(
            action_space,
            &policy_layers,
            self.activation_function,
            self.log_std_init,
        )
    }
}
