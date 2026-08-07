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
pub struct PolicyBuilder {
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
    /// Creates a policy builder with the default two-layer architecture.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the hidden layer sizes.
    pub fn with_hidden_layers(mut self, hidden_layers: Vec<usize>) -> Self {
        self.hidden_layers = hidden_layers;
        self
    }

    /// Sets the activation function used between hidden layers.
    pub fn with_activation_function(mut self, activation_function: ActivationFunction) -> Self {
        self.activation_function = activation_function;
        self
    }

    /// Sets the initial log standard deviation for Gaussian policies.
    pub fn with_log_std_init(mut self, log_std_init: f32) -> Self {
        self.log_std_init = log_std_init;
        self
    }

    /// Builds a Candle policy on `device`.
    pub fn build_candle<T: R2lTensor>(
        &self,
        observation_size: usize,
        action_space: Space<T>,
        device: &Device,
    ) -> anyhow::Result<CandlePolicyKind> {
        let (policy, _) = self.build_candle_with_varmap(observation_size, action_space, device)?;
        Ok(policy)
    }

    pub(crate) fn build_candle_with_varmap<T: R2lTensor>(
        &self,
        observation_size: usize,
        action_space: Space<T>,
        device: &Device,
    ) -> anyhow::Result<(CandlePolicyKind, VarMap)> {
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
    pub fn build_burn<B: Backend, T: R2lTensor>(
        &self,
        observation_size: usize,
        action_space: Space<T>,
    ) -> BurnPolicyKind<B> {
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
