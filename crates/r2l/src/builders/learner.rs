use burn::tensor::Device as BurnDevice;
use candle_core::Device;
use r2l_core::{
    env::EnvDescription,
    error::Error,
    models::ActivationFunction,
    networks::{MlpConfig, NetworkConfig},
    tensor::R2lTensor,
};
use r2l_distributions::learning_modules::burn_lm::{
    PolicyValueLearner as BurnPolicyValueLearner, PolicyValueOptimizer as BurnPolicyValueOptimizer,
};
use r2l_distributions::learning_modules::candle_lm::{
    PolicyValueLearner as CandlePolicyValueLearner, PolicyValueOptimizer,
};
use serde::{Deserialize, Serialize};

use crate::LearningHook;
use crate::hooks::progress::SharedTrainingProgress;
use crate::{
    OptimizerConfig,
    builders::{networks::NetworkBuilder, policy::PolicyBuilder},
};

/// Policy, value-network, and optimizer settings used to construct a learner.
#[derive(Debug, Serialize, Deserialize)]
pub struct LearnerConfig {
    /// Policy architecture and exploration settings.
    pub policy_config: PolicyBuilder,
    /// Architecture of the value network.
    pub value_network: NetworkConfig,
    /// Optimizer settings for joint or separate policy and value updates.
    pub optimizer: OptimizerConfig,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            policy_config: PolicyBuilder::default(),
            value_network: NetworkConfig::Mlp(MlpConfig {
                hidden_layers: vec![64, 64],
                activation: ActivationFunction::default(),
            }),
            optimizer: OptimizerConfig::default(),
        }
    }
}

impl LearnerConfig {
    /// Builds a Candle policy/value learner with fresh parameters and optimizer state.
    ///
    /// # Arguments
    /// * `description` - Observation and action spaces used to size the networks.
    /// * `device` - Device on which the networks and their parameters are allocated.
    ///
    /// # Errors
    /// Returns an error if the network configuration is invalid or backend initialization fails.
    pub fn build_candle_learner<T: R2lTensor>(
        &self,
        description: &EnvDescription<T>,
        device: &Device,
    ) -> Result<CandlePolicyValueLearner, Error> {
        let (policy, policy_varmap) = self
            .policy_config
            .build_candle_with_varmap(description, device)?;
        let value_varmap = match self.optimizer {
            OptimizerConfig::Joint(_) => policy_varmap.clone(),
            OptimizerConfig::Split { .. } => candle_nn::VarMap::new(),
        };
        let vb = r2l_distributions::networks::candle::seeded_var_builder(
            &value_varmap,
            candle_core::DType::F32,
            device,
        );
        let value = NetworkBuilder::new(self.value_network.clone()).build_candle(
            &description.observation_space.observation_shape(),
            1,
            &vb.pp("value"),
        )?;
        let optimizer = match &self.optimizer {
            OptimizerConfig::Joint(config) => PolicyValueOptimizer::joint(
                &policy_varmap,
                config.candle_params(),
                config.gradient_clipping.max_norm(),
            )?,
            OptimizerConfig::Split { policy, value } => PolicyValueOptimizer::split(
                &policy_varmap,
                &value_varmap,
                policy.candle_params(),
                value.candle_params(),
                policy.gradient_clipping.max_norm(),
                value.gradient_clipping.max_norm(),
            )?,
        };
        CandlePolicyValueLearner::new(policy, value, optimizer, device.clone())
    }

    /// Builds a Burn policy/value learner on the Flex device with fresh optimizer state.
    ///
    /// # Arguments
    /// * `description` - Observation and action spaces used to size the networks.
    ///
    /// # Errors
    /// Returns an error if the network configuration is invalid or policy construction fails.
    pub fn build_burn_learner<T: R2lTensor>(
        &self,
        description: &EnvDescription<T>,
    ) -> Result<BurnPolicyValueLearner, Error> {
        let device = BurnDevice::flex();
        let policy = self.policy_config.build_burn(description, &device)?;
        let value_net = NetworkBuilder::new(self.value_network.clone()).build_burn(
            &description.observation_space.observation_shape(),
            1,
            &device,
        )?;
        let optimizer = match &self.optimizer {
            OptimizerConfig::Joint(config) => BurnPolicyValueOptimizer::joint(
                &config.burn_config(),
                config.learning_rate.value(1.0),
            ),
            OptimizerConfig::Split {
                policy: policy_config,
                value,
            } => BurnPolicyValueOptimizer::split(
                &policy_config.burn_config(),
                policy_config.learning_rate.value(1.0),
                &value.burn_config(),
                value.learning_rate.value(1.0),
            ),
        };
        Ok(BurnPolicyValueLearner::new(policy, value_net, optimizer))
    }
}

/// Loss settings shared by PPO and A2C learning hooks.
#[derive(Serialize, Deserialize)]
pub struct LearningHookConfig {
    /// Whether to normalize advantages before learning.
    pub normalize_advantage: bool,
    /// Entropy bonus coefficient.
    pub entropy_coeff: f32,
    /// Value-loss coefficient.
    pub vf_coeff: f32,
}

impl Default for LearningHookConfig {
    fn default() -> Self {
        Self::new(true)
    }
}

impl LearningHookConfig {
    pub(crate) fn new(normalize_advantage: bool) -> Self {
        Self {
            normalize_advantage,
            entropy_coeff: 0.0,
            vf_coeff: 1.0,
        }
    }

    /// Builds learning hooks using the learner's optimizer schedules.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - Fresh algorithm-specific learning and reporting state.
    /// * `optimizer` - Supplies the authoritative policy and value learning-rate schedules.
    /// * `progress` - Shared training progress used to evaluate the schedules.
    pub(crate) fn build<A>(
        &self,
        algorithm: A,
        optimizer: &OptimizerConfig,
        progress: SharedTrainingProgress,
    ) -> LearningHook<A> {
        let (policy_learning_rate_schedule, value_learning_rate_schedule) =
            optimizer.learning_rate_schedules();
        LearningHook {
            normalize_advantage: self.normalize_advantage,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            policy_learning_rate_schedule,
            value_learning_rate_schedule,
            progress,
            algorithm,
        }
    }
}
