use std::marker::PhantomData;

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

#[derive(Debug, Serialize, Deserialize)]
pub struct LearnerConfig {
    pub policy_config: PolicyBuilder,
    pub value_network: NetworkConfig,
    pub optimizer: OptimizerConfig,
}

impl LearnerConfig {
    pub(crate) fn new<T: R2lTensor>(env_description: &EnvDescription<T>) -> Result<Self, Error> {
        Ok(Self {
            policy_config: PolicyBuilder::new(env_description)?,
            value_network: NetworkConfig::Mlp(MlpConfig {
                hidden_layers: vec![64, 64],
                activation: ActivationFunction::default(),
            }),
            optimizer: OptimizerConfig::default(),
        })
    }

    pub fn build_candle_learner(&self, device: &Device) -> Result<CandlePolicyValueLearner, Error> {
        let (policy, policy_varmap) = self.policy_config.build_candle_with_varmap(device)?;
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
            &self.policy_config.observation_space.observation_shape(),
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

    pub fn build_burn_learner(&self) -> Result<BurnPolicyValueLearner, Error> {
        let device = BurnDevice::flex();
        let policy = self.policy_config.build_burn(&device)?;
        let value_net = NetworkBuilder::new(self.value_network.clone()).build_burn(
            &self.policy_config.observation_space.observation_shape(),
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
pub struct LearningHookConfig {
    pub(super) normalize_advantage: bool,
    pub(super) entropy_coeff: f32,
    pub(super) vf_coeff: f32,
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
    pub(crate) fn build<M, A>(
        &self,
        algorithm: A,
        optimizer: &OptimizerConfig,
        progress: SharedTrainingProgress,
    ) -> LearningHook<M, A> {
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
            _lm: PhantomData,
        }
    }
}
