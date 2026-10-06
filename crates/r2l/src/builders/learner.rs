use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::mpsc::Sender;

use burn::backend::ndarray::NdArrayDevice;
use candle_core::Device;
use r2l_core::{error::Error, networks::NetworkConfig};
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearner as BurnPolicyValueLearner;
use r2l_distributions::learning_modules::burn_lm::PolicyValueLearner;
use r2l_distributions::learning_modules::candle_lm::{
    PolicyValueLearner as CandlePolicyValueLearner, PolicyValueOptimizer,
};
use serde::{Deserialize, Serialize};

use crate::hooks::progress::SharedTrainingProgress;
use crate::snapshot::PPOSettingsB;
use crate::{
    A2CRolloutStats, ClipRangeSchedule, LearningHook, LearningRateSchedule, PPORolloutStats,
    PPOSettings,
};
use crate::{
    BurnBackend, OptimizerConfig,
    builders::{networks::NetworkBuilder, policy::PolicyBuilder},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct LearnerConfig {
    pub policy_config: PolicyBuilder,
    pub value_network: NetworkConfig,
    pub optimizer: OptimizerConfig,
}

impl LearnerConfig {
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

    pub fn build_burn_learner(&self) -> Result<BurnPolicyValueLearner<BurnBackend>, Error> {
        let policy = self.policy_config.build_burn::<BurnBackend>()?;
        let value_net = NetworkBuilder::new(self.value_network.clone()).build_burn::<BurnBackend>(
            &self.policy_config.observation_space.observation_shape(),
            1,
            &NdArrayDevice::default(),
        )?;
        Ok(match &self.optimizer {
            OptimizerConfig::Joint(config) => BurnPolicyValueLearner::joint_with_network(
                policy,
                value_net,
                &config.burn_config(),
                config.learning_rate.value(1.0),
            ),
            OptimizerConfig::Split {
                policy: policy_config,
                value,
            } => BurnPolicyValueLearner::split_with_network(
                policy,
                value_net,
                &policy_config.burn_config(),
                policy_config.learning_rate.value(1.0),
                &value.burn_config(),
                value.learning_rate.value(1.0),
            ),
        })
    }
}

pub struct PPOConfig {
    total_epochs: NonZeroUsize,
    target_kl: Option<f32>,
    clip_range_schedule: ClipRangeSchedule,
    reporter: Option<Sender<PPORolloutStats>>,
}

pub enum AlgorithmConfiguration {
    Ppo(PPOConfig),
    A2C {
        reporter: Option<Sender<A2CRolloutStats>>,
    },
}

pub struct LearningHookConfig {
    normalize_advantage: bool,
    entropy_coeff: f32,
    vf_coeff: f32,
    policy_learning_rate_schedule: LearningRateSchedule,
    value_learning_rate_schedule: LearningRateSchedule,
}

impl LearningHookConfig {
    pub fn burn_ppo_hook(
        &self,
        algorithm: PPOSettingsB,
        progress: SharedTrainingProgress,
    ) -> LearningHook<PolicyValueLearner<BurnBackend>, PPOSettings> {
        let algorithm = algorithm.to_settings();
        LearningHook {
            normalize_advantage: self.normalize_advantage,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            policy_learning_rate_schedule: self.policy_learning_rate_schedule,
            value_learning_rate_schedule: self.value_learning_rate_schedule,
            progress,
            algorithm,
            _lm: PhantomData,
        }
    }
}
