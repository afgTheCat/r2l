mod new_ppo;
pub mod old_ppo;

use candle_core::{DType, Device, Error, Tensor};
use r2l_agents::LearningModuleKind;
use r2l_agents::candle_agents::ModuleWithValueFunction;
use r2l_agents::candle_agents::ppo::{CandlePPOCore, PPOHooksTrait};
use r2l_agents::candle_agents::ppo::{HookResult, PPOBatchData};
use r2l_api::builders::agents::ppo::PPOBuilder;
use r2l_api::builders::sampler::{EnvPoolType, SamplerType};
use r2l_candle_lm::candle_rollout_buffer::{CandleRolloutBuffer, RolloutBatch};
use r2l_candle_lm::tensors::{PolicyLoss, ValueLoss};
use r2l_core::env_builder::EnvBuilderType;
use r2l_core::on_policy_algorithm::{
    DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm,
};
use r2l_core::{Algorithm, distributions::Policy, utils::rollout_buffer::Advantages};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::{any::Any, f64};

const ENV_NAME: &str = "Pendulum-v1";

pub type EventBox = Box<dyn Any + Send + Sync>;

#[derive(Debug, Default, Clone)]
pub struct PPOProgress {
    pub clip_fractions: Vec<f32>,
    pub entropy_losses: Vec<f32>,
    pub policy_losses: Vec<f32>,
    pub value_losses: Vec<f32>,
    pub clip_range: f32,
    pub approx_kl: f32,
    pub explained_variance: f32,
    pub progress: f64,
    pub std: f32,
    pub avarage_reward: f32,
    pub learning_rate: f64,
}

impl PPOProgress {
    pub fn clear(&mut self) -> Self {
        std::mem::take(self)
    }

    pub fn collect_batch_data(
        &mut self,
        ratio: &Tensor,
        entropy_loss: &Tensor,
        value_loss: &Tensor,
        policy_loss: &Tensor,
    ) -> candle_core::Result<()> {
        let clip_fraction = (ratio - 1.)?
            .abs()?
            .gt(self.clip_range)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        self.clip_fractions.push(clip_fraction);
        self.entropy_losses.push(entropy_loss.to_scalar()?);
        self.value_losses.push(value_loss.to_scalar()?);
        self.policy_losses.push(policy_loss.to_scalar()?);
        Ok(())
    }
}
