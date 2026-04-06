use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    losses::PolicyValuesLosses,
    policies::{LearningModule, ValueFunction},
    sampler::buffer::TrajectoryContainer,
    tensor::R2lTensor,
    utils::rollout_buffer::{Advantages, Returns},
};

use crate::ppo2::RolloutLearningModule;
use crate::{BatchIndexIterator, HookResult, buffers_advantages_and_returns, sample};

pub trait A2CTensorOps: R2lTensor {
    fn calculate_a2c_policy_loss(logp: &Self, advantages: &Self) -> anyhow::Result<Self>;
    fn calculate_a2c_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self>;
}

impl A2CTensorOps for candle_core::Tensor {
    fn calculate_a2c_policy_loss(logp: &Self, advantages: &Self) -> anyhow::Result<Self> {
        Ok(advantages.mul(logp)?.neg()?.mean_all()?)
    }

    fn calculate_a2c_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self> {
        Ok(returns.sub(values_pred)?.sqr()?.mean_all()?)
    }
}

impl<B: AutodiffBackend> A2CTensorOps for burn::Tensor<B, 1> {
    fn calculate_a2c_policy_loss(logp: &Self, advantages: &Self) -> anyhow::Result<Self> {
        Ok((advantages.clone() * logp.clone()).neg().mean())
    }

    fn calculate_a2c_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self> {
        let value_diff = returns.clone() - values_pred.clone();
        Ok((value_diff.clone() * value_diff).mean())
    }
}

pub trait A2CModule2:
    RolloutLearningModule
    + LearningModule<Losses: PolicyValuesLosses<<Self as RolloutLearningModule>::LearningTensor>>
    + ValueFunction<Tensor = <Self as RolloutLearningModule>::LearningTensor>
{
}

impl<T> A2CModule2 for T
where
    T: RolloutLearningModule
        + LearningModule<Losses: PolicyValuesLosses<<T as RolloutLearningModule>::LearningTensor>>
        + ValueFunction<Tensor = <T as RolloutLearningModule>::LearningTensor>,
    T::LearningTensor: A2CTensorOps,
{
}

pub struct NewA2CParams {
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for NewA2CParams {
    fn default() -> Self {
        Self {
            gamma: 0.98,
            lambda: 0.8,
            sample_size: 64,
        }
    }
}

pub trait NewA2CHooksTrait<M: A2CModule2>
where
    M::LearningTensor: A2CTensorOps,
{
    fn before_learning_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _params: &mut NewA2CParams,
        _module: &mut M,
        _buffers: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct DefaultNewA2CHooks;

impl<M: A2CModule2> NewA2CHooksTrait<M> for DefaultNewA2CHooks where M::LearningTensor: A2CTensorOps {}

pub struct NewA2C<Module: A2CModule2, Hooks: NewA2CHooksTrait<Module>>
where
    Module::LearningTensor: A2CTensorOps,
{
    pub params: NewA2CParams,
    pub lm: Module,
    pub hooks: Hooks,
}

impl<Module: A2CModule2, Hooks: NewA2CHooksTrait<Module>> NewA2C<Module, Hooks>
where
    Module::LearningTensor: A2CTensorOps,
{
    fn batch_loop<B: TrajectoryContainer<Tensor = Module::InferenceTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.params.sample_size);
        let lm = &mut self.lm;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indices, Module::lifter);
            let advantages = lm.tensor_from_slice(&advantages.sample(&indices));
            let returns = lm.tensor_from_slice(&returns.sample(&indices));
            let logp = lm.get_policy().log_probs(&observations, &actions)?;
            let values_pred = lm.calculate_values(&observations)?;
            let policy_loss =
                Module::LearningTensor::calculate_a2c_policy_loss(&logp, &advantages)?;
            let value_loss =
                Module::LearningTensor::calculate_a2c_value_loss(&returns, &values_pred)?;
            let losses = Module::Losses::losses(policy_loss, value_loss);
            lm.update(losses)?;
        }
    }
}

impl<M: A2CModule2, H: NewA2CHooksTrait<M>> Agent for NewA2C<M, H>
where
    M::LearningTensor: A2CTensorOps,
{
    type Tensor = M::InferenceTensor;
    type Policy = M::InferencePolicy;

    fn policy(&self) -> Self::Policy {
        self.lm.get_inference_policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns(
            buffers,
            &self.lm,
            self.params.gamma,
            self.params.lambda,
            M::lifter,
        )?;
        crate::process_hook_result!(self.hooks.before_learning_hook(
            &mut self.params,
            &mut self.lm,
            buffers,
            &mut advantages,
            &mut returns
        ));
        self.batch_loop(buffers, &advantages, &returns)?;
        Ok(())
    }
}
