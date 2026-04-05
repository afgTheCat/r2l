pub mod burn;
pub mod candle;

use anyhow::Result;
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
    sampler::buffer::TrajectoryContainer,
    tensor::{R2lTensor, R2lTensorOp},
    utils::rollout_buffer::{Advantages, Logps, Returns},
};

use crate::{BatchIndexIterator, HookResult, buffers_advantages_and_returns, logps, sample};

pub trait PPOLosses<T> {
    fn ppo_losses(policy_loss: T, value_loss: T) -> Self;
}

pub trait RolloutLearningModule {
    type InferenceTensor: R2lTensor;
    type LearningTensor: R2lTensorOp;

    type InferencePolicy: Policy<Tensor = Self::InferenceTensor>;
    type Policy: Policy<Tensor = Self::LearningTensor>;

    // convesion between the inference tensor and the learning tensor
    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor;

    // conversion between raw data and the learning tensor
    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor;

    // returns a new inference policy, possibly to run rollouts
    fn get_inference_policy(&self) -> Self::InferencePolicy;

    // returns the real policy ref, so that we may calculate intermediate stuff for learning logp
    fn get_policy(&self) -> &Self::Policy;
}

// NOTE: heavily in progress
pub trait PPOModule2:
    RolloutLearningModule
    + LearningModule<Losses: PPOLosses<<Self as RolloutLearningModule>::LearningTensor>>
    + ValueFunction<Tensor = <Self as RolloutLearningModule>::LearningTensor>
{
}

pub struct NewPPOParams {
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for NewPPOParams {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            lambda: 0.8,
            gamma: 0.98,
            sample_size: 64,
        }
    }
}

pub struct NewPPOBatchData<T: R2lTensor> {
    pub logp: T,
    pub values_pred: T,
    pub logp_diff: T,
    pub ratio: T,
}

impl NewPPOBatchData<candle_core::Tensor> {
    pub fn clip_fraction(&self, clip_range: f32) -> candle_core::Result<f32> {
        (&self.ratio - 1.)?
            .abs()?
            .gt(clip_range)?
            .to_dtype(candle_core::DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()
    }

    pub fn approx_kl(&self) -> candle_core::Result<f32> {
        let ratio = self.ratio.detach();
        let log_ratio = self.logp_diff.detach();
        ratio
            .sub(&candle_core::Tensor::ones_like(&ratio)?)?
            .sub(&log_ratio)?
            .mean_all()?
            .to_scalar::<f32>()
    }
}

pub trait NewPPOHooksTrait<M: PPOModule2> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _params: &mut NewPPOParams,
        _module: &mut M,
        _buffers: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _params: &mut NewPPOParams,
        _module: &mut M,
        _buffers: &[B],
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _params: &mut NewPPOParams,
        _module: &mut M,
        _losses: &mut <M as LearningModule>::Losses,
        _data: &NewPPOBatchData<M::LearningTensor>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct NewPPO<Module: PPOModule2, Hooks: NewPPOHooksTrait<Module>> {
    pub params: NewPPOParams,
    pub lm: Module,
    pub hooks: Hooks,
}

impl<Module: PPOModule2, Hooks: NewPPOHooksTrait<Module>> NewPPO<Module, Hooks> {
    fn batch_loop<B: TrajectoryContainer<Tensor = Module::InferenceTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.params.sample_size);
        let lm = &mut self.lm;
        loop {
            let Some(indicies) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indicies, Module::lifter);
            let advantages = lm.tensor_from_slice(&advantages.sample(&indicies));
            let logp_old = lm.tensor_from_slice(&logps.sample(&indicies));
            let returns = lm.tensor_from_slice(&returns.sample(&indicies));
            let logp = lm.get_policy().log_probs(&observations, &actions)?;
            let values_pred = lm.calculate_values(&observations)?;
            let value_loss = Module::LearningTensor::calculate_value_loss(&returns, &values_pred)?;
            let logp_diff = Module::LearningTensor::calculate_logp_diff(&logp, &logp_old)?;
            let ratio = Module::LearningTensor::calculate_ratio(&logp_diff)?;
            let policy_loss = Module::LearningTensor::calculate_policy_loss(
                &ratio,
                &advantages,
                self.params.clip_range,
            )?;
            let mut losses = Module::Losses::ppo_losses(policy_loss, value_loss);
            let ppo_data = NewPPOBatchData {
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            match self
                .hooks
                .batch_hook(&mut self.params, lm, &mut losses, &ppo_data)?
            {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
            lm.update(losses)?;
        }
    }

    fn learning_loop<B: TrajectoryContainer<Tensor = Module::InferenceTensor>>(
        &mut self,
        buffers: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> anyhow::Result<()> {
        loop {
            self.batch_loop(buffers, &advantages, &logps, &returns)?;
            let rollout_hook_res = self
                .hooks
                .rollout_hook(&mut self.params, &mut self.lm, buffers);
            crate::process_hook_result!(rollout_hook_res);
        }
    }
}

impl<M: PPOModule2, H: NewPPOHooksTrait<M>> Agent for NewPPO<M, H> {
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
        let logps = logps(buffers, &self.policy());
        self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}
