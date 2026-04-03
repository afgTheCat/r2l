pub mod burn;
pub mod candle;

use anyhow::Result;
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    policies::ValueFunction,
    sampler::buffer::TrajectoryContainer,
    tensor::{R2lTensor, R2lTensorOp},
    utils::rollout_buffer::{Advantages, Logps, Returns},
};

use crate::{BatchIndexIterator, HookResult, buffers_advantages_and_returns, logps, sample};

// NOTE: heavily in progress
pub trait PPOModule2 {
    // The tensor type returned to env
    type InferenceTensor: R2lTensorOp;
    // The tensor type used internally for learning
    type Tensor: R2lTensorOp;
    // What we need is an inference policy type (maybe actor?)
    type InferencePolicy: Policy<Tensor = Self::InferenceTensor>;
    // The policy that has autograd
    type Policy: Policy<Tensor = Self::Tensor>;
    // The value function
    type ValueFunction: ValueFunction<Tensor = Self::Tensor>;
    // The losses
    type Losses;

    fn get_inference_policy(&self) -> Self::InferencePolicy;

    fn get_policy(&self) -> &Self::Policy;

    fn update(&mut self, losses: Self::Losses) -> Result<()>;

    fn value_func(&self) -> &Self::ValueFunction;

    // TODO: to be removed
    fn tensor_from_slice(&self, slice: &[f32]) -> Self::Tensor;

    // TODO: to be removed
    fn lifter(t: &Self::InferenceTensor) -> Self::Tensor;

    fn get_losses(policy_loss: Self::Tensor, value_loss: Self::Tensor) -> Self::Losses;
}

pub struct NewPPOParams {
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

pub struct NewPPOBatchData<T: R2lTensor> {
    pub logp: T,
    pub values_pred: T,
    pub logp_diff: T,
    pub ratio: T,
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
        _losses: &mut M::Losses,
        _data: &NewPPOBatchData<M::Tensor>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct NewPPO<Module: PPOModule2, Hooks: NewPPOHooksTrait<Module>> {
    params: NewPPOParams,
    lm: Module,
    hooks: Hooks,
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
            let values_pred = lm.value_func().calculate_values(&observations)?;
            let value_loss = Module::Tensor::calculate_value_loss(&returns, &values_pred)?;
            let logp_diff = Module::Tensor::calculate_logp_diff(&logp, &logp_old)?;
            let ratio = Module::Tensor::calculate_ratio(&logp_diff)?;
            let policy_loss =
                Module::Tensor::calculate_policy_loss(&ratio, &advantages, self.params.clip_range)?;
            let mut losses = Module::get_losses(policy_loss, value_loss);
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
            self.lm.value_func(),
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
