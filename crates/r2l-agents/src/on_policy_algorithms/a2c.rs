use anyhow::Result;
use r2l_core::{
    buffers::TrajectoryContainer,
    models::Policy,
    on_policy::{
        algorithm::Agent, learning_module::OnPolicyLearningModule, losses::PolicyValuesLosses,
    },
    tensor::R2lTensorMath,
};

use crate::{
    HookResult,
    on_policy_algorithms::{
        Advantages, BatchIndexIterator, Returns, buffers_advantages_and_returns, sample,
    },
};

pub struct A2CParams {
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for A2CParams {
    fn default() -> Self {
        Self {
            gamma: 0.98,
            lambda: 0.8,
            sample_size: 64,
        }
    }
}

pub trait A2CHook<M: OnPolicyLearningModule> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = M::InferenceTensor>>(
        &mut self,
        _params: &mut A2CParams,
        _module: &mut M,
        _buffers: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct A2C<Module: OnPolicyLearningModule, Hooks: A2CHook<Module>> {
    pub params: A2CParams,
    pub lm: Module,
    pub hooks: Hooks,
}

impl<Module: OnPolicyLearningModule, Hooks: A2CHook<Module>> A2C<Module, Hooks> {
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
            let policy_loss = advantages.mul(&logp)?.neg()?.mean()?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean()?;
            let losses = Module::Losses::losses(policy_loss, value_loss);
            lm.update(losses)?;
        }
    }
}

impl<M: OnPolicyLearningModule, H: A2CHook<M>> Agent for A2C<M, H> {
    type Tensor = M::InferenceTensor;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
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
