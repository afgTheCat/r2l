//! Prototype A2C training path that consumes trajectory batches directly.

use anyhow::Result;
use r2l_core::{
    buffers::TrajectoryBatch,
    models::{LearningModule, Policy},
    on_policy::{
        algorithm::Agent, learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses,
    },
    tensor::R2lTensor,
};

use crate::{
    HookResult,
    on_policy_algorithms::{
        Advantages, BatchIndexIterator, Returns, batches_advantages_and_returns, sample,
    },
};

/// Hyperparameters controlling A2C training behavior.
pub struct A2CParams {
    /// Discount factor used for return and advantage estimation.
    pub gamma: f32,
    /// GAE lambda used for return and advantage estimation.
    pub lambda: f32,
    /// Minibatch size used during the learning pass.
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

/// Per-minibatch data exposed to [`A2CHook::batch_hook`].
pub struct A2CBatchData<T: R2lTensor> {
    /// Sampled observations in the minibatch.
    pub observations: Vec<T>,
    /// Sampled actions in the minibatch.
    pub actions: Vec<T>,
    /// Policy log-probabilities for the sampled actions.
    pub logp: T,
    /// Value-function predictions for the sampled observations.
    pub values_pred: T,
}

/// Hook interface for customizing A2C training over trajectory batches.
pub trait A2CHook<M: OnPolicyLearningModule> {
    fn before_learning_hook<B: TrajectoryBatch<M::InferenceTensor>>(
        &mut self,
        _params: &mut A2CParams,
        _module: &mut M,
        _batches: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        _module: &mut M,
        _losses: &mut <M as LearningModule>::Losses,
        _data: &A2CBatchData<M::LearningTensor>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<B: TrajectoryBatch<M::InferenceTensor>>(
        &mut self,
        _params: &mut A2CParams,
        _module: &mut M,
        _batches: &[B],
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

/// Prototype Advantage Actor-Critic algorithm over finalized trajectory batches.
pub struct A2C<Module: OnPolicyLearningModule, Hooks: A2CHook<Module>> {
    /// A2C hyperparameters.
    pub params: A2CParams,
    /// Learning module containing policy, value function, and optimizer state.
    pub lm: Module,
    /// Hook implementation used to customize learning behavior.
    pub hooks: Hooks,
}

impl<Module: OnPolicyLearningModule, Hooks: A2CHook<Module>> A2C<Module, Hooks> {
    fn batch_loop<B: TrajectoryBatch<Module::InferenceTensor>>(
        &mut self,
        batches: &[B],
        advantages: &Advantages,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(batches, self.params.sample_size);
        let lm = &mut self.lm;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(batches, &indices, Module::lifter);
            let advantages = lm.tensor_from_slice(&advantages.sample(&indices));
            let returns = lm.tensor_from_slice(&returns.sample(&indices));
            let logp = lm.policy().log_probs(&observations, &actions)?;
            let values_pred = lm.values(&observations)?;
            let policy_loss = advantages.mul(&logp)?.neg()?.mean()?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean()?;
            let mut losses = Module::Losses::from_policy_value_losses(policy_loss, value_loss);
            let a2c_data = A2CBatchData {
                observations,
                actions,
                logp,
                values_pred,
            };
            r2l_core::return_on_hook_result!(self.hooks.batch_hook(
                &mut self.params,
                lm,
                &mut losses,
                &a2c_data
            )?);
            lm.update(losses)?;
        }
    }

    /// Prototype learning entrypoint over finalized trajectory batches.
    pub fn learn<B: TrajectoryBatch<Module::InferenceTensor>>(
        &mut self,
        batches: &[B],
    ) -> Result<()> {
        let (mut advantages, mut returns) = batches_advantages_and_returns(
            batches,
            &self.lm,
            self.params.gamma,
            self.params.lambda,
            Module::lifter,
        )?;
        r2l_core::return_on_hook_result!(self.hooks.before_learning_hook(
            &mut self.params,
            &mut self.lm,
            batches,
            &mut advantages,
            &mut returns
        )?);
        self.batch_loop(batches, &advantages, &returns)?;
        r2l_core::return_on_hook_result!(self.hooks.after_learning_hook(
            &mut self.params,
            &mut self.lm,
            batches
        )?);
        Ok(())
    }
}

impl<M: OnPolicyLearningModule, H: A2CHook<M>> Agent for A2C<M, H> {
    type Tensor = M::InferenceTensor;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
        self.lm.inference_policy()
    }

    fn learn<B: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[B]) -> Result<()> {
        A2C::learn(self, buffers)
    }
}
