//! Prototype A2C training path that consumes trajectory batches directly.

use r2l_core::{
    buffers::TrajectoryBatch,
    error::Result,
    models::{Learner, Policy},
    on_policy::{
        algorithm::Agent, learning_module::OnPolicyLearner, losses::FromPolicyValueLosses,
    },
    tensor::R2lTensor,
};

use crate::{
    HookResult,
    on_policy_algorithms::{
        Advantages, Returns, ShuffledBatchIndices, batches_advantages_and_returns, sample,
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
pub trait A2CHook<M: OnPolicyLearner> {
    /// Runs after advantages and returns are computed and before minibatching.
    ///
    /// # Errors
    ///
    /// Returns an error if the hook cannot complete.
    fn before_learning_hook<B: TrajectoryBatch<M::InferenceTensor>>(
        &mut self,
        _params: &mut A2CParams,
        _module: &mut M,
        _batches: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        Ok(HookResult::Continue)
    }

    /// Runs after minibatch losses are computed and before the optimizer update.
    ///
    /// # Errors
    ///
    /// Returns an error if the hook cannot complete.
    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        _module: &mut M,
        _losses: &mut <M as Learner>::Losses,
        _data: &A2CBatchData<M::LearningTensor>,
    ) -> Result<HookResult> {
        Ok(HookResult::Continue)
    }

    /// Runs after all minibatches have been processed.
    ///
    /// # Errors
    ///
    /// Returns an error if the hook cannot complete.
    fn after_learning_hook<B: TrajectoryBatch<M::InferenceTensor>>(
        &mut self,
        _params: &mut A2CParams,
        _module: &mut M,
        _batches: &[B],
    ) -> Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

/// Prototype Advantage Actor-Critic algorithm over finalized trajectory batches.
pub struct A2C<Module: OnPolicyLearner, Hooks: A2CHook<Module>> {
    /// A2C hyperparameters.
    pub params: A2CParams,
    /// Learner containing policy, value function, and optimizer state.
    pub lm: Module,
    /// Hook implementation used to customize learning behavior.
    pub hooks: Hooks,
}

impl<Module: OnPolicyLearner, Hooks: A2CHook<Module>> A2C<Module, Hooks> {
    fn policy_loss<T: R2lTensor>(advantages: &T, logp: &T) -> Result<T> {
        Ok(advantages.mul(logp)?.neg()?.mean()?)
    }

    fn value_loss<T: R2lTensor>(returns: &T, values_pred: &T) -> Result<T> {
        Ok(returns.sub(values_pred)?.sqr()?.mean()?)
    }
}

struct A2CObjective;

impl A2CObjective {
    fn policy_loss<T: R2lTensor>(advantages: &T, logp: &T) -> Result<T> {
        Ok(advantages.mul(logp)?.neg()?.mean()?)
    }

    fn value_loss<T: R2lTensor>(returns: &T, values_pred: &T) -> Result<T> {
        Ok(returns.sub(values_pred)?.sqr()?.mean()?)
    }
}

impl<Module: OnPolicyLearner, Hooks: A2CHook<Module>> A2C<Module, Hooks> {
    fn batch_loop<B: TrajectoryBatch<Module::InferenceTensor>>(
        &mut self,
        batches: &[B],
        advantages: &Advantages,
        returns: &Returns,
    ) -> Result<()> {
        let mut batch_indices = ShuffledBatchIndices::new(batches, self.params.sample_size);
        let lm = &mut self.lm;
        loop {
            let Some(indices) = batch_indices.next_batch() else {
                return Ok(());
            };
            let (observations, actions) = sample(batches, &indices, Module::lifter);
            let advantages = lm.tensor_from_slice(&advantages.sample(&indices))?;
            let returns = lm.tensor_from_slice(&returns.sample(&indices))?;
            let logp = lm.policy().log_probs(&observations, &actions)?;
            let values_pred = lm.values(&observations)?;
            let policy_loss = A2CObjective::policy_loss(&advantages, &logp)?;
            let value_loss = A2CObjective::value_loss(&returns, &values_pred)?;
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
    ///
    /// # Errors
    ///
    /// Returns an error if tensor computation, a hook, or the optimizer update fails.
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

impl<M: OnPolicyLearner, H: A2CHook<M>> Agent for A2C<M, H> {
    type Tensor = M::InferenceTensor;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
        self.lm.inference_policy()
    }

    fn learn<B: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[B]) -> Result<()> {
        A2C::learn(self, buffers)
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.lm.set_learning_rate(learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use r2l_core::tensor::{R2lTensor, VecTensor};

    use super::A2CObjective;

    fn scalar(tensor: &VecTensor) -> f32 {
        tensor.to_vec().unwrap()[0]
    }

    #[test]
    fn policy_loss_matches_negative_advantage_weighted_log_probability() {
        let advantages = VecTensor::from_vec(vec![2.0, -1.0]);
        let logp = VecTensor::from_vec(vec![-0.5, -2.0]);
        let loss = A2CObjective::policy_loss(&advantages, &logp).unwrap();
        assert!((scalar(&loss) + 0.5).abs() < 1e-6);
    }

    #[test]
    fn value_loss_is_mean_squared_error() {
        let returns = VecTensor::from_vec(vec![1.0, 3.0]);
        let predictions = VecTensor::from_vec(vec![2.0, 0.0]);
        let loss = A2CObjective::value_loss(&returns, &predictions).unwrap();
        assert!((scalar(&loss) - 5.0).abs() < 1e-6);
    }
}
