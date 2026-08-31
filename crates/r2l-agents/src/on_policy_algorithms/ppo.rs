//! Prototype PPO training path that consumes trajectory batches directly.

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
        Advantages, Logps, Returns, ShuffledBatchIndices, batches_advantages_and_returns, logps,
        sample,
    },
};

/// Hyperparameters controlling PPO training behavior.
#[derive(Debug)]
pub struct PPOParams {
    /// Clipping range applied to the PPO policy ratio.
    pub clip_range: f32,
    /// Discount factor used for return and advantage estimation.
    pub gamma: f32,
    /// GAE lambda used for advantage estimation.
    pub lambda: f32,
    /// Minibatch size used during each PPO epoch.
    pub sample_size: usize,
}

impl Default for PPOParams {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            lambda: 0.8,
            gamma: 0.98,
            sample_size: 64,
        }
    }
}

/// Per-minibatch data exposed to [`PPOHook::batch_hook`].
pub struct PPOBatchData<T: R2lTensor> {
    /// Sampled observations in the minibatch.
    pub observations: Vec<T>,
    /// Sampled actions in the minibatch.
    pub actions: Vec<T>,
    /// Current policy log-probabilities for the sampled actions.
    pub logp: T,
    /// Value-function predictions for the sampled observations.
    pub values_pred: T,
    /// Difference between current and old log-probabilities.
    pub logp_diff: T,
    /// Probability ratio `exp(logp_diff)` used by the PPO objective.
    pub ratio: T,
}

/// Hook interface for customizing PPO training over [`TrajectoryBatch`] inputs.
pub trait PPOHook<M: OnPolicyLearner> {
    /// Runs after advantages and returns are computed and before PPO epochs.
    ///
    /// # Errors
    ///
    /// Returns an error if the hook cannot complete.
    fn before_learning_hook<B: TrajectoryBatch<M::InferenceTensor>>(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _batches: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        Ok(HookResult::Continue)
    }

    /// Runs after each PPO epoch and controls whether another epoch is performed.
    ///
    /// # Errors
    ///
    /// Returns an error if the hook cannot complete.
    fn rollout_hook<B: TrajectoryBatch<M::InferenceTensor>>(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _batches: &[B],
    ) -> Result<HookResult> {
        Ok(HookResult::Break)
    }

    /// Runs after minibatch losses are computed and before the optimizer update.
    ///
    /// # Errors
    ///
    /// Returns an error if the hook cannot complete.
    fn batch_hook(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _losses: &mut <M as Learner>::Losses,
        _data: &PPOBatchData<M::LearningTensor>,
    ) -> Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

/// Prototype PPO variant over finalized trajectory batches.
pub struct PPO<Module: OnPolicyLearner, Hooks: PPOHook<Module>> {
    /// PPO hyperparameters.
    pub params: PPOParams,
    /// Learner containing policy, value function, and optimizer state.
    pub lm: Module,
    /// Hook implementation used to customize learning behavior.
    pub hooks: Hooks,
}

struct PPOObjective;

impl PPOObjective {
    fn policy_loss<T: R2lTensor>(ratio: &T, advantages: &T, clip_range: f32) -> Result<T> {
        let clip_ratio = ratio.clamp(1. - clip_range, 1. + clip_range)?;
        let clipped_adv = clip_ratio.mul(advantages)?;
        let ratio_adv = ratio.mul(advantages)?;
        Ok(ratio_adv.minimum(&clipped_adv)?.neg()?.mean()?)
    }

    fn value_loss<T: R2lTensor>(returns: &T, values_pred: &T) -> Result<T> {
        Ok(returns.sub(values_pred)?.sqr()?.mean()?)
    }
}

impl<Module: OnPolicyLearner, Hooks: PPOHook<Module>> PPO<Module, Hooks> {
    fn batch_loop<B: TrajectoryBatch<Module::InferenceTensor>>(
        &mut self,
        batches: &[B],
        advantages: &Advantages,
        logps: &Logps,
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
            let logp_old = lm.tensor_from_slice(&logps.sample(&indices))?;
            let returns = lm.tensor_from_slice(&returns.sample(&indices))?;
            let logp = lm.policy().log_probs(&observations, &actions)?;
            let values_pred = lm.values(&observations)?;
            let value_loss = PPOObjective::value_loss(&returns, &values_pred)?;
            let logp_diff = logp.sub(&logp_old)?;
            let ratio = logp_diff.exp()?;
            let policy_loss =
                PPOObjective::policy_loss(&ratio, &advantages, self.params.clip_range)?;
            let mut losses = Module::Losses::from_policy_value_losses(policy_loss, value_loss);
            let ppo_data = PPOBatchData {
                observations,
                actions,
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            r2l_core::return_on_hook_result!(self.hooks.batch_hook(
                &mut self.params,
                lm,
                &mut losses,
                &ppo_data
            )?);
            lm.update(losses)?;
        }
    }

    fn learning_loop<B: TrajectoryBatch<Module::InferenceTensor>>(
        &mut self,
        batches: &[B],
        advantages: &Advantages,
        returns: &Returns,
        logps: &Logps,
    ) -> Result<()> {
        loop {
            self.batch_loop(batches, advantages, logps, returns)?;
            let rollout_hook_res = self
                .hooks
                .rollout_hook(&mut self.params, &mut self.lm, batches);
            r2l_core::return_on_hook_result!(rollout_hook_res?);
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
        let actor = self.lm.inference_policy();
        let logps = logps(batches, &actor)?;
        self.learning_loop(batches, &advantages, &returns, &logps)?;
        Ok(())
    }
}

impl<M: OnPolicyLearner, H: PPOHook<M>> Agent for PPO<M, H> {
    type Tensor = M::InferenceTensor;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
        self.lm.inference_policy()
    }

    fn learn<B: TrajectoryBatch<Self::Tensor>>(&mut self, buffers: &[B]) -> Result<()> {
        PPO::learn(self, buffers)
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.lm.set_learning_rate(learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use r2l_core::tensor::{R2lTensor, VecTensor};

    use super::PPOObjective;

    fn scalar(tensor: &VecTensor) -> f32 {
        tensor.to_vec().unwrap()[0]
    }

    #[test]
    fn clipped_policy_loss_matches_hand_calculation() {
        let ratios = VecTensor::from_vec(vec![1.3, 0.7, 1.1, 0.9]);
        let advantages = VecTensor::from_vec(vec![1.0, 1.0, -1.0, -1.0]);
        let loss = PPOObjective::policy_loss(&ratios, &advantages, 0.2).unwrap();
        assert!((scalar(&loss) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn clipping_uses_the_pessimistic_surrogate_for_negative_advantages() {
        let ratios = VecTensor::from_vec(vec![0.7]);
        let advantages = VecTensor::from_vec(vec![-1.0]);
        let loss = PPOObjective::policy_loss(&ratios, &advantages, 0.2).unwrap();
        assert!((scalar(&loss) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn value_loss_is_mean_squared_error() {
        let returns = VecTensor::from_vec(vec![1.0, 2.0, 5.0]);
        let predictions = VecTensor::from_vec(vec![0.0, 4.0, 3.0]);
        let loss = PPOObjective::value_loss(&returns, &predictions).unwrap();
        assert!((scalar(&loss) - 3.0).abs() < 1e-6);
    }
}
