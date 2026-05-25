//! Prototype PPO training path that consumes trajectory batches directly.

use anyhow::Result;
use r2l_core::{
    buffers::gen_buffer::TrajectoryBatchT,
    models::{LearningModule, Policy},
    on_policy::{
        algorithm::Agent, learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses,
    },
    tensor::{R2lTensor, R2lTensorMath},
};

use crate::{
    HookResult,
    on_policy_algorithms::{
        Advantages, BatchIndexIterator, Logps, Returns, batches_advantages_and_returns, logps,
        sample,
    },
};

/// Hyperparameters controlling PPO training behavior.
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
pub trait PPOHook<M: OnPolicyLearningModule> {
    fn before_learning_hook<B: TrajectoryBatchT<M::InferenceTensor>>(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _batches: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryBatchT<M::InferenceTensor>>(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _batches: &[B],
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _losses: &mut <M as LearningModule>::Losses,
        _data: &PPOBatchData<M::LearningTensor>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

/// Prototype PPO variant over finalized trajectory batches.
pub struct PPO<Module: OnPolicyLearningModule, Hooks: PPOHook<Module>> {
    /// PPO hyperparameters.
    pub params: PPOParams,
    /// Learning module containing policy, value function, and optimizer state.
    pub lm: Module,
    /// Hook implementation used to customize learning behavior.
    pub hooks: Hooks,
}

impl<Module: OnPolicyLearningModule, Hooks: PPOHook<Module>> PPO<Module, Hooks> {
    fn batch_loop<B: TrajectoryBatchT<Module::InferenceTensor>>(
        &mut self,
        batches: &[B],
        advantages: &Advantages,
        logps: &Logps,
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
            let logp_old = lm.tensor_from_slice(&logps.sample(&indices));
            let returns = lm.tensor_from_slice(&returns.sample(&indices));
            let logp = lm.policy().log_probs(&observations, &actions)?;
            let values_pred = lm.values(&observations)?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean()?;
            let logp_diff = logp.sub(&logp_old)?;
            let ratio = logp_diff.exp()?;
            let clip_ratio =
                ratio.clamp(1. - self.params.clip_range, 1. + self.params.clip_range)?;
            let clipped_adv = clip_ratio.mul(&advantages)?;
            let ratio_adv = ratio.mul(&advantages)?;
            let policy_loss = ratio_adv.minimum(&clipped_adv)?.neg()?.mean()?;
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

    fn learning_loop<B: TrajectoryBatchT<Module::InferenceTensor>>(
        &mut self,
        batches: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> anyhow::Result<()> {
        loop {
            self.batch_loop(batches, &advantages, &logps, &returns)?;
            let rollout_hook_res = self
                .hooks
                .rollout_hook(&mut self.params, &mut self.lm, batches);
            r2l_core::return_on_hook_result!(rollout_hook_res?);
        }
    }

    /// Prototype learning entrypoint over finalized trajectory batches.
    pub fn learn<B: TrajectoryBatchT<Module::InferenceTensor>>(
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
        self.learning_loop(batches, advantages, returns, logps)?;
        Ok(())
    }
}

impl<M: OnPolicyLearningModule, H: PPOHook<M>> Agent for PPO<M, H> {
    type Tensor = M::InferenceTensor;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
        self.lm.inference_policy()
    }

    fn learn<B: TrajectoryBatchT<Self::Tensor>>(&mut self, buffers: &[B]) -> Result<()> {
        PPO::learn(self, buffers)
    }
}
