use crate::tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred};
use crate::utils::rollout_buffer::{RolloutBatch, RolloutBuffer};
use candle_core::{Result, Tensor};
use r2l_macros::policy_hook;

pub struct PPOBatchData {
    pub logp: Logp,
    pub values_pred: ValuesPred,
    pub logp_diff: LogpDiff,
    pub ratio: Tensor,
}

// TODO: maybe we want the PPO<P> to be an argument instead of just P?
#[policy_hook]
pub trait BatchHook<P> {
    fn call_hook(
        &mut self,
        policy: &mut P,
        rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<bool>;
}

#[policy_hook]
#[allow(clippy::ptr_arg)]
trait BeforeLearningHook<P> {
    fn call_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &mut Vec<RolloutBuffer>,
    ) -> candle_core::Result<bool>;
}

#[policy_hook]
#[allow(clippy::ptr_arg)]
trait AfterTrainingHook<P> {
    fn call_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &Vec<RolloutBuffer>,
    ) -> candle_core::Result<bool>;
}

pub struct PPOHooks<P> {
    before_learning: Option<Box<dyn BeforeLearningHook<P>>>,
    batch_hook: Option<Box<dyn BatchHook<P>>>,
    after_training: Box<dyn AfterTrainingHook<P>>,
}

impl<P> PPOHooks<P> {
    pub fn empty() -> Self {
        Self {
            before_learning: None,
            batch_hook: None,
            after_training: IntoAfterTrainingHook::into_boxed(|| Ok(true)),
        }
    }

    pub fn call_before_training_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &mut Vec<RolloutBuffer>,
    ) -> Result<bool> {
        if let Some(hook) = &mut self.before_learning {
            hook.call_hook(policy, rollout_buffers)
        } else {
            Ok(false)
        }
    }

    pub fn call_batch_hook(
        &mut self,
        policy: &mut P,
        rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> Result<bool> {
        if let Some(hook) = &mut self.batch_hook {
            hook.call_hook(policy, rollout_batch, policy_loss, value_loss, data)
        } else {
            Ok(false)
        }
    }

    pub fn call_after_training_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &Vec<RolloutBuffer>,
    ) -> Result<bool> {
        self.after_training.call_hook(policy, rollout_buffers)
    }

    pub fn add_batching_hook<H>(mut self, batch_hook: impl IntoBatchHook<P, H>) -> Self {
        self.batch_hook = Some(batch_hook.into_boxed());
        self
    }

    pub fn set_batching_hook<H>(&mut self, batch_hook: impl IntoBatchHook<P, H>) {
        self.batch_hook = Some(batch_hook.into_boxed());
    }

    pub fn add_after_learning_hook<H>(
        mut self,
        after_learning_hook: impl IntoAfterTrainingHook<P, H>,
    ) -> Self {
        self.after_training = after_learning_hook.into_boxed();
        self
    }

    pub fn set_after_learning_hook<H>(
        &mut self,
        after_learning_hook: impl IntoAfterTrainingHook<P, H>,
    ) {
        self.after_training = after_learning_hook.into_boxed()
    }

    pub fn add_before_learning_hook<H>(
        mut self,
        before_learning_hook: impl IntoBeforeLearningHook<P, H>,
    ) -> Self {
        self.before_learning = Some(before_learning_hook.into_boxed());
        self
    }

    pub fn set_before_learning_hook<H>(
        &mut self,
        before_learning_hook: impl IntoBeforeLearningHook<P, H>,
    ) {
        self.before_learning = Some(before_learning_hook.into_boxed());
    }
}
