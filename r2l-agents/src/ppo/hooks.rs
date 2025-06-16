use candle_core::{Result, Tensor};
use r2l_core::{
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
    utils::rollout_buffer::{Advantages, Returns, RolloutBatch, RolloutBuffer},
};
use r2l_macros::policy_hook;

pub enum HookResult {
    Continue,
    Break,
}

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
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<bool>;
}

#[policy_hook]
#[allow(clippy::ptr_arg)]
trait RolloutHook<P> {
    fn call_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &Vec<RolloutBuffer>,
    ) -> candle_core::Result<bool>;
}

// TODO: should we add after learning hook?
// before_learning -> preprocessing hook?
pub struct PPOHooks<P> {
    // called before the rollout loop is called
    before_learning: Option<Box<dyn BeforeLearningHook<P>>>,
    // called at the end of each rollout cycle
    rollout_hook: Box<dyn RolloutHook<P>>,
    // called before training the model actually happens
    batch_hook: Option<Box<dyn BatchHook<P>>>,
}

impl<P> PPOHooks<P> {
    pub fn empty() -> Self {
        Self {
            before_learning: None,
            batch_hook: None,
            rollout_hook: IntoRolloutHook::into_boxed(|| Ok(true)),
        }
    }

    pub fn call_before_training_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> Result<HookResult> {
        if let Some(hook) = &mut self.before_learning {
            let should_stop = hook.call_hook(policy, rollout_buffers, advantages, returns)?;
            if should_stop {
                Ok(HookResult::Break)
            } else {
                Ok(HookResult::Continue)
            }
        } else {
            Ok(HookResult::Continue)
        }
    }

    pub fn call_batch_hook(
        &mut self,
        policy: &mut P,
        rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> Result<HookResult> {
        if let Some(hook) = &mut self.batch_hook {
            let should_stop =
                hook.call_hook(policy, rollout_batch, policy_loss, value_loss, data)?;
            if should_stop {
                Ok(HookResult::Break)
            } else {
                Ok(HookResult::Continue)
            }
        } else {
            Ok(HookResult::Break)
        }
    }

    // TODO: this is confusing
    pub fn call_rollout_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &Vec<RolloutBuffer>,
    ) -> Result<HookResult> {
        let should_stop = self.rollout_hook.call_hook(policy, rollout_buffers)?;
        if should_stop {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    pub fn add_batching_hook<H>(mut self, batch_hook: impl IntoBatchHook<P, H>) -> Self {
        self.batch_hook = Some(batch_hook.into_boxed());
        self
    }

    pub fn set_batching_hook<H>(&mut self, batch_hook: impl IntoBatchHook<P, H>) {
        self.batch_hook = Some(batch_hook.into_boxed());
    }

    pub fn add_rollout_hook<H>(mut self, rollout_hook: impl IntoRolloutHook<P, H>) -> Self {
        self.rollout_hook = rollout_hook.into_boxed();
        self
    }

    pub fn set_after_learning_hook<H>(&mut self, rollout_hook: impl IntoRolloutHook<P, H>) {
        self.rollout_hook = rollout_hook.into_boxed()
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
