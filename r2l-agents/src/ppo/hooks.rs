use candle_core::{Result, Tensor};
use r2l_core::{
    policies::PolicyWithValueFunction,
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
    utils::rollout_buffer::{Advantages, Returns, RolloutBatch, RolloutBuffer},
};
use r2l_macros::policy_hook;

use crate::ppo::ppo2::PPOHooksTrait;

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
pub trait BeforeLearningHook<P> {
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
pub trait RolloutHook<P> {
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
    pub before_learning: Box<dyn BeforeLearningHook<P>>,
    // called at the end of each rollout cycle
    pub rollout_hook: Box<dyn RolloutHook<P>>,
    // called before training the model actually happens
    pub batch_hook: Box<dyn BatchHook<P>>,
}

impl<P: PolicyWithValueFunction> PPOHooksTrait<P> for PPOHooks<P> {
    fn before_learning_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> Result<HookResult> {
        let should_stop =
            self.before_learning
                .call_hook(policy, rollout_buffers, advantages, returns)?;
        if should_stop {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn rollout_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &Vec<RolloutBuffer>,
    ) -> Result<HookResult> {
        todo!()
    }

    fn batch_hook(
        &mut self,
        policy: &mut P,
        rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> Result<HookResult> {
        todo!()
    }
}

impl<P> PPOHooks<P> {
    pub fn empty() -> Self {
        Self {
            before_learning: IntoBeforeLearningHook::into_boxed(|| Ok(false)),
            batch_hook: IntoBatchHook::into_boxed(|| Ok(false)),
            rollout_hook: IntoRolloutHook::into_boxed(|| Ok(true)),
        }
    }

    pub fn call_before_learning_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> Result<HookResult> {
        let should_stop =
            self.before_learning
                .call_hook(policy, rollout_buffers, advantages, returns)?;
        if should_stop {
            Ok(HookResult::Break)
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
        let should_stop =
            self.batch_hook
                .call_hook(policy, rollout_batch, policy_loss, value_loss, data)?;
        if should_stop {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
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
        self.batch_hook = batch_hook.into_boxed();
        self
    }

    pub fn set_batching_hook<H>(&mut self, batch_hook: impl IntoBatchHook<P, H>) {
        self.batch_hook = batch_hook.into_boxed();
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
        self.before_learning = before_learning_hook.into_boxed();
        self
    }

    pub fn set_before_learning_hook<H>(
        &mut self,
        before_learning_hook: impl IntoBeforeLearningHook<P, H>,
    ) {
        self.before_learning = before_learning_hook.into_boxed();
    }
}
