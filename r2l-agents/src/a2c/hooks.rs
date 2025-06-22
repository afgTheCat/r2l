use crate::ppo::hooks::HookResult;
use candle_core::Result;
use r2l_core::utils::rollout_buffer::{Advantages, Returns, RolloutBuffer};
use r2l_macros::policy_hook;

#[policy_hook]
#[allow(clippy::ptr_arg)]
trait A2CBeforeLearningHook<P> {
    fn call_hook(
        &mut self,
        policy: &mut P,
        rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<bool>;
}

pub struct A2CHooks<P> {
    before_learning: Box<dyn A2CBeforeLearningHook<P>>,
}

impl<P> A2CHooks<P> {
    pub fn empty() -> Self {
        Self {
            before_learning: IntoA2CBeforeLearningHook::into_boxed(|| Ok(false)),
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
}
