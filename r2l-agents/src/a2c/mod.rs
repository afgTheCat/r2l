pub mod builder;
pub mod hooks;

use crate::a2c::hooks::A2CHooks;
use crate::ppo::hooks::HookResult;
use candle_core::{Device, Result};
use r2l_core::{
    agents::Agent,
    distributions::Distribution,
    policies::PolicyWithValueFunction,
    utils::rollout_buffer::{
        RolloutBatchIterator, RolloutBuffer, calculate_advantages_and_returns,
    },
};

// This changes the contorlflow, returning on hook break
macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub struct A2C<P: PolicyWithValueFunction> {
    policy: P,
    hooks: A2CHooks<P>,
    device: Device,
    gamma: f32,
    lambda: f32,
    sample_size: usize,
}

impl<P: PolicyWithValueFunction> A2C<P> {
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        loop {
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let distribution = self.policy.distribution();
            let logps = distribution.log_probs(&batch.observations, &batch.actions)?;
            let values_pred = self.policy.calculate_values(&batch.observations)?;
            let value_loss = &batch.returns.sub(&values_pred)?.sqr()?.mean_all()?;
            let policy_loss = &batch.advantages.mul(&logps)?.neg()?.mean_all()?;
            self.policy.update(policy_loss, value_loss)?;
        }
    }
}

impl<P: PolicyWithValueFunction> Agent for A2C<P> {
    type Policy = P;

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn learn(&mut self, mut rollouts: Vec<RolloutBuffer>) -> candle_core::Result<()> {
        let (mut advantages, mut returns) =
            calculate_advantages_and_returns(&rollouts, &self.policy, self.gamma, self.lambda);
        let before_learning_hook_res = self.hooks.call_before_learning_hook(
            &mut self.policy,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        );
        process_hook_result!(before_learning_hook_res);
        let mut batch_iter = RolloutBatchIterator::new(
            &rollouts,
            &advantages,
            &returns,
            self.sample_size,
            self.device.clone(),
        );
        self.batching_loop(&mut batch_iter)?;
        Ok(())
    }
}
