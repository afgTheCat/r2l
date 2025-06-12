use candle_core::{Device, Result};
use r2l_core::{
    agents::Agent,
    distributions::Distribution,
    policies::{Policy, PolicyWithValueFunction},
    utils::{
        mini_batching::create_rollout_buffer_iterator,
        rollout_buffer::{RolloutBatch, RolloutBuffer},
    },
};

pub struct A2C<P: PolicyWithValueFunction> {
    policy: P,
    device: Device,
    gamma: f32,
    lambda: f32,
    sample_size: usize,
}

impl<P: PolicyWithValueFunction> A2C<P> {
    fn train_single_batch(&mut self, batch: RolloutBatch) -> Result<bool> {
        let distribution = self.policy.distribution();
        let logps = distribution.log_probs(&batch.observations, &batch.actions)?;
        // TODO: this one is ugly, value prediction is not neccessarly implemented for all policies
        let values_pred = self.policy.calculate_values(&batch.observations)?;
        let value_loss = &batch.returns.sub(&values_pred)?.sqr()?.mean_all()?;
        let policy_loss = &batch.advantages.mul(&logps)?.neg()?.mean_all()?;
        self.policy.update(policy_loss, value_loss)?;
        Ok(false)
    }
}

impl<P: PolicyWithValueFunction> Agent for A2C<P> {
    fn policy(&self) -> &impl Policy {
        &self.policy
    }

    fn learn(&mut self, mut rollout_buffers: Vec<RolloutBuffer>) -> candle_core::Result<()> {
        for rollout in rollout_buffers.iter_mut() {
            rollout.calculate_advantages_and_returns(&self.policy, self.gamma, self.lambda)?;
        }
        let rollout_buffer_iter =
            create_rollout_buffer_iterator(&rollout_buffers, self.sample_size, self.device.clone());
        for batch in rollout_buffer_iter {
            self.train_single_batch(batch)?;
        }
        Ok(())
    }
}
