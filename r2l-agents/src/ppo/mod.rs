pub mod builder;
pub mod hooks;

use candle_core::{Device, Result, Tensor};
use hooks::{PPOBatchData, PPOHooks};
use r2l_core::{
    agents::Agent,
    distributions::Distribution,
    on_policy_algorithm::OnPolicyAlgorithm,
    policies::{Policy, PolicyWithValueFunction},
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
    utils::rollout_buffer::{
        RolloutBatch, RolloutBatchIterator, RolloutBuffer, calculate_advantages_and_returns,
    },
};
use std::ops::Deref;

pub struct PPO<P: PolicyWithValueFunction> {
    pub policy: P,
    pub hooks: PPOHooks<P>,
    pub clip_range: f32,
    pub device: Device,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl<P: PolicyWithValueFunction> PPO<P> {
    fn train_single_batch(&mut self, batch: RolloutBatch) -> Result<bool> {
        let distribution = self.distribution();
        let logp = Logp(distribution.log_probs(&batch.observations, &batch.actions)?);
        let values_pred = ValuesPred(self.policy.calculate_values(&batch.observations)?);
        let mut value_loss = ValueLoss(batch.returns.sub(&values_pred)?.sqr()?.mean_all()?);
        let logp_diff = LogpDiff((logp.deref() - &batch.logp_old)?);
        let ratio = logp_diff.exp()?;
        let clip_adv =
            (ratio.clamp(1. - self.clip_range, 1. + self.clip_range)? * batch.advantages.clone())?;
        let mut policy_loss = PolicyLoss(
            Tensor::minimum(&(&ratio * &batch.advantages)?, &clip_adv)?
                .neg()?
                .mean_all()?,
        );
        let ppo_data = PPOBatchData {
            logp,
            values_pred,
            logp_diff,
            ratio,
        };
        let should_continue = self.hooks.call_batch_hook(
            &mut self.policy,
            &batch,
            &mut policy_loss,
            &mut value_loss,
            &ppo_data,
        )?;
        self.policy.update(&policy_loss, &value_loss)?;
        Ok(should_continue)
    }
}

impl<P: PolicyWithValueFunction> Agent for PPO<P> {
    fn policy(&self) -> &impl Policy {
        &self.policy
    }

    // TODO: functinally done, but could be made more readable
    fn learn(&mut self, mut rollouts: Vec<RolloutBuffer>) -> candle_core::Result<()> {
        let (mut advantages, mut returns) =
            calculate_advantages_and_returns(&rollouts, &self.policy, self.gamma, self.lambda);
        if self.hooks.call_before_training_hook(
            &mut self.policy,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        )? {
            return Ok(());
        }
        loop {
            let rollout_batch_iter = RolloutBatchIterator::new(
                &rollouts,
                &advantages,
                &returns,
                self.sample_size,
                self.device.clone(),
            );
            for batch in rollout_batch_iter {
                if self.train_single_batch(batch)? {
                    break;
                }
            }
            if self
                .hooks
                .call_after_training_hook(&mut self.policy, &rollouts)?
            {
                return Ok(());
            }
        }
    }
}

pub type PPOAlgorithm<E, Policy> = OnPolicyAlgorithm<E, PPO<Policy>>;
