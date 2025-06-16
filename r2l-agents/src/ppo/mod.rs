pub mod builder;
pub mod hooks;

use crate::ppo::hooks::HookResult;
use candle_core::{Device, Result, Tensor};
use hooks::{PPOBatchData, PPOHooks};
use r2l_core::{
    agents::Agent,
    distributions::Distribution,
    on_policy_algorithm::OnPolicyAlgorithm,
    policies::{Policy, PolicyWithValueFunction},
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
    utils::rollout_buffer::{
        Advantages, Returns, RolloutBatchIterator, RolloutBuffer, calculate_advantages_and_returns,
    },
};
use std::ops::Deref;

// This changes the contorlflow, returning on hook break
macro_rules! process_hook {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

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
    // batch loop
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        loop {
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let distribution = self.distribution();
            let logp = Logp(distribution.log_probs(&batch.observations, &batch.actions)?);
            let values_pred = ValuesPred(self.policy.calculate_values(&batch.observations)?);
            let mut value_loss = ValueLoss(batch.returns.sub(&values_pred)?.sqr()?.mean_all()?);
            let logp_diff = LogpDiff((logp.deref() - &batch.logp_old)?);
            let ratio = logp_diff.exp()?;
            let clip_adv = (ratio.clamp(1. - self.clip_range, 1. + self.clip_range)?
                * batch.advantages.clone())?;
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
            let hook_result = self.hooks.call_batch_hook(
                &mut self.policy,
                &batch,
                &mut policy_loss,
                &mut value_loss,
                &ppo_data,
            )?;
            self.policy.update(&policy_loss, &value_loss)?;
            match hook_result {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
        }
    }

    // rollout loop
    fn rollout_loop(
        &mut self,
        rollouts: &Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> Result<()> {
        loop {
            let mut batch_iter = RolloutBatchIterator::new(
                &rollouts,
                advantages,
                returns,
                self.sample_size,
                self.device.clone(),
            );
            self.batching_loop(&mut batch_iter)?;
            process_hook!(self.hooks.call_rollout_hook(&mut self.policy, &rollouts));
        }
    }
}

impl<P: PolicyWithValueFunction> Agent for PPO<P> {
    fn policy(&self) -> &impl Policy {
        &self.policy
    }

    // TODO: functinally done, but could be made more readable
    fn learn(&mut self, mut rollouts: Vec<RolloutBuffer>) -> Result<()> {
        let (mut advantages, mut returns) =
            calculate_advantages_and_returns(&rollouts, &self.policy, self.gamma, self.lambda);
        process_hook!(self.hooks.call_before_training_hook(
            &mut self.policy,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        ));
        // we could handle rollout loop here, and use it in the after learning hoop
        self.rollout_loop(&rollouts, &mut advantages, &mut returns)?;
        Ok(())
    }
}

pub type PPOAlgorithm<E, Policy> = OnPolicyAlgorithm<E, PPO<Policy>>;
