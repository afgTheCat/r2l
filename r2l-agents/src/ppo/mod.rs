pub mod hooks;
pub mod ppo2;

use crate::ppo::hooks::HookResult;
use candle_core::{Device, Result, Tensor};
use hooks::{PPOBatchData, PPOHooks};
use r2l_core::{
    agents::Agent,
    distributions::Distribution,
    on_policy_algorithm::OnPolicyAlgorithm,
    policies::PolicyWithValueFunction,
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
    utils::rollout_buffer::{
        Advantages, Returns, RolloutBatchIterator, RolloutBuffer, calculate_advantages_and_returns,
    },
};
use std::ops::Deref;

// This changes the contorlflow, returning on hook break
macro_rules! process_hook_result {
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
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

impl<P: PolicyWithValueFunction> PPO<P> {
    // batch loop. It is kinda retarded that we crete a new RolloutBatchIterator for this tbh
    // we could do batches on the buffer and we would not need to crete this iteratir. Since we
    // already gonna reset the rollout batches that seems the me the best api that we can do.
    // Another thing that we achieve that way is that we dont have to normalize shit probably.
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
            let rollout_hook_res = self.hooks.call_rollout_hook(&mut self.policy, &rollouts);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<P: PolicyWithValueFunction> Agent for PPO<P> {
    type Policy = P;

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    // TODO: functinally done, but could be made more readable
    fn learn(&mut self, mut rollouts: Vec<RolloutBuffer>) -> Result<()> {
        let (mut advantages, mut returns) =
            calculate_advantages_and_returns(&rollouts, &self.policy, self.gamma, self.lambda);
        let before_learning_hook_res = self.hooks.call_before_learning_hook(
            &mut self.policy,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        );
        process_hook_result!(before_learning_hook_res);
        // we could handle rollout loop here, and use it in the after learning hoop
        self.rollout_loop(&rollouts, &mut advantages, &mut returns)?;
        Ok(())
    }
}

pub type PPOAlgorithm<E, Policy> = OnPolicyAlgorithm<E, PPO<Policy>>;
