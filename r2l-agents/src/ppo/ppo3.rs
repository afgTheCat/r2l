use crate::ppo::hooks::{HookResult, PPOBatchData};
use candle_core::{Device, Result, Tensor};
use r2l_core::{
    agents::Agent2,
    distributions::Distribution,
    policies::{
        ValueFunction,
        learning_modules::{LearningModule, LearningModuleKind, PolicyValuesLosses},
    },
    tensors::{Logp, LogpDiff, PolicyLoss, ValueLoss, ValuesPred},
    utils::rollout_buffer::{
        Advantages, Returns, RolloutBatch, RolloutBatchIterator, RolloutBuffer,
        calculate_advantages_and_returns2,
    },
};
use std::ops::Deref;

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub trait PPO3LearningModule: LearningModule<Losses = PolicyValuesLosses> + ValueFunction {}

impl PPO3LearningModule for LearningModuleKind {}

pub trait PPP3HooksTrait<LM: PPO3LearningModule> {
    fn before_learning_hook(
        &mut self,
        learning_module: &mut LM,
        rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> Result<HookResult>;

    fn rollout_hook(
        &mut self,
        learning_module: &mut LM,
        rollout_buffers: &Vec<RolloutBuffer>,
    ) -> Result<HookResult>;

    fn batch_hook(
        &mut self,
        learning_module: &mut LM,
        rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> Result<HookResult>;
}

pub struct PPO3<D: Distribution, LM: PPO3LearningModule> {
    pub distribution: D,
    pub learning_module: LM,
    pub hooks: Box<dyn PPP3HooksTrait<LM>>,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

impl<D: Distribution, LM: PPO3LearningModule> PPO3<D, LM> {
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        loop {
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let logp = Logp(
                self.distribution
                    .log_probs(&batch.observations, &batch.actions)?,
            );
            let values_pred =
                ValuesPred(self.learning_module.calculate_values(&batch.observations)?);
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
            let hook_result = self.hooks.batch_hook(
                &mut self.learning_module,
                &batch,
                &mut policy_loss,
                &mut value_loss,
                &ppo_data,
            )?;
            self.learning_module.update(PolicyValuesLosses {
                policy_loss,
                value_loss,
            })?;
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
            let rollout_hook_res = self
                .hooks
                .rollout_hook(&mut self.learning_module, &rollouts);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<D: Distribution, LM: PPO3LearningModule> Agent2 for PPO3<D, LM> {
    type Dist = D;

    fn distribution(&self) -> &Self::Dist {
        &self.distribution
    }

    fn learn(&mut self, mut rollouts: Vec<RolloutBuffer>) -> Result<()> {
        let (mut advantages, mut returns) = calculate_advantages_and_returns2(
            &rollouts,
            &self.learning_module,
            self.gamma,
            self.lambda,
        );
        let before_learning_hook_res = self.hooks.before_learning_hook(
            &mut self.learning_module,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        );
        process_hook_result!(before_learning_hook_res);
        self.rollout_loop(&rollouts, &mut advantages, &mut returns)?;
        Ok(())
    }
}
