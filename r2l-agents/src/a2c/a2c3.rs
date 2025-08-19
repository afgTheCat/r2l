use crate::ppo::hooks::HookResult;
use candle_core::{Device, Result};
use r2l_core::{
    agents::Agent2,
    distributions::Distribution,
    policies::{
        ValueFunction,
        learning_modules::{LearningModule, LearningModuleKind, PolicyValuesLosses},
    },
    tensors::{PolicyLoss, ValueLoss},
    utils::rollout_buffer::{
        Advantages, Returns, RolloutBatchIterator, RolloutBuffer, calculate_advantages_and_returns2,
    },
};

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub trait A2C3LearningModule: LearningModule<Losses = PolicyValuesLosses> + ValueFunction {}

impl A2C3LearningModule for LearningModuleKind {}

pub trait A2C3Hooks<LM: A2C3LearningModule> {
    fn before_learning_hook(
        &mut self,
        learning_module: &mut LM,
        rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> Result<HookResult>;
}

pub struct DefaultA2C3Hooks;

// TODO: we have to rething this anyways
impl<LM: A2C3LearningModule> A2C3Hooks<LM> for DefaultA2C3Hooks {
    fn before_learning_hook(
        &mut self,
        _learning_module: &mut LM,
        _rollout_buffers: &mut Vec<RolloutBuffer>,
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        todo!()
    }
}

pub struct A2C3<D: Distribution, LM: A2C3LearningModule> {
    pub distribution: D,
    pub learning_module: LM,
    pub hooks: Box<dyn A2C3Hooks<LM>>,
    pub device: Device,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl<D: Distribution, LM: A2C3LearningModule> A2C3<D, LM> {
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        loop {
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let logps = self
                .distribution
                .log_probs(&batch.observations, &batch.actions)?;
            let values_pred = self.learning_module.calculate_values(&batch.observations)?;
            let value_loss = ValueLoss(batch.returns.sub(&values_pred)?.sqr()?.mean_all()?);
            let policy_loss = PolicyLoss(batch.advantages.mul(&logps)?.neg()?.mean_all()?);
            self.learning_module.update(PolicyValuesLosses {
                policy_loss,
                value_loss,
            })?;
        }
    }
}

impl<D: Distribution, LM: A2C3LearningModule> Agent2 for A2C3<D, LM> {
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
