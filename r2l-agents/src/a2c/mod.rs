use crate::ppo::HookResult;
use candle_core::{Device, Result, Tensor};
use r2l_core::{
    agents::Agent,
    distributions::Distribution,
    policies::{
        ValueFunction,
        learning_modules::{LearningModule, LearningModuleKind, PolicyValuesLosses},
    },
    tensors::{PolicyLoss, ValueLoss},
    utils::rollout_buffer::{
        Advantages, Logps, Returns, RolloutBatchIterator, RolloutBuffer,
        calculate_advantages_and_returns,
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

pub trait A2CLearningModule: LearningModule<Losses = PolicyValuesLosses> + ValueFunction {}

impl A2CLearningModule for LearningModuleKind {}

pub trait A2CHooks<LM: A2CLearningModule> {
    fn before_learning_hook(
        &mut self,
        learning_module: &mut LM,
        rollout_buffers: &mut Vec<RolloutBuffer<Tensor>>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> Result<HookResult>;
}

pub struct DefaultA2CHooks;

// TODO: we have to rething this anyways
impl<LM: A2CLearningModule> A2CHooks<LM> for DefaultA2CHooks {
    fn before_learning_hook(
        &mut self,
        _learning_module: &mut LM,
        _rollout_buffers: &mut Vec<RolloutBuffer<Tensor>>,
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        todo!()
    }
}

pub struct A2C<D: Distribution, LM: A2CLearningModule> {
    pub distribution: D,
    pub learning_module: LM,
    pub hooks: Box<dyn A2CHooks<LM>>,
    pub device: Device,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl<D: Distribution<Tensor = Tensor>, LM: A2CLearningModule> A2C<D, LM> {
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        loop {
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let logps = self
                .distribution
                .log_probs(batch.observations.clone(), batch.actions.clone())?;
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

impl<D: Distribution<Tensor = Tensor>, LM: A2CLearningModule> Agent for A2C<D, LM> {
    type Dist = D;

    fn distribution(&self) -> &Self::Dist {
        &self.distribution
    }

    fn learn(&mut self, mut rollouts: Vec<RolloutBuffer<Tensor>>) -> Result<()> {
        let (mut advantages, mut returns) = calculate_advantages_and_returns(
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
        let logps = Logps(
            rollouts
                .iter()
                .map(|roll| {
                    let states = Tensor::stack(&roll.states[0..roll.states.len() - 1], 0).unwrap();
                    let actions = Tensor::stack(&roll.actions, 0).unwrap();
                    self.distribution()
                        .log_probs(states, actions)
                        .map(|t| t.squeeze(0).unwrap().to_vec1().unwrap())
                })
                .collect::<Result<Vec<Vec<f32>>>>()?,
        );
        process_hook_result!(before_learning_hook_res);
        let mut batch_iter = RolloutBatchIterator::new(
            &rollouts,
            &advantages,
            &returns,
            &logps,
            self.sample_size,
            self.device.clone(),
        );
        self.batching_loop(&mut batch_iter)?;
        Ok(())
    }
}
