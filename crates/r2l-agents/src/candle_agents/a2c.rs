use crate::candle_agents::{ModuleWithValueFunction, ppo::HookResult};
use anyhow::Result;
use candle_core::{Device, Tensor as CandleTensor};
use r2l_candle_lm::{
    candle_rollout_buffer::{
        CandleRolloutBuffer, RolloutBatchIterator, calculate_advantages_and_returns,
    },
    learning_module2::PolicyValuesLosses,
    tensors::{PolicyLoss, ValueLoss},
};
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    policies::{LearningModule, ValueFunction},
    utils::rollout_buffer::{Advantages, Logps, Returns, RolloutBuffer},
};

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub trait A2CHooks<M: ModuleWithValueFunction> {
    fn before_learning_hook(
        &mut self,
        learning_module: &mut A2CCore<M>,
        rollout_buffers: &mut Vec<CandleRolloutBuffer>,
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<HookResult>;
}

pub struct DefaultA2CHooks;

// TODO: we have to rething this anyways
impl<M: ModuleWithValueFunction> A2CHooks<M> for DefaultA2CHooks {
    fn before_learning_hook(
        &mut self,
        _learning_module: &mut A2CCore<M>,
        _rollout_buffers: &mut Vec<CandleRolloutBuffer>,
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        todo!()
    }
}

pub struct A2CCore<M: ModuleWithValueFunction> {
    pub module: M,
    pub device: Device,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

pub struct A2C<M: ModuleWithValueFunction> {
    pub a2c: A2CCore<M>,
    pub hooks: Box<dyn A2CHooks<M>>,
}

impl<M: ModuleWithValueFunction> A2C<M> {
    fn batching_loop(&mut self, batch_iter: &mut RolloutBatchIterator) -> Result<()> {
        loop {
            let Some(batch) = batch_iter.next() else {
                return Ok(());
            };
            let logps = self
                .a2c
                .module
                .get_policy_ref()
                .log_probs(&batch.observations, &batch.actions)?;
            let values_pred = self
                .a2c
                .module
                .value_func()
                .calculate_values(&batch.observations)?;
            let value_loss = ValueLoss(batch.returns.sub(&values_pred)?.sqr()?.mean_all()?);
            let policy_loss = PolicyLoss(batch.advantages.mul(&logps)?.neg()?.mean_all()?);
            self.a2c
                .module
                .learning_module()
                .update(PolicyValuesLosses {
                    policy_loss,
                    value_loss,
                })?;
        }
    }
}

impl<M: ModuleWithValueFunction> Agent for A2C<M> {
    type Policy = <M as ModuleWithValueFunction>::P;

    fn policy(&self) -> Self::Policy {
        self.a2c.module.get_inference_policy()
    }

    fn learn(&mut self, rollouts: Vec<RolloutBuffer<CandleTensor>>) -> Result<()> {
        let mut rollouts: Vec<CandleRolloutBuffer> = rollouts
            .into_iter()
            .map(|rb| CandleRolloutBuffer::from(rb))
            .collect();
        let (mut advantages, mut returns) = calculate_advantages_and_returns(
            &rollouts,
            self.a2c.module.value_func(),
            self.a2c.gamma,
            self.a2c.lambda,
        );
        let before_learning_hook_res = self.hooks.before_learning_hook(
            &mut self.a2c,
            &mut rollouts,
            &mut advantages,
            &mut returns,
        );
        let logps = Logps(
            rollouts
                .iter()
                .map(|roll| {
                    let states = &roll.0.states[0..roll.0.states.len() - 1];
                    let actions = &roll.0.actions;
                    self.policy()
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
            self.a2c.sample_size,
            self.a2c.device.clone(),
        );
        self.batching_loop(&mut batch_iter)?;
        Ok(())
    }
}
