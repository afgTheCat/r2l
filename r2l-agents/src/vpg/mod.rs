use crate::ppo::PPOLearningModule;
use candle_core::{Device, Result, Tensor};
use r2l_core::{
    agents::Agent,
    distributions::Distribution,
    policies::{
        ValueFunction,
        learning_modules::{LearningModule, PolicyValuesLosses},
    },
    tensors::{PolicyLoss, ValueLoss},
    utils::rollout_buffer::{
        Logps, RolloutBatch, RolloutBatchIterator, RolloutBuffer, calculate_advantages_and_returns,
    },
};

pub trait VPG3LearningModule: LearningModule<Losses = PolicyValuesLosses> + ValueFunction {}

pub struct VPG<D: Distribution, LM: PPOLearningModule> {
    pub distribution: D,
    pub learning_module: LM,
    device: Device,
    gamma: f32,
    lambda: f32,
    sample_size: usize,
}

impl<D: Distribution<Tensor = Tensor>, LM: PPOLearningModule> VPG<D, LM> {
    fn train_single_batch(&mut self, batch: RolloutBatch) -> Result<bool> {
        let policy_loss = PolicyLoss(batch.advantages.mul(&batch.logp_old)?.neg()?.mean_all()?);
        let values_pred = self.learning_module.calculate_values(&batch.observations)?;
        let value_loss = ValueLoss(batch.returns.sub(&values_pred)?.sqr()?.mean_all()?);
        self.learning_module.update(PolicyValuesLosses {
            policy_loss,
            value_loss,
        })?;
        Ok(true)
    }
}

impl<D: Distribution<Tensor = Tensor>, LM: PPOLearningModule> Agent for VPG<D, LM> {
    type Dist = D;

    fn distribution(&self) -> &Self::Dist {
        &self.distribution
    }

    fn learn(&mut self, rollouts: Vec<RolloutBuffer>) -> Result<()> {
        let (advantages, returns) = calculate_advantages_and_returns(
            &rollouts,
            &self.learning_module,
            self.gamma,
            self.lambda,
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
        let rollout_batch_iter = RolloutBatchIterator::new(
            &rollouts,
            &advantages,
            &returns,
            &logps,
            self.sample_size,
            self.device.clone(),
        );
        for batch in rollout_batch_iter {
            self.train_single_batch(batch)?;
        }
        Ok(())
    }
}
