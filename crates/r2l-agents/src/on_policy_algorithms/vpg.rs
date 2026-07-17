//! Prototype VPG training path that consumes trajectory batches directly.

use anyhow::Result;
use r2l_core::{
    buffers::TrajectoryBatch,
    models::Policy,
    on_policy::{
        algorithm::Agent, learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses,
    },
    tensor::R2lTensor,
};

use crate::on_policy_algorithms::{
    Advantages, BatchIndexIterator, Returns, batches_advantages_and_returns, sample,
};

/// Hyperparameters controlling VPG training behavior.
pub struct VPGParams {
    /// Discount factor used for return and advantage estimation.
    pub gamma: f32,
    /// GAE lambda used for advantage estimation.
    pub lambda: f32,
    /// Minibatch size used during the learning pass.
    pub sample_size: usize,
}

impl Default for VPGParams {
    fn default() -> Self {
        Self {
            gamma: 0.98,
            lambda: 0.8,
            sample_size: 64,
        }
    }
}

/// Prototype Vanilla Policy Gradient algorithm over finalized trajectory batches.
pub struct VPG<Module: OnPolicyLearningModule> {
    /// VPG hyperparameters.
    pub params: VPGParams,
    /// Learning module containing policy, value function, and optimizer state.
    pub lm: Module,
}

impl<Module: OnPolicyLearningModule> VPG<Module> {
    fn batch_loop<B: TrajectoryBatch<Module::InferenceTensor, State = Module::InferenceState>>(
        &mut self,
        batches: &[B],
        advantages: &Advantages,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(batches, self.params.sample_size);
        let lm = &mut self.lm;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(batches, &indices, Module::lifter);
            let advantages = lm.tensor_from_slice(&advantages.sample(&indices));
            let returns = lm.tensor_from_slice(&returns.sample(&indices));
            let logp = lm.policy().log_probs(&observations, &actions)?;
            let values_pred = lm.values(&observations)?;
            let policy_loss = advantages.mul(&logp)?.neg()?.mean()?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean()?;
            let losses = Module::Losses::from_policy_value_losses(policy_loss, value_loss);
            lm.update(losses)?;
        }
    }

    /// Prototype learning entrypoint over finalized trajectory batches.
    pub fn learn<B: TrajectoryBatch<Module::InferenceTensor, State = Module::InferenceState>>(
        &mut self,
        batches: &[B],
    ) -> Result<()> {
        let (advantages, returns) = batches_advantages_and_returns(
            batches,
            &self.lm,
            self.params.gamma,
            self.params.lambda,
            Module::lifter,
        )?;
        self.batch_loop(batches, &advantages, &returns)?;
        Ok(())
    }
}

impl<M: OnPolicyLearningModule> Agent for VPG<M> {
    type Tensor = M::InferenceTensor;
    type RolloutState = M::InferenceState;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
        self.lm.inference_policy()
    }

    fn learn<B: TrajectoryBatch<Self::Tensor, State = Self::RolloutState>>(
        &mut self,
        buffers: &[B],
    ) -> Result<()> {
        VPG::learn(self, buffers)
    }
}
