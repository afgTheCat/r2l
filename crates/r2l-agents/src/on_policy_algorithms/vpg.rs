//! Vanilla Policy Gradient implementation.

use anyhow::Result;
use r2l_core::{
    buffers::TrajectoryContainer,
    models::Policy,
    on_policy::{
        algorithm::Agent, learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses,
    },
    tensor::R2lTensorMath,
};

use crate::on_policy_algorithms::{
    Advantages, BatchIndexIterator, Returns, buffers_advantages_and_returns, sample,
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

/// Vanilla Policy Gradient algorithm over an [`OnPolicyLearningModule`].
///
/// `VPG` computes rollout advantages and returns, then performs one learning
/// pass over minibatches sampled from the collected trajectories.
pub struct VPG<Module: OnPolicyLearningModule> {
    /// VPG hyperparameters.
    pub params: VPGParams,
    /// Learning module containing policy, value function, and optimizer state.
    pub lm: Module,
}

impl<Module: OnPolicyLearningModule> VPG<Module> {
    fn batch_loop<B: TrajectoryContainer<Tensor = Module::InferenceTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.params.sample_size);
        let lm = &mut self.lm;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indices, Module::lifter);
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
}

impl<M: OnPolicyLearningModule> Agent for VPG<M> {
    type Tensor = M::InferenceTensor;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
        self.lm.inference_policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> Result<()> {
        let (advantages, returns) = buffers_advantages_and_returns(
            buffers,
            &self.lm,
            self.params.gamma,
            self.params.lambda,
            M::lifter,
        )?;
        self.batch_loop(buffers, &advantages, &returns)?;
        Ok(())
    }
}
