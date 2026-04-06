use anyhow::Result;
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    losses::PolicyValuesLosses,
    policies::{LearningModule, ValueFunction},
    sampler::buffer::TrajectoryContainer,
    tensor::R2lTensorMath,
};

use crate::ppo::RolloutLearningModule;
use crate::{BatchIndexIterator, buffers_advantages_and_returns, sample};

pub trait VPGModule:
    RolloutLearningModule<LearningTensor: R2lTensorMath>
    + LearningModule<Losses: PolicyValuesLosses<<Self as RolloutLearningModule>::LearningTensor>>
    + ValueFunction<Tensor = <Self as RolloutLearningModule>::LearningTensor>
{
}

pub struct NewVPGParams {
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for NewVPGParams {
    fn default() -> Self {
        Self {
            gamma: 0.98,
            lambda: 0.8,
            sample_size: 64,
        }
    }
}

pub struct NewVPG<Module: VPGModule> {
    pub params: NewVPGParams,
    pub lm: Module,
}

impl<Module: VPGModule> NewVPG<Module> {
    fn batch_loop<B: TrajectoryContainer<Tensor = Module::InferenceTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &r2l_core::utils::rollout_buffer::Advantages,
        returns: &r2l_core::utils::rollout_buffer::Returns,
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
            let logp = lm.get_policy().log_probs(&observations, &actions)?;
            let values_pred = lm.calculate_values(&observations)?;
            let policy_loss = advantages.mul(&logp)?.neg()?.mean()?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean()?;
            let losses = Module::Losses::losses(policy_loss, value_loss);
            lm.update(losses)?;
        }
    }
}

impl<M: VPGModule> Agent for NewVPG<M> {
    type Tensor = M::InferenceTensor;
    type Policy = M::InferencePolicy;

    fn policy(&self) -> Self::Policy {
        self.lm.get_inference_policy()
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
