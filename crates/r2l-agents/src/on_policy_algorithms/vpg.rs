use anyhow::Result;
use r2l_core::{
    agents::Agent, distributions::Policy, losses::PolicyValuesLosses,
    sampler::buffer::TrajectoryContainer, tensor::R2lTensorMath,
};

use crate::{
    BatchIndexIterator, buffers_advantages_and_returns,
    on_policy_algorithms::OnPolicyLearningModule, sample,
};

pub struct VPGParams {
    pub gamma: f32,
    pub lambda: f32,
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

pub struct VPG<Module: OnPolicyLearningModule> {
    pub params: VPGParams,
    pub lm: Module,
}

impl<Module: OnPolicyLearningModule> VPG<Module> {
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

impl<M: OnPolicyLearningModule> Agent for VPG<M> {
    type Tensor = M::InferenceTensor;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
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
