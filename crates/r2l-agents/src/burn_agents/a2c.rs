use burn::tensor::{Tensor as BurnTensor, backend::AutodiffBackend};
use r2l_burn_lm::{
    learning_module::{BurnPolicy, ParalellActorCriticLM, PolicyValuesLosses},
    tensors::{PolicyLoss, ValueLoss},
};
use r2l_core::policies::{LearningModule, ValueFunction};
use r2l_core::utils::rollout_buffer::{Advantages, Returns};
use r2l_core::{agents::Agent, sampler::buffer::TrajectoryContainer};

use crate::{
    BatchIndexIterator, HookResult, buffers_advantages_and_returns, burn_agents::uplift_tensor,
    sample,
};

pub trait BurnA2CHooks<B: AutodiffBackend, D: BurnPolicy<B>> {
    fn before_learning_hook<T: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        _agent: &mut BurnA2CCore5<B, D>,
        _buffers: &[T],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct DefaultBurnA2CHooks;

impl<B: AutodiffBackend, D: BurnPolicy<B>> BurnA2CHooks<B, D> for DefaultBurnA2CHooks {}

pub struct BurnA2CCore5<B: AutodiffBackend, D: BurnPolicy<B>> {
    pub lm: ParalellActorCriticLM<B, D>,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

pub struct BurnA2C5<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnA2CHooks<B, D>> {
    pub a2c: BurnA2CCore5<B, D>,
    pub hooks: H,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnA2CHooks<B, D>> BurnA2C5<B, D, H> {
    pub fn new(a2c: BurnA2CCore5<B, D>, hooks: H) -> Self {
        Self { a2c, hooks }
    }

    fn batching_loop<C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        buffers: &[C],
        advantages: &Advantages,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.a2c.sample_size);
        let a2c = &mut self.a2c;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indices, uplift_tensor);
            let advantages = advantages.sample(&indices);
            let advantages = BurnTensor::from_data(advantages.as_slice(), &Default::default());
            let returns = returns.sample(&indices);
            let returns = BurnTensor::from_data(returns.as_slice(), &Default::default());
            let logps = a2c.lm.model.distr.log_probs(&observations, &actions)?;
            let values_pred = a2c.lm.calculate_values(&observations)?;
            let value_diff = returns - values_pred;
            let value_loss = ValueLoss((value_diff.clone() * value_diff).mean());
            let policy_loss = PolicyLoss((advantages * logps).neg().mean());
            a2c.lm.update(PolicyValuesLosses {
                policy_loss: policy_loss.0,
                value_loss: value_loss.0,
            })?;
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, H: BurnA2CHooks<B, D>> Agent for BurnA2C5<B, D, H> {
    type Tensor = BurnTensor<B::InnerBackend, 1>;
    type Policy = D::InnerModule;

    fn policy(&self) -> Self::Policy {
        self.a2c.lm.model.distr.valid()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns(
            buffers,
            &self.a2c.lm,
            self.a2c.gamma,
            self.a2c.lambda,
            uplift_tensor,
        )?;
        crate::process_hook_result!(self.hooks.before_learning_hook(
            &mut self.a2c,
            buffers,
            &mut advantages,
            &mut returns
        ));
        self.batching_loop(buffers, &advantages, &returns)?;
        Ok(())
    }
}
