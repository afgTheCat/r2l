use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::AutodiffBackend;
use r2l_burn_lm::{BurnModuleWithValueFunction, learning_module::PolicyValuesLosses};
use r2l_core::{
    agents::Agent,
    distributions::Policy,
    policies::ValueFunction,
    sampler::buffer::TrajectoryContainer,
    utils::rollout_buffer::{Advantages, Logps, Returns},
};
use std::marker::PhantomData;

use crate::{
    BatchIndexIterator, HookResult, buffers_advantages_and_returns, logps,
    ppo::{PPOBatchData, PPOHooksTrait, PPOParams},
    sample,
};

fn uplift_tensor<const N: usize, B: AutodiffBackend>(
    tensor: &BurnTensor<B::InnerBackend, N>,
) -> BurnTensor<B, N> {
    BurnTensor::from_data(tensor.to_data(), &Default::default())
}

pub struct BurnPPO<B: AutodiffBackend, M: BurnModuleWithValueFunction<B>, H: PPOHooksTrait<M>> {
    ppo: PPOParams<M>,
    hooks: H,
    _phantom: PhantomData<B>,
}

impl<B: AutodiffBackend, M: BurnModuleWithValueFunction<B>, H: PPOHooksTrait<M>> BurnPPO<B, M, H> {
    fn batch_loop<C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        buffers: &[C],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.ppo.sample_size);
        let ppo = &mut self.ppo;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indices, uplift_tensor);
            let advantages = advantages.sample(&indices);
            let advantages = BurnTensor::from_data(advantages.as_slice(), &Default::default());
            let logp_old = logps.sample(&indices);
            let logp_old = BurnTensor::from_data(logp_old.as_slice(), &Default::default());
            let returns = returns.sample(&indices);
            let returns = BurnTensor::from_data(returns.as_slice(), &Default::default());
            let logp = ppo.module.get_policy().log_probs(&observations, &actions)?;
            let values_pred = ppo.module.value_func().calculate_values(&observations)?;
            let value_diff = returns.clone() - values_pred.clone();
            let value_loss = (value_diff.clone() * value_diff).mean();
            let logp_diff = logp.clone() - logp_old;
            let ratio = logp_diff.clone().exp();
            let clip_adv = ratio
                .clone()
                .clamp(1. - ppo.clip_range, 1. + ppo.clip_range)
                * advantages.clone();
            let policy_loss = (-(ratio.clone() * advantages).min_pair(clip_adv)).mean();
            let ppo_data = PPOBatchData {
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            let mut losses = PolicyValuesLosses::new(policy_loss, value_loss);
            match self.hooks.batch_hook(ppo, &mut losses, &ppo_data)? {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
            ppo.module.update(losses)?;
        }
    }

    fn learning_loop<C: TrajectoryContainer<Tensor = BurnTensor<B::InnerBackend, 1>>>(
        &mut self,
        buffers: &[C],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> anyhow::Result<()> {
        loop {
            self.batch_loop(buffers, &advantages, &logps, &returns)?;
            crate::process_hook_result!(self.hooks.rollout_hook(buffers, &mut self.ppo));
        }
    }
}

impl<
    B: burn::tensor::backend::AutodiffBackend,
    M: BurnModuleWithValueFunction<B>,
    H: PPOHooksTrait<M>,
> Agent for BurnPPO<B, M, H>
{
    type Tensor = M::InferenceTensor;

    type Policy = M::InferencePolicy;

    fn policy(&self) -> Self::Policy {
        self.ppo.module.get_inference_policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns(
            buffers,
            self.ppo.module.value_func(),
            self.ppo.gamma,
            self.ppo.lambda,
            uplift_tensor,
        )?;
        crate::process_hook_result!(self.hooks.before_learning_hook(
            &mut self.ppo,
            buffers,
            &mut advantages,
            &mut returns
        ));
        let logps = logps(buffers, &self.policy());
        self.learning_loop(buffers, advantages, returns, logps)?;
        Ok(())
    }
}
