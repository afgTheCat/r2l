use r2l_candle_lm::CandleModuleWithValueFunction;
use r2l_core::{
    agents::Agent,
    sampler::buffer::TrajectoryContainer,
    utils::rollout_buffer::{Advantages, Logps, Returns},
};

use crate::ppo::{PPOModule, PPOHooksTrait, PPOParams};
use crate::{BatchIndexIterator, HookResult, buffers_advantages_and_returns, logps, sample};
use candle_core::Tensor as CandleTensor;

// Something like this might just work
pub struct CandlePPO<M: CandleModuleWithValueFunction, H: PPOHooksTrait<M>> {
    ppo: PPOParams<M>,
    hooks: H,
    device: candle_core::Device,
}

pub trait CandlePPOHooksTrait<M: CandleModuleWithValueFunction>: PPOHooksTrait<M> {}

impl<M: CandleModuleWithValueFunction, H: PPOHooksTrait<M>> CandlePPO<M, H> {
    fn batch_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.ppo.sample_size);
        let ppo = &mut self.ppo;
        loop {
            let Some(indicies) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indicies, |t| t.clone());
            let advantages = advantages.sample(&indicies);
            let advantages = CandleTensor::from_slice(&advantages, advantages.len(), &self.device)?;
            let logp_old = logps.sample(&indicies);
            let logp_old = CandleTensor::from_slice(&logp_old, logp_old.len(), &self.device)?;
            let returns = returns.sample(&indicies);
            let returns = CandleTensor::from_slice(&returns, returns.len(), &self.device)?;
            let (mut losses, ppo_data) = ppo.module.ppo_batch_data_and_losses(
                &observations,
                &actions,
                &advantages,
                &logp_old,
                &returns,
                ppo.clip_range,
            )?;
            match self.hooks.batch_hook(ppo, &mut losses, &ppo_data)? {
                HookResult::Break => return Ok(()),
                HookResult::Continue => {}
            }
            ppo.module.update(losses)?;
        }
    }

    fn learning_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> anyhow::Result<()> {
        loop {
            self.batch_loop(buffers, &advantages, &logps, &returns)?;
            let rollout_hook_res = self.hooks.rollout_hook(buffers, &mut self.ppo);
            crate::process_hook_result!(rollout_hook_res);
        }
    }
}

impl<M: CandleModuleWithValueFunction, H: PPOHooksTrait<M>> Agent for CandlePPO<M, H> {
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
            |t| t.clone(),
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
