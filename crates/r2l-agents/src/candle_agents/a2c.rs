use crate::candle_agents::ModuleWithValueFunction;
use crate::{BatchIndexIterator, HookResult, buffers_advantages_and_returns, sample};
use anyhow::Result;
use candle_core::{Device, Error, Tensor as CandleTensor};
use r2l_candle_lm::learning_module::PolicyValuesLosses;
use r2l_core::distributions::Policy;
use r2l_core::policies::ValueFunction;
use r2l_core::utils::rollout_buffer::{Advantages, Returns};
use r2l_core::{agents::Agent, sampler::buffer::TrajectoryContainer};

pub trait A2CHooks<M: ModuleWithValueFunction> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        _agent: &mut CandleA2CCore5<M>,
        _buffers: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct DefaultA2CHooks;

impl<M: ModuleWithValueFunction> A2CHooks<M> for DefaultA2CHooks {}

pub struct CandleA2CCore5<M: ModuleWithValueFunction> {
    pub module: M,
    pub device: Device,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

pub struct CandleA2C5<M: ModuleWithValueFunction, H: A2CHooks<M>> {
    pub a2c: CandleA2CCore5<M>,
    pub hooks: H,
}

impl<M: ModuleWithValueFunction, H: A2CHooks<M>> CandleA2C5<M, H> {
    pub fn new(a2c: CandleA2CCore5<M>, hooks: H) -> Self {
        Self { a2c, hooks }
    }

    fn batching_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        returns: &Returns,
    ) -> Result<()> {
        let mut index_iterator = BatchIndexIterator::new(buffers, self.a2c.sample_size);
        let a2c = &mut self.a2c;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(buffers, &indices, |t| t.clone());
            let advantages = advantages.sample(&indices);
            let advantages = CandleTensor::from_slice(&advantages, advantages.len(), &a2c.device)?;
            let returns = returns.sample(&indices);
            let returns = CandleTensor::from_slice(&returns, returns.len(), &a2c.device)?;
            let logps = a2c
                .module
                .get_policy_ref()
                .log_probs(&observations, &actions)
                .map_err(Error::wrap)?;
            let values_pred = a2c
                .module
                .value_func()
                .calculate_values(&observations)
                .map_err(Error::wrap)?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean_all()?;
            let policy_loss = advantages.mul(&logps)?.neg()?.mean_all()?;
            a2c.module
                .update(PolicyValuesLosses {
                    policy_loss,
                    value_loss,
                })
                .map_err(Error::wrap)?;
        }
    }
}

impl<M: ModuleWithValueFunction, H: A2CHooks<M>> Agent for CandleA2C5<M, H> {
    type Tensor = CandleTensor;
    type Policy = <M as ModuleWithValueFunction>::P;

    fn policy(&self) -> Self::Policy {
        self.a2c.module.get_inference_policy()
    }

    fn learn<C: TrajectoryContainer<Tensor = Self::Tensor>>(
        &mut self,
        buffers: &[C],
    ) -> anyhow::Result<()> {
        let (mut advantages, mut returns) = buffers_advantages_and_returns(
            buffers,
            self.a2c.module.value_func(),
            self.a2c.gamma,
            self.a2c.lambda,
            |t| t.clone(),
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
