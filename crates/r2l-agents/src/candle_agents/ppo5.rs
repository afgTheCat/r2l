use crate::CandleTensor;
use crate::candle_agents::ppo::PPOBatchData;
use crate::candle_agents::{ModuleWithValueFunction, ppo::HookResult};
use burn::tensor::Tensor;
use candle_core::{Device, Result};
use r2l_candle_lm::tensors::PolicyLoss;
use r2l_candle_lm::tensors::ValueLoss;
use r2l_core::distributions::Policy;
use r2l_core::policies::ValueFunction;
use r2l_core::utils::rollout_buffer::{Advantages, Logps, Returns};
use r2l_core::{agents::Agent5, sampler5::buffer::TrajectoryContainer};

macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}

pub trait PPOHooksTrait5<M: ModuleWithValueFunction> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        agent: &mut CandlePPOCore5<M>,
        buffers: &[B],
        advantages: &mut Advantages,
        returns: &mut Returns,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        agent: &mut CandlePPOCore5<M>,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        agent: &mut CandlePPOCore5<M>,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

pub struct CandlePPOCore5<M: ModuleWithValueFunction> {
    pub module: M,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
    pub device: Device,
}

pub struct CandlePPO5<M: ModuleWithValueFunction, H: PPOHooksTrait5<M>> {
    pub ppo: CandlePPOCore5<M>,
    pub hooks: H,
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait5<M>> CandlePPO5<M, H> {
    fn batching_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> Result<()> {
        todo!()
        // let mut index_iterator = buffers.index_iterator(self.ppo.sample_size);
        // let ppo = &mut self.ppo;
        // loop {
        //     let Some(indicies) = index_iterator.iter() else {
        //         return Ok(());
        //     };
        //     let (observations, actions) = buffers.sample(&indicies);
        //     let advantages = advantages.sample(&indicies);
        //     let advantages = CandleTensor::from_slice(&advantages, advantages.len(), &ppo.device)?;
        //     let logp_old = logps.sample(&indicies);
        //     let logp_old = CandleTensor::from_slice(&logp_old, logp_old.len(), &ppo.device)?;
        //     let returns = returns.sample(&indicies);
        //     let returns = CandleTensor::from_slice(&returns, returns.len(), &ppo.device)?;
        //     let logp = Logp(
        //         ppo.module
        //             .get_policy_ref()
        //             .log_probs(&observations, &actions)?,
        //     );
        //     let values_pred = ValuesPred(ppo.module.value_func().calculate_values(&observations)?);
        //     let mut value_loss = ValueLoss(returns.sub(&values_pred)?.sqr()?.mean_all()?);
        //     let logp_diff = LogpDiff((logp.deref() - &logp_old)?);
        //     let ratio = logp_diff.exp()?;
        //     let clip_adv =
        //         (ratio.clamp(1. - ppo.clip_range, 1. + ppo.clip_range)? * advantages.clone())?;
        //     let mut policy_loss = PolicyLoss(
        //         CandleTensor::minimum(&(&ratio * &advantages)?, &clip_adv)?
        //             .neg()?
        //             .mean_all()?,
        //     );
        //     let ppo_data = PPOBatchData {
        //         logp,
        //         values_pred,
        //         logp_diff,
        //         ratio,
        //     };
        //     let hook_result =
        //         self.hooks
        //             .batch_hook(ppo, &mut policy_loss, &mut value_loss, &ppo_data)?;
        //     ppo.module.learning_module().update(PolicyValuesLosses {
        //         policy_loss,
        //         value_loss,
        //     })?;
        //     match hook_result {
        //         HookResult::Break => return Ok(()),
        //         HookResult::Continue => {}
        //     }
        // }
    }

    fn learning_loop<B: TrajectoryContainer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> Result<()> {
        loop {
            // self.batching_loop(&buffers, &advantages, &logps, &returns)?;
            let rollout_hook_res = self.hooks.rollout_hook(&buffers, &mut self.ppo);
            process_hook_result!(rollout_hook_res);
        }
    }
}

impl<M: ModuleWithValueFunction, H: PPOHooksTrait5<M>> Agent5 for CandlePPO5<M, H> {
    type Tensor = CandleTensor;

    type Policy = <M as ModuleWithValueFunction>::P;

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
        )?;
        process_hook_result!(self.hooks.before_learning_hook(
            &mut self.ppo,
            &buffers,
            &mut advantages,
            &mut returns
        ));
        let logps = logps(buffers, &self.policy());
        self.learning_loop(buffers, advantages, returns, logps)?;
        todo!()
    }
}

// TODO: should be somewhere else
pub fn buffer_advantages_and_returns(
    buffer: &impl TrajectoryContainer<Tensor = CandleTensor>,
    value_func: &impl ValueFunction<Tensor = CandleTensor>,
    gamma: f32,
    lambda: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut states = buffer.states().map(|t| t.clone()).collect::<Vec<_>>();
    states.push(buffer.next_states().last().unwrap().clone());
    let values_stacked = value_func.calculate_values(&states).unwrap();
    let values: Vec<f32> = values_stacked.to_vec1()?;
    let total_steps = buffer.rewards().count();
    let mut advantages: Vec<f32> = vec![0.; total_steps];
    let mut returns: Vec<f32> = vec![0.; total_steps];
    let mut last_gae_lam: f32 = 0.;

    let mut dones = buffer
        .terminated()
        .zip(buffer.trancuated())
        .map(|(terminated, trancuated)| terminated || trancuated);

    for i in (0..total_steps).rev() {
        let next_non_terminal = if dones.nth(i).unwrap() {
            last_gae_lam = 0.;
            0f32
        } else {
            1.
        };
        let delta = buffer.rewards().nth(i).unwrap() + next_non_terminal * gamma * values[i + 1]
            - values[i];
        last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
        advantages[i] = last_gae_lam;
        returns[i] = last_gae_lam + values[i];
    }
    Ok((advantages, returns))
}

pub fn buffers_advantages_and_returns<B: TrajectoryContainer<Tensor = CandleTensor>>(
    buffers: &[B],
    value_func: &impl ValueFunction<Tensor = CandleTensor>,
    gamma: f32,
    lambda: f32,
) -> Result<(Advantages, Returns)> {
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];
    for buffer in buffers {
        let (advantages, returns) =
            buffer_advantages_and_returns(buffer, value_func, gamma, lambda)?;
        advantage_vec.push(advantages);
        returns_vec.push(returns);
    }
    Ok((Advantages(advantage_vec), Returns(returns_vec)))
}

pub fn logps<B: TrajectoryContainer<Tensor = CandleTensor>>(
    buffers: &[B],
    policy: &impl Policy<Tensor = CandleTensor>,
) -> Logps {
    todo!()
}

// pub fn logps<PT: R2lTensor + From<B::Tensor>>(
//         &self,
//         policy: &impl Policy<Tensor = PT>,
//     ) -> Logps {
//         match self {
//             Self::RefCounted(buffers) => {
//                 let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
//                 let mut logps = vec![];
//                 for buff in &buffers {
//                     let states = buff.convert_states();
//                     let actions = buff.convert_actions();
//                     let logp = policy
//                         .log_probs(&states, &actions)
//                         .map(|t| t.to_vec())
//                         .unwrap();
//                     logps.push(logp);
//                 }
//                 Logps(logps)
//             }
//             Self::AtomicRefCounted(buffers) => {
//                 let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
//                 let mut logps = vec![];
//                 for buff in &buffers {
//                     let states = buff.convert_states();
//                     let actions = buff.convert_actions();
//                     let logp = policy
//                         .log_probs(&states, &actions)
//                         .map(|t| t.to_vec())
//                         .unwrap();
//                     logps.push(logp);
//                 }
//                 Logps(logps)
//             }
//         }
//     }
