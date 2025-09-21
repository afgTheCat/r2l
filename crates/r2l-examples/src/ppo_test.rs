use crate::{EventBox, PPOProgress};
use candle_core::{Device, Tensor as CandleTensor};
use r2l_agents::{
    LearningModuleKind,
    candle_agents::{ModuleWithValueFunction, ppo::HookResult, ppo2::PPOHooksTrait2},
};
use r2l_api::builders::{
    agents::ppo::PPOBuilder,
    sampler::{EnvBuilderType, EnvPoolType, SamplerType, SamplerType2},
    sampler_hooks2::EvaluatorNormalizerOptions,
};
use r2l_core::{
    distributions::Policy,
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks2, LearningSchedule, OnPolicyAlgorithm2},
    sampler2::env_pools::builder::EnvPoolBuilder,
};
use std::sync::{Arc, mpsc::Sender};

const ENV_NAME: &str = "Pendulum-v1";

pub struct PPOHook2 {
    current_epoch: usize,   // track the current epoch
    total_epochs: usize,    // current epoch we are in
    current_rollout: usize, // track the current rollout
    total_rollouts: usize,  // current rollout
    ent_coeff: f32,         // entropy coefficient
    clip_range: f32,        // for logging
    target_kl: f32,         // to control the learning

    pub progress: PPOProgress, // I suppose this should not really be here
    pub tx: Sender<EventBox>,
}

impl PPOHook2 {
    fn new(
        total_epochs: usize,
        total_rollouts: usize,
        ent_coeff: f32,
        clip_range: f32,
        target_kl: f32,
        tx: Sender<EventBox>,
    ) -> Self {
        Self {
            current_epoch: 0,
            total_epochs,
            current_rollout: 0,
            total_rollouts,
            ent_coeff,
            clip_range,
            target_kl,
            progress: PPOProgress::default(),
            tx,
        }
    }
}

impl PPOHooksTrait2<LearningModuleKind> for PPOHook2 {
    fn before_learning_hook<B: r2l_core::sampler2::Buffer<Tensor = CandleTensor>>(
        &mut self,
        _agent: &mut r2l_agents::candle_agents::ppo2::CandlePPOCore2<LearningModuleKind>,
        buffers: &[B],
        advantages: &mut r2l_core::utils::rollout_buffer::Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> candle_core::Result<r2l_agents::candle_agents::ppo::HookResult> {
        self.current_epoch = 0;
        let mut total_rewards: f32 = 0.;
        let mut total_episodes: usize = 0;
        for rb in buffers {
            total_rewards += rb.rewards().iter().sum::<f32>();
            total_episodes += rb.dones().iter().filter(|x| **x).count();
        }
        advantages.normalize();
        let avarage_reward = total_rewards / total_episodes as f32;
        let progress = self.current_rollout as f64 / self.total_rollouts as f64;
        self.progress.avarage_reward = avarage_reward;
        self.progress.progress = progress;
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: r2l_core::sampler2::Buffer<Tensor = CandleTensor>>(
        &mut self,
        buffers: &[B],
        agent: &mut r2l_agents::candle_agents::ppo2::CandlePPOCore2<LearningModuleKind>,
    ) -> candle_core::Result<HookResult> {
        self.current_epoch += 1;
        let should_stop = self.current_epoch == self.total_epochs;
        if should_stop {
            // snapshot the learned things, API can be much better
            self.current_rollout += 1;
            self.progress.std = agent.module.get_policy_ref().std().unwrap();
            self.progress.learning_rate = agent.module.learning_module().policy_learning_rate();
            let progress = self.progress.clear();
            self.tx.send(Box::new(progress)).unwrap();
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        agent: &mut r2l_agents::candle_agents::ppo2::CandlePPOCore2<LearningModuleKind>,
        policy_loss: &mut r2l_candle_lm::tensors::PolicyLoss,
        value_loss: &mut r2l_candle_lm::tensors::ValueLoss,
        data: &r2l_agents::candle_agents::ppo::PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        let entropy = agent.module.get_policy_ref().entropy().unwrap();
        let device = entropy.device();
        let entropy_loss = (CandleTensor::full(self.ent_coeff, (), &device)? * entropy.neg()?)?;
        self.progress
            .collect_batch_data(&data.ratio, &entropy_loss, value_loss, policy_loss)?;
        // TODO: this breaks the computation graph. We need to explore our options here. The most
        // reasonable choice seems to be that we switch up our hook interface by not only allowing
        // booleans to be returned, but that seems a lot of work right now
        // *policy_loss = PolicyLoss(policy_loss.add(&entropy_loss)?);

        // TODO: this seems to slow down the learning process quite a bit. Maybe there is an issue with
        // the learning rate?
        // let approx_kl = (data
        //     .ratio
        //     .detach()
        //     .exp()?
        //     .sub(&Tensor::ones_like(&data.ratio.detach())?))?
        // .sub(&data.ratio.detach())?
        // .mean_all()?
        // .to_scalar::<f32>()?;
        // if approx_kl > 1.5 * self.target_kl {
        // } else {
        // }
        // Ok(approx_kl > 1.5 * app_data.target_kl)
        Ok(HookResult::Continue)
    }
}

pub fn train_ppo2(tx: Sender<EventBox>) -> anyhow::Result<()> {
    let total_rollouts = 300;
    let ppo_hook = PPOHook2::new(10, total_rollouts, 0., 0., 0.01, tx);
    let device = Device::Cpu;
    let sampler = SamplerType2 {
        env_pool_builder: EnvPoolBuilder::default(),
        preprocessor_options: EvaluatorNormalizerOptions::default(),
    }
    .build(EnvBuilderType::EnvBuilder {
        builder: Arc::new(r2l_gym::GymEnvBuilder::new(ENV_NAME)),
        n_envs: 1,
    });
    let env_description = sampler.env_description();
    let agent = PPOBuilder::default().build2(&device, &env_description, ppo_hook)?;
    let mut algo = OnPolicyAlgorithm2 {
        sampler,
        agent,
        hooks: DefaultOnPolicyAlgorightmsHooks2::new(LearningSchedule::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }),
    };
    algo.train()
}

#[cfg(test)]
mod test {
    use std::sync::mpsc;
    use std::sync::mpsc::{Receiver, Sender};

    use crate::{EventBox, ppo_test::train_ppo2};

    #[test]
    fn train_ppo_complicated() {
        let (event_tx, event_rx): (Sender<EventBox>, Receiver<EventBox>) = mpsc::channel();
        train_ppo2(event_tx).unwrap();
    }
}
