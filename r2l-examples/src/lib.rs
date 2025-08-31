use candle_core::{DType, Device, Error, Tensor};
use r2l_agents::ppo::PPP3HooksTrait;
use r2l_agents::ppo::{HookResult, PPOBatchData};
use r2l_api::builders::agents::ppo::PPOBuilder;
use r2l_api::builders::sampler::{EnvBuilderType, EnvPoolType, SamplerType};
use r2l_candle_lm::candle_rollout_buffer::{CandleRolloutBuffer, RolloutBatch};
use r2l_candle_lm::distributions::DistributionKind;
use r2l_candle_lm::learning_module::LearningModuleKind;
use r2l_candle_lm::tensors::{PolicyLoss, ValueLoss};
use r2l_core::on_policy_algorithm::{
    DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm,
};
use r2l_core::{Algorithm, distributions::Distribution, utils::rollout_buffer::Advantages};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::{any::Any, f64};

const ENV_NAME: &str = "Pendulum-v1";

pub type EventBox = Box<dyn Any + Send + Sync>;

#[derive(Debug, Default, Clone)]
pub struct PPOProgress {
    pub clip_fractions: Vec<f32>,
    pub entropy_losses: Vec<f32>,
    pub policy_losses: Vec<f32>,
    pub value_losses: Vec<f32>,
    pub clip_range: f32,
    pub approx_kl: f32,
    pub explained_variance: f32,
    pub progress: f64,
    pub std: f32,
    pub avarage_reward: f32,
    pub learning_rate: f64,
}

#[derive(Debug, Clone)]
pub struct PPOHook {
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

impl PPOHook {
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

impl PPP3HooksTrait<DistributionKind, LearningModuleKind> for PPOHook {
    fn before_learning_hook(
        &mut self,
        _learning_module: &mut LearningModuleKind,
        _distribution: &DistributionKind,
        rollout_buffers: &mut Vec<CandleRolloutBuffer>,
        advantages: &mut Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> candle_core::Result<HookResult> {
        self.current_epoch = 0;
        let mut total_rewards: f32 = 0.;
        let mut total_episodes: usize = 0;
        for rb in rollout_buffers {
            total_rewards += rb.0.rewards.iter().sum::<f32>();
            total_episodes += rb.0.dones.iter().filter(|x| **x).count();
        }
        advantages.normalize();
        let avarage_reward = total_rewards / total_episodes as f32;
        let progress = self.current_rollout as f64 / self.total_rollouts as f64;
        self.progress.avarage_reward = avarage_reward;
        self.progress.progress = progress;
        Ok(HookResult::Continue)
    }

    fn rollout_hook(
        &mut self,
        learning_module: &mut LearningModuleKind,
        distribution: &DistributionKind,
        _rollout_buffers: &Vec<CandleRolloutBuffer>,
    ) -> candle_core::Result<HookResult> {
        self.current_epoch += 1;
        let should_stop = self.current_epoch == self.total_epochs;
        if should_stop {
            // snapshot the learned things, API can be much better
            self.current_rollout += 1;
            self.progress.std = distribution.std().unwrap();
            self.progress.learning_rate = learning_module.policy_learning_rate();
            let progress = self.progress.clear();
            self.tx.send(Box::new(progress)).map_err(Error::wrap)?;
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        _learning_module: &mut LearningModuleKind,
        distribution: &DistributionKind,
        _rollout_batch: &RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        let entropy = distribution.entropy().unwrap();
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.ent_coeff, (), &device)? * entropy.neg()?)?;
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

impl PPOProgress {
    pub fn clear(&mut self) -> Self {
        std::mem::take(self)
    }

    pub fn collect_batch_data(
        &mut self,
        ratio: &Tensor,
        entropy_loss: &Tensor,
        value_loss: &Tensor,
        policy_loss: &Tensor,
    ) -> candle_core::Result<()> {
        let clip_fraction = (ratio - 1.)?
            .abs()?
            .gt(self.clip_range)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        self.clip_fractions.push(clip_fraction);
        self.entropy_losses.push(entropy_loss.to_scalar()?);
        self.value_losses.push(value_loss.to_scalar()?);
        self.policy_losses.push(policy_loss.to_scalar()?);
        Ok(())
    }
}

pub fn train_ppo(tx: Sender<EventBox>) -> anyhow::Result<()> {
    let total_rollouts = 300;
    let ppo_hook = PPOHook::new(10, total_rollouts, 0., 0., 0.01, tx);
    let device = Device::Cpu;
    let sampler = SamplerType {
        capacity: 2048,
        hook_options: Default::default(),
        env_pool_type: EnvPoolType::VecVariable, // TODO: Change this to VecVariable
    }
    .build_with_builder_type(
        EnvBuilderType::EnvBuilder {
            builder: Arc::new(ENV_NAME.to_owned()),
            n_envs: 1,
        },
        &device,
    );
    let env_description = sampler.env_description();
    let mut agent = PPOBuilder::default().build(&device, &env_description)?;
    agent.hooks = Box::new(ppo_hook);
    let mut algo = OnPolicyAlgorithm {
        sampler,
        agent,
        hooks: DefaultOnPolicyAlgorightmsHooks::new(LearningSchedule::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }),
    };
    algo.train()
}
