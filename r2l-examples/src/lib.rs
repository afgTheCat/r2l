use candle_core::{DType, Device, Error, Tensor};
use once_cell::sync::Lazy;
use r2l_agents::ppo::hooks::{HookResult, PPOBatchData};
use r2l_agents::ppo::ppo3::PPP3HooksTrait;
use r2l_api::builders::env_pool::VecPoolType;
use r2l_api::builders::{agents::ppo::PPO3Builder, env_pool::SequentialEnvHookTypes};
use r2l_core::distributions::DistributionKind;
use r2l_core::policies::learning_modules::LearningModuleKind;
use r2l_core::{
    Algorithm,
    distributions::Distribution,
    env::RolloutMode,
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm2, OnPolicyHooks},
    // policies::{Policy, PolicyKind},
    tensors::{PolicyLoss, ValueLoss},
    utils::rollout_buffer::{Advantages, RolloutBuffer},
};
use std::sync::{Mutex, mpsc::Sender};
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
        rollout_buffers: &mut Vec<RolloutBuffer>,
        advantages: &mut Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> candle_core::Result<r2l_agents::ppo::hooks::HookResult> {
        self.current_epoch = 0;
        let mut total_rewards: f32 = 0.;
        let mut total_episodes: usize = 0;
        for rb in rollout_buffers {
            total_rewards += rb.rewards.iter().sum::<f32>();
            total_episodes += rb.dones.iter().filter(|x| **x).count();
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
        _rollout_buffers: &Vec<RolloutBuffer>,
    ) -> candle_core::Result<r2l_agents::ppo::hooks::HookResult> {
        self.current_epoch += 1;
        let should_stop = self.current_epoch == self.total_epochs;
        if should_stop {
            // snapshot the learned things, API can be much better
            self.current_rollout += 1;
            self.progress.std = distribution.std()?;
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
        _rollout_batch: &r2l_core::utils::rollout_buffer::RolloutBatch,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<r2l_agents::ppo::hooks::HookResult> {
        let entropy = distribution.entropy()?;
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
        // let approx_kl = (batch_data
        //     .ratio
        //     .exp()?
        //     .sub(&Tensor::ones_like(&batch_data.ratio)?))?
        // .sub(&batch_data.ratio)?
        // .mean_all()?
        // .to_scalar::<f32>()?;
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

#[allow(dead_code)]
#[derive(Default)]
struct AppData {
    current_epoch: usize,                 // to control the learning
    total_epochs: usize,                  // to control the learning
    current_rollout: usize,               // does not need it any more
    total_rollouts: usize,                // does not need it any more
    ent_coeff: f32,                       // to control the learning
    clip_range: f32,                      // for logging
    target_kl: f32,                       // to control the learning
    current_progress_report: PPOProgress, // collect the learning stuff
}

// maybe I will try channels at one point, for now a mutex is fine
// TODO: replace this with the hook data
static SHARED_APP_DATA: Lazy<Mutex<AppData>> = Lazy::new(|| {
    let app_data = AppData::default();
    Mutex::new(app_data)
});

// fn batch_hook(
//     policy: &mut PolicyKind,
//     policy_loss: &mut PolicyLoss,
//     value_loss: &mut ValueLoss,
//     batch_data: &PPOBatchData,
// ) -> candle_core::Result<bool> {
//     let entropy = policy.distribution().entropy()?;
//     let device = entropy.device();
//     let mut app_data = SHARED_APP_DATA.lock().unwrap();
//     let entropy_loss = (Tensor::full(app_data.ent_coeff, (), &device)? * entropy.neg()?)?;
//     app_data.current_progress_report.collect_batch_data(
//         &batch_data.ratio,
//         &entropy_loss,
//         value_loss,
//         policy_loss,
//     )?;
//
//     // TODO: this breaks the computation graph. We need to explore our options here. The most
//     // reasonable choice seems to be that we switch up our hook interface by not only allowing
//     // booleans to be returned, but that seems a lot of work right now
//     // *policy_loss = PolicyLoss(policy_loss.add(&entropy_loss)?);
//
//     // TODO: this seems to slow down the learning process quite a bit. Maybe there is an issue with
//     // the learning rate?
//     // let approx_kl = (batch_data
//     //     .ratio
//     //     .exp()?
//     //     .sub(&Tensor::ones_like(&batch_data.ratio)?))?
//     // .sub(&batch_data.ratio)?
//     // .mean_all()?
//     // .to_scalar::<f32>()?;
//     // Ok(approx_kl > 1.5 * app_data.target_kl)
//     Ok(false)
// }

// fn before_learning_hook(
//     rollout_buffers: &mut Vec<RolloutBuffer>,
//     advantages: &mut Advantages,
// ) -> candle_core::Result<bool> {
//     let mut app_data = SHARED_APP_DATA.lock().unwrap();
//     app_data.current_epoch = 0;
//     let mut total_rewards: f32 = 0.;
//     let mut total_episodes: usize = 0;
//     for rb in rollout_buffers {
//         total_rewards += rb.rewards.iter().sum::<f32>();
//         total_episodes += rb.dones.iter().filter(|x| **x).count();
//     }
//     advantages.normalize();
//     let avarage_reward = total_rewards / total_episodes as f32;
//     let progress = app_data.current_rollout as f64 / app_data.total_rollouts as f64;
//     app_data.current_progress_report.avarage_reward = avarage_reward;
//     app_data.current_progress_report.progress = progress;
//     Ok(false)
// }

// enum AfterLearningHookResult {
//     ShouldStop,
//     ShouldContinue,
// }
//
// #[allow(clippy::ptr_arg)]
// fn after_learning_hook_inner(
//     policy: &mut PolicyKind,
// ) -> candle_core::Result<AfterLearningHookResult> {
//     let mut app_data = SHARED_APP_DATA.lock().unwrap();
//     app_data.current_epoch += 1;
//     let should_stop = app_data.current_epoch == app_data.total_epochs;
//     if should_stop {
//         // snapshot the learned things, API can be much better
//         app_data.current_rollout += 1;
//         // the std after learning
//         app_data.current_progress_report.std = policy.distribution().std()?;
//         app_data.current_progress_report.learning_rate = policy.policy_learning_rate();
//         Ok(AfterLearningHookResult::ShouldStop)
//     } else {
//         Ok(AfterLearningHookResult::ShouldContinue)
//     }
// }

pub fn train_ppo(tx: Sender<EventBox>) -> candle_core::Result<()> {
    let total_rollouts = 300;
    let ppo_hook = PPOHook::new(10, total_rollouts, 0., 0., 0.01, tx);
    // let after_learning_hook =
    //     move |policy: &mut PolicyKind| match after_learning_hook_inner(policy)? {
    //         AfterLearningHookResult::ShouldStop => {
    //             let mut app_data = SHARED_APP_DATA.lock().unwrap();
    //             let progress = app_data.current_progress_report.clear();
    //             tx.send(Box::new(progress)).map_err(Error::wrap)?;
    //             Ok(true)
    //         }
    //         AfterLearningHookResult::ShouldContinue => Ok(false),
    //     };
    let device = Device::Cpu;
    let mut builder = PPO3Builder::default();
    builder.sample_size = 64;

    let env_pool = VecPoolType::Sequential(SequentialEnvHookTypes::None).build(
        &Device::Cpu,
        ENV_NAME.to_owned(),
        10,
    )?;
    let env_description = env_pool.env_description.clone();

    let mut agent = builder.build(&device, &env_description)?;
    agent.hooks = Box::new(ppo_hook);
    // agent.hooks = PPOHooks::empty()
    //     .add_before_learning_hook(before_learning_hook)
    //     .add_batching_hook(batch_hook)
    //     .add_rollout_hook(after_learning_hook);

    let mut algo = OnPolicyAlgorithm2 {
        env_pool,
        agent,
        learning_schedule: LearningSchedule::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        },
        rollout_mode: RolloutMode::StepBound { n_steps: 1024 },
        hooks: OnPolicyHooks::default(),
    };
    algo.train()
}
