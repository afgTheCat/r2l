use crate::ENV_NAME;
use crate::EventBox;
use crate::PPOProgress;

use candle_core::{DType, Device, Error, Tensor};
use r2l_agents::HookResult;
use r2l_agents::candle_agents::LearningModuleKind;
use r2l_agents::candle_agents::ModuleWithValueFunction;
use r2l_agents::candle_agents::ppo::PPOBatchData;
use r2l_agents::candle_agents::ppo::PPOHooks;
use r2l_api::builders::agents::ppo::PPOBuilder;
use r2l_candle_lm::tensors::{PolicyLoss, ValueLoss};
use r2l_core::env_builder::EnvBuilderType;
use r2l_core::on_policy_algorithm::DefaultOnPolicyAlgorightmsHooks5;
use r2l_core::on_policy_algorithm::LearningSchedule;
use r2l_core::on_policy_algorithm::OnPolicyAlgorithm5;
use r2l_core::sampler::FinalSampler;
use r2l_core::sampler::Location;
use r2l_core::sampler::buffer::StepTrajectoryBound;
use r2l_core::{distributions::Policy, utils::rollout_buffer::Advantages};
use std::f64;
use std::sync::Arc;
use std::sync::mpsc::Sender;

impl PPOProgress {
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

#[derive(Debug, Clone)]
struct PPOHook {
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

impl PPOHooks<LearningModuleKind> for PPOHook {
    fn before_learning_hook<B: r2l_core::sampler::buffer::TrajectoryContainer<Tensor = Tensor>>(
        &mut self,
        _agent: &mut r2l_agents::candle_agents::ppo::CandlePPOCore5<LearningModuleKind>,
        buffers: &[B],
        advantages: &mut Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> candle_core::Result<HookResult> {
        self.current_epoch = 0;
        let mut total_rewards: f32 = 0.;
        let mut total_episodes: usize = 0;
        for buffer in buffers {
            total_rewards += buffer.rewards().sum::<f32>();
            total_episodes += buffer.dones().filter(|x| *x).count();
        }
        advantages.normalize();
        let avarage_reward = total_rewards / total_episodes as f32;
        let progress = self.current_rollout as f64 / self.total_rollouts as f64;
        self.progress.avarage_reward = avarage_reward;
        self.progress.progress = progress;
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: r2l_core::sampler::buffer::TrajectoryContainer<Tensor = Tensor>>(
        &mut self,
        _buffers: &[B],
        agent: &mut r2l_agents::candle_agents::ppo::CandlePPOCore5<LearningModuleKind>,
    ) -> candle_core::Result<HookResult> {
        self.current_epoch += 1;
        let should_stop = self.current_epoch == self.total_epochs;
        if should_stop {
            // snapshot the learned things, API can be much better
            self.current_rollout += 1;
            self.progress.std = agent.module.get_policy_ref().std().unwrap();
            self.progress.learning_rate = agent.module.learning_module().policy_learning_rate();
            let progress = self.progress.clear();
            self.tx.send(Box::new(progress)).map_err(Error::wrap)?;
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        agent: &mut r2l_agents::candle_agents::ppo::CandlePPOCore5<LearningModuleKind>,
        policy_loss: &mut PolicyLoss,
        value_loss: &mut ValueLoss,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        let entropy = agent.module.get_policy_ref().entropy().unwrap();
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.ent_coeff, (), device)? * entropy.neg()?)?;
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

pub fn new_train_ppo(tx: Sender<EventBox>) -> anyhow::Result<()> {
    let total_rollouts = 300;
    let ppo_hook = PPOHook::new(10, total_rollouts, 0., 0., 0.01, tx);
    let device = Device::Cpu;
    let env_builder = EnvBuilderType::EnvBuilder {
        builder: Arc::new(r2l_gym::GymEnvBuilder::new(ENV_NAME)),
        n_envs: 5,
    };
    let sampler = FinalSampler::build(
        env_builder,
        StepTrajectoryBound::new(2048),
        None,
        Location::Thread,
    );
    let env_description = sampler.env_description();
    let agent = PPOBuilder::default().build5(&device, &env_description, ppo_hook)?;
    let mut algo = OnPolicyAlgorithm5 {
        sampler,
        agent,
        hooks: DefaultOnPolicyAlgorightmsHooks5::new(LearningSchedule::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }),
    };
    algo.train()
}
