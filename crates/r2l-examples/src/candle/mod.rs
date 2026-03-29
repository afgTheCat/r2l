use crate::ENV_NAME;
use crate::EventBox;
use crate::PPOProgress;

use candle_core::{Device, Error, Tensor};
use r2l_agents::HookResult;
use r2l_agents::candle_agents::LearningModuleKind;
use r2l_agents::candle_agents::ModuleWithValueFunction;
use r2l_agents::candle_agents::ppo::CandlePPOCore;
use r2l_agents::candle_agents::ppo::PPOBatchData;
use r2l_agents::candle_agents::ppo::PPOHooks;
use r2l_api::builders::agents::ppo::PPOBuilder;
use r2l_candle_lm::learning_module::PolicyValuesLosses;
use r2l_core::env_builder::EnvBuilderType;
use r2l_core::on_policy_algorithm::DefaultOnPolicyAlgorightmsHooks5;
use r2l_core::on_policy_algorithm::LearningSchedule;
use r2l_core::on_policy_algorithm::OnPolicyAlgorithm5;
use r2l_core::sampler::FinalSampler;
use r2l_core::sampler::Location;
use r2l_core::sampler::buffer::StepTrajectoryBound;
use r2l_core::sampler::buffer::TrajectoryContainer;
use r2l_core::{distributions::Policy, utils::rollout_buffer::Advantages};
use std::f64;
use std::sync::Arc;
use std::sync::mpsc::Sender;

impl PPOProgress {
    pub fn collect_batch_data(
        &mut self,
        entropy_loss: f32,
        value_loss: f32,
        policy_loss: f32,
        clip_fraction: f32,
        clip_range: f32,
        approx_kl: f32,
    ) -> candle_core::Result<()> {
        self.clip_fractions.push(clip_fraction);
        self.entropy_losses.push(entropy_loss);
        self.value_losses.push(value_loss);
        self.policy_losses.push(policy_loss);
        self.clip_range.push(clip_range);
        self.approx_kl.push(approx_kl);
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
    max_grad_norm: f32,     // gradient clipping threshold
    target_kl: f32,         // to control the learning
    vf_coef: f32,

    pub progress: PPOProgress, // I suppose this should not really be here
    pub tx: Sender<EventBox>,
}

impl PPOHook {
    fn new(
        total_epochs: usize,
        total_rollouts: usize,
        ent_coeff: f32,
        max_grad_norm: f32,
        target_kl: f32,
        tx: Sender<EventBox>,
    ) -> Self {
        Self {
            current_epoch: 0,
            total_epochs,
            current_rollout: 0,
            total_rollouts,
            ent_coeff,
            max_grad_norm,
            target_kl,
            vf_coef: 1.,
            progress: PPOProgress::default(),
            tx,
        }
    }
}

impl PPOHooks<LearningModuleKind> for PPOHook {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = Tensor>>(
        &mut self,
        agent: &mut CandlePPOCore<LearningModuleKind>,
        buffers: &[B],
        advantages: &mut Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> candle_core::Result<HookResult> {
        // This is a bit redundant, the idea here would be for us to be able to change this on the fly.
        agent
            .module
            .learning_module
            .set_grad_clipping(Some(self.max_grad_norm));
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

    fn rollout_hook<B: TrajectoryContainer<Tensor = Tensor>>(
        &mut self,
        _buffers: &[B],
        agent: &mut CandlePPOCore<LearningModuleKind>,
    ) -> candle_core::Result<HookResult> {
        self.current_epoch += 1;
        let should_stop = self.current_epoch == self.total_epochs;
        if should_stop {
            // snapshot the learned things, API can be much better
            self.current_rollout += 1;
            self.progress.std = agent.module.get_policy_ref().std().unwrap();
            self.progress.learning_rate = agent.module.policy_learning_rate();
            let progress = self.progress.clear();
            self.tx.send(Box::new(progress)).map_err(Error::wrap)?;
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        agent: &mut CandlePPOCore<LearningModuleKind>,
        losses: &mut PolicyValuesLosses,
        data: &PPOBatchData,
    ) -> candle_core::Result<HookResult> {
        losses.set_vf_coeff(Some(self.vf_coef));
        let policy_loss = losses.policy_loss.to_scalar()?;
        let value_loss = losses.value_loss.to_scalar()?;
        let entropy = agent.module.get_policy_ref().entropy().unwrap();
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.ent_coeff, (), device)? * entropy.neg()?)?;
        let entropy = entropy_loss.to_scalar()?;
        let clip_fraction = data.clip_fraction(agent.clip_range)?;
        losses.apply_entropy(entropy_loss).map_err(Error::wrap)?;
        let approx_kl = data.approx_kl()?;
        self.progress.collect_batch_data(
            entropy,
            value_loss,
            policy_loss,
            clip_fraction,
            agent.clip_range,
            approx_kl,
        )?;
        if approx_kl > 1.5 * self.target_kl {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }
}

const ENT_COEFF: f32 = 0.001;
const MAX_GRAD_NORM: f32 = 0.5;
const TARGET_KL: f32 = 0.01;

pub fn train_ppo(tx: Sender<EventBox>) -> anyhow::Result<()> {
    let total_rollouts = 300;
    let ppo_hook = PPOHook::new(10, total_rollouts, ENT_COEFF, MAX_GRAD_NORM, TARGET_KL, tx);
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
    let agent = PPOBuilder::default().build(&device, &env_description, ppo_hook)?;
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
