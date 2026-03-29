use crate::{ENV_NAME, EventBox, PPOProgress};
use burn::{
    backend::{Autodiff, NdArray},
    grad_clipping::GradientClipping,
    optim::AdamWConfig,
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_agents::{
    HookResult,
    burn_agents::ppo::{BurnPPO, BurnPPOCore, BurnPPOHooksTrait, PPOBatchData},
};
use r2l_burn_lm::{
    distributions::diagonal_distribution::DiagGaussianDistribution,
    learning_module::{BurnPolicy, ParalellActorCriticLM, ParalellActorModel, PolicyValuesLosses},
};
use r2l_core::{
    env_builder::EnvBuilderType,
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks5, LearningSchedule, OnPolicyAlgorithm5},
    sampler::{FinalSampler, Location, buffer::StepTrajectoryBound},
    utils::rollout_buffer::{Advantages, Returns},
};
use std::{f64, sync::Arc, sync::mpsc::Sender};

type BurnBackend = Autodiff<NdArray>;

#[derive(Clone)]
struct PPOHook {
    current_epoch: usize,
    total_epochs: usize,
    current_rollout: usize,
    total_rollouts: usize,
    ent_coeff: f32,
    target_kl: f32,
    max_grad_norm: GradientClipping,
    vf_coef: f32,

    pub progress: PPOProgress,
    pub tx: Sender<EventBox>,
}

impl PPOHook {
    fn new(
        total_epochs: usize,
        total_rollouts: usize,
        ent_coeff: f32,
        target_kl: f32,
        max_grad_norm: GradientClipping,
        tx: Sender<EventBox>,
    ) -> Self {
        Self {
            current_epoch: 0,
            total_epochs,
            current_rollout: 0,
            total_rollouts,
            ent_coeff,
            target_kl,
            vf_coef: 1.,
            progress: PPOProgress::default(),
            tx,
            max_grad_norm,
        }
    }
}

fn scalar<B: burn::prelude::Backend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor.to_data().to_vec().unwrap()[0]
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> BurnPPOHooksTrait<B, D> for PPOHook {
    fn before_learning_hook<
        T: r2l_core::sampler::buffer::TrajectoryContainer<Tensor = Tensor<B::InnerBackend, 1>>,
    >(
        &mut self,
        agent: &mut BurnPPOCore<B, D>,
        buffers: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        agent.lm.set_grad_clipping(self.max_grad_norm.clone());
        self.current_epoch = 0;
        let mut total_rewards: f32 = 0.;
        let mut total_episodes: usize = 0;
        for buffer in buffers {
            total_rewards += buffer.rewards().sum::<f32>();
            total_episodes += buffer.dones().filter(|done| *done).count();
        }
        advantages.normalize();
        let avarage_reward = total_rewards / total_episodes as f32;
        let progress = self.current_rollout as f64 / self.total_rollouts as f64;
        self.progress.avarage_reward = avarage_reward;
        self.progress.progress = progress;
        Ok(HookResult::Continue)
    }

    fn rollout_hook<
        T: r2l_core::sampler::buffer::TrajectoryContainer<Tensor = Tensor<B::InnerBackend, 1>>,
    >(
        &mut self,
        agent: &mut BurnPPOCore<B, D>,
        _buffers: &[T],
    ) -> candle_core::Result<HookResult> {
        self.current_epoch += 1;
        let should_stop = self.current_epoch == self.total_epochs;
        if should_stop {
            self.current_rollout += 1;
            self.progress.std = agent.lm.model.distr.std().unwrap();
            // TODO: we should get this from the model, not hard code it here
            self.progress.learning_rate = 3e-4;
            let progress = self.progress.clear();
            self.tx
                .send(Box::new(progress))
                .map_err(candle_core::Error::wrap)?;
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        agent: &mut BurnPPOCore<B, D>,
        losses: &mut PolicyValuesLosses<B>,
        data: &PPOBatchData<B>,
    ) -> candle_core::Result<HookResult> {
        losses.set_vf_coeff(Some(self.vf_coef));
        let entropy = agent.lm.model.distr.entropy().unwrap();
        let entropy_loss = entropy.neg() * self.ent_coeff;
        let entropy = scalar(&entropy_loss);
        let policy_loss = scalar(&losses.policy_loss);
        let value_loss = scalar(&losses.value_loss);

        let ratio: Vec<f32> = data.ratio.to_data().to_vec().unwrap();
        let clip_fraction = ratio
            .iter()
            .filter(|value| (**value - 1.).abs() > agent.clip_range)
            .count() as f32
            / ratio.len() as f32;
        losses.apply_entropy(entropy_loss);
        let log_ratio: Vec<f32> = data.logp_diff.to_data().to_vec().unwrap();
        let approx_kl = ratio
            .iter()
            .zip(log_ratio.iter())
            .map(|(ratio, log_ratio)| (ratio - 1.) - log_ratio)
            .sum::<f32>()
            / ratio.len() as f32;
        self.progress.clip_fractions.push(clip_fraction);
        self.progress.entropy_losses.push(entropy);
        self.progress.value_losses.push(value_loss);
        self.progress.policy_losses.push(policy_loss);
        self.progress.clip_range.push(agent.clip_range);
        self.progress.approx_kl.push(approx_kl);

        if approx_kl > 1.5 * self.target_kl {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }
}

pub fn train_ppo(tx: Sender<EventBox>) -> anyhow::Result<()> {
    let total_rollouts = 300;
    let ppo_hook = PPOHook::new(
        10,
        total_rollouts,
        0.001,
        0.01,
        GradientClipping::Norm(0.5),
        tx,
    );
    let env_builder = EnvBuilderType::EnvBuilder {
        builder: Arc::new(r2l_gym::GymEnvBuilder::new(ENV_NAME)),
        n_envs: 5,
    };
    let sampler = FinalSampler::build(
        env_builder,
        StepTrajectoryBound::new(2048),
        None,
        Location::Vec,
    );
    let env_description = sampler.env_description();
    let action_size = env_description.action_space.size();
    let observation_size = env_description.observation_space.size();
    let policy_layers = &[observation_size, 64, 64, action_size];
    let value_layers = &[observation_size, 64, 64, 1];
    let distr: DiagGaussianDistribution<BurnBackend> =
        DiagGaussianDistribution::build(policy_layers);
    let value_net = r2l_burn_lm::sequential::Sequential::build(value_layers);
    let model = ParalellActorModel::new(distr, value_net);
    let lm = ParalellActorCriticLM::new(model, AdamWConfig::new().init());
    let agent = BurnPPO::new(BurnPPOCore::new(lm, 0.2, 64, 0.98, 0.8), ppo_hook);
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
