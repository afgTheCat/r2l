use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::{
    HookResult,
    on_policy_algorithms::ppo::{PPOBatchData, PPOHook, PPOParams},
};
use r2l_burn::learning_module::{BurnActorCriticLMKind, BurnPolicy, BurnPolicyValuesLosses};
use r2l_candle::learning_module::{CandlePolicyValuesLosses, R2lCandleLearningModule};
use r2l_core::{
    distributions::Policy, policies::OnPolicyLearningModule, sampler::buffer::TrajectoryContainer,
};
use std::{marker::PhantomData, sync::mpsc::Sender};

#[derive(Debug, Clone)]
pub struct BatchStats {
    pub clip_fraction: f32,
    pub entropy_loss: f32,
    pub policy_loss: f32,
    pub approx_kl: f32,
    pub value_loss: f32,
}

#[derive(Default, Debug, Clone)]
pub struct PPOStats {
    pub batch_stats: Vec<BatchStats>,
    pub std: f32,
    pub avarage_reward: f32,
    pub learning_rate: f64,
}

impl PPOStats {
    pub fn collect_batch_data(&mut self, batch_stats: BatchStats) {
        self.batch_stats.push(batch_stats);
    }
}

pub struct TargetKl {
    pub target: f32,
    pub target_exceeded: bool,
}

impl TargetKl {
    pub fn target_kl_exceeded(&mut self) -> bool {
        std::mem::take(&mut self.target_exceeded)
    }
}

pub struct DefaultPPOHookReporter {
    pub report: PPOStats,
    pub tx: Sender<PPOStats>,
}

pub struct DefaultPPOHook<T = ()> {
    pub normalize_advantage: bool,
    pub total_epochs: usize,
    pub entropy_coeff: f32,
    pub vf_coeff: Option<f32>,
    pub target_kl: Option<TargetKl>,
    pub gradient_clipping: Option<f32>,
    pub current_epoch: usize,
    pub reporter: Option<DefaultPPOHookReporter>,
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PPOHook<BurnActorCriticLMKind<B, D>>
    for DefaultPPOHook<BurnActorCriticLMKind<B, D>>
{
    fn before_learning_hook<
        T: TrajectoryContainer<Tensor = burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    >(
        &mut self,
        _params: &mut PPOParams,
        module: &mut BurnActorCriticLMKind<B, D>,
        _buffers: &[T],
        advantages: &mut r2l_core::utils::rollout_buffer::Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> anyhow::Result<HookResult> {
        self.current_epoch = 0;
        if self.normalize_advantage {
            advantages.normalize();
        }
        if let Some(max_grad_norm) = self.gradient_clipping {
            module.set_grad_clipping(GradientClipping::Norm(max_grad_norm));
        }
        Ok(HookResult::Continue)
    }

    fn rollout_hook<
        T: TrajectoryContainer<Tensor = burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    >(
        &mut self,
        _params: &mut PPOParams,
        module: &mut BurnActorCriticLMKind<B, D>,
        buffers: &[T],
    ) -> anyhow::Result<HookResult> {
        self.current_epoch += 1;
        let target_kl_exceeded = if let Some(target_kl) = &mut self.target_kl {
            target_kl.target_kl_exceeded()
        } else {
            false
        };
        let should_stop = self.current_epoch == self.total_epochs || target_kl_exceeded;
        if should_stop {
            if let Some(DefaultPPOHookReporter { report, tx }) = &mut self.reporter {
                let mut total_rewards: f32 = 0.;
                let mut total_episodes: usize = 0;
                for buffer in buffers {
                    total_rewards += buffer.rewards().sum::<f32>();
                    total_episodes += buffer.dones().filter(|done| *done).count();
                }
                report.avarage_reward = total_rewards / total_episodes as f32;
                report.std = module.get_policy().std().unwrap();
                report.learning_rate = module.policy_learning_rate();
                tx.send(std::mem::take(report)).unwrap();
            }
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnActorCriticLMKind<B, D>,
        losses: &mut BurnPolicyValuesLosses<B>,
        data: &PPOBatchData<burn::Tensor<B, 1>>,
    ) -> anyhow::Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.get_policy().entropy().unwrap();
        let entropy_loss = entropy.neg() * self.entropy_coeff;
        let approx_kl = {
            let ratio: Vec<f32> = data.ratio.to_data().to_vec().unwrap();
            let log_ratio: Vec<f32> = data.logp_diff.to_data().to_vec().unwrap();
            ratio
                .iter()
                .zip(log_ratio.iter())
                .map(|(ratio, log_ratio)| (ratio - 1.) - log_ratio)
                .sum::<f32>()
                / ratio.len() as f32
        };

        if let Some(DefaultPPOHookReporter { report, .. }) = &mut self.reporter {
            let ratio: Vec<f32> = data.ratio.to_data().to_vec().unwrap();
            let clip_fraction = ratio
                .iter()
                .filter(|value| (**value - 1.).abs() > params.clip_range)
                .count() as f32
                / ratio.len() as f32;
            report.collect_batch_data(BatchStats {
                clip_fraction,
                policy_loss: losses.policy_loss.to_data().to_vec::<f32>().unwrap()[0],
                entropy_loss: entropy_loss.to_data().to_vec::<f32>().unwrap()[0],
                value_loss: losses.value_loss.to_data().to_vec::<f32>().unwrap()[0],
                approx_kl,
            });
        }
        if self.entropy_coeff != 0. {
            losses.apply_entropy(entropy_loss);
        }
        if let Some(target_kl) = &mut self.target_kl {
            if approx_kl > 1.5 * target_kl.target {
                target_kl.target_exceeded = true;
                Ok(HookResult::Break)
            } else {
                Ok(HookResult::Continue)
            }
        } else {
            Ok(HookResult::Continue)
        }
    }
}

impl PPOHook<R2lCandleLearningModule> for DefaultPPOHook<R2lCandleLearningModule> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = candle_core::Tensor>>(
        &mut self,
        _params: &mut PPOParams,
        module: &mut R2lCandleLearningModule,
        _buffers: &[B],
        advantages: &mut r2l_core::utils::rollout_buffer::Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> anyhow::Result<HookResult> {
        self.current_epoch = 0;
        if self.normalize_advantage {
            advantages.normalize();
        }
        module.set_grad_clipping(self.gradient_clipping);
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryContainer<Tensor = candle_core::Tensor>>(
        &mut self,
        _params: &mut PPOParams,
        module: &mut R2lCandleLearningModule,
        buffers: &[B],
    ) -> anyhow::Result<HookResult> {
        self.current_epoch += 1;
        let target_kl_exceeded = if let Some(target_kl) = &mut self.target_kl {
            target_kl.target_kl_exceeded()
        } else {
            false
        };
        let should_stop = self.current_epoch == self.total_epochs || target_kl_exceeded;
        if should_stop {
            if let Some(DefaultPPOHookReporter { report, tx }) = &mut self.reporter {
                let mut total_rewards: f32 = 0.;
                let mut total_episodes: usize = 0;
                for buffer in buffers {
                    total_rewards += buffer.rewards().sum::<f32>();
                    total_episodes += buffer.dones().filter(|x| *x).count();
                }
                report.avarage_reward = total_rewards / total_episodes as f32;
                report.std = module.get_policy().std().unwrap();
                report.learning_rate = module.policy_learning_rate();
                tx.send(std::mem::take(report)).unwrap();
            }
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        params: &mut PPOParams,
        module: &mut R2lCandleLearningModule,
        losses: &mut CandlePolicyValuesLosses,
        data: &PPOBatchData<candle_core::Tensor>,
    ) -> anyhow::Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.get_policy().entropy().unwrap();
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
        let ratio = data.ratio.detach();
        let log_ratio = data.logp_diff.detach();
        let approx_kl = ratio
            .sub(&candle_core::Tensor::ones_like(&ratio)?)?
            .sub(&log_ratio)?
            .mean_all()?
            .to_scalar::<f32>()?;
        if let Some(DefaultPPOHookReporter { report, .. }) = &mut self.reporter {
            let clip_fraction = (&data.ratio - 1.)?
                .abs()?
                .gt(params.clip_range)?
                .to_dtype(candle_core::DType::F32)?
                .mean_all()?
                .to_scalar::<f32>()?;
            report.collect_batch_data(BatchStats {
                clip_fraction,
                policy_loss: losses.policy_loss.to_scalar()?,
                entropy_loss: entropy_loss.to_scalar()?,
                value_loss: losses.value_loss.to_scalar()?,
                approx_kl,
            });
        }
        if self.entropy_coeff != 0. {
            losses.apply_entropy(entropy_loss)?;
        }
        if let Some(target_kl) = &mut self.target_kl {
            if approx_kl > 1.5 * target_kl.target {
                target_kl.target_exceeded = true;
                Ok(HookResult::Break)
            } else {
                Ok(HookResult::Continue)
            }
        } else {
            Ok(HookResult::Continue)
        }
    }
}
