use crate::{
    agents::{ppo::burn::R2lBurnLearningModule, ppo::candle::R2lCandleLearningModule},
    hooks::ppo::{BatchStats, PPOHook, PPOHookReporter},
};
use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::{
    HookResult,
    ppo::{NewPPOBatchData, NewPPOHooksTrait, NewPPOParams, RolloutLearningModule},
};
use r2l_burn_lm::learning_module::{BurnPolicy, BurnPolicyValuesLosses};
use r2l_candle_lm::learning_module::CandlePolicyValuesLosses;
use r2l_core::{distributions::Policy, sampler::buffer::TrajectoryContainer};

impl<B: AutodiffBackend, D: BurnPolicy<B>> NewPPOHooksTrait<R2lBurnLearningModule<B, D>>
    for PPOHook<R2lBurnLearningModule<B, D>>
{
    fn before_learning_hook<
        T: TrajectoryContainer<Tensor = burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    >(
        &mut self,
        _params: &mut NewPPOParams,
        module: &mut R2lBurnLearningModule<B, D>,
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
        _params: &mut NewPPOParams,
        module: &mut R2lBurnLearningModule<B, D>,
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
            if let Some(PPOHookReporter { report, tx }) = &mut self.reporter {
                let mut total_rewards: f32 = 0.;
                let mut total_episodes: usize = 0;
                for buffer in buffers {
                    total_rewards += buffer.rewards().sum::<f32>();
                    total_episodes += buffer.dones().filter(|done| *done).count();
                }
                report.avarage_reward = total_rewards / total_episodes as f32;
                report.std = module.get_policy().std().unwrap();
                report.learning_rate = 3e-4;
                tx.send(std::mem::take(report)).unwrap();
            }
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        params: &mut NewPPOParams,
        module: &mut R2lBurnLearningModule<B, D>,
        losses: &mut BurnPolicyValuesLosses<B>,
        data: &NewPPOBatchData<burn::Tensor<B, 1>>,
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

        if let Some(PPOHookReporter { report, .. }) = &mut self.reporter {
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

impl NewPPOHooksTrait<R2lCandleLearningModule> for PPOHook<R2lCandleLearningModule> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = candle_core::Tensor>>(
        &mut self,
        _params: &mut NewPPOParams,
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
        _params: &mut NewPPOParams,
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
            if let Some(PPOHookReporter { report, tx }) = &mut self.reporter {
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
        params: &mut NewPPOParams,
        module: &mut R2lCandleLearningModule,
        losses: &mut CandlePolicyValuesLosses,
        data: &NewPPOBatchData<candle_core::Tensor>,
    ) -> anyhow::Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.get_policy().entropy().unwrap();
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
        let approx_kl = data.approx_kl()?;
        if let Some(PPOHookReporter { report, .. }) = &mut self.reporter {
            report.collect_batch_data(BatchStats {
                clip_fraction: data.clip_fraction(params.clip_range)?,
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
