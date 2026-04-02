use crate::hooks::ppo::{BatchStats, PPOHook};
use candle_core::Tensor;
use r2l_agents::{
    HookResult,
    ppo::{PPOBatchData, PPOHooksTrait, PPOParams},
};
use r2l_candle_lm::{R2lCandleLearningModule, learning_module::PolicyValuesLosses};
use r2l_core::{
    distributions::Policy, policies::ModuleWithValueFunction, sampler::buffer::TrajectoryContainer,
};

impl PPOHooksTrait<R2lCandleLearningModule> for PPOHook<R2lCandleLearningModule> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = Tensor>>(
        &mut self,
        agent: &mut PPOParams<R2lCandleLearningModule>,
        _buffers: &[B],
        advantages: &mut r2l_core::utils::rollout_buffer::Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> anyhow::Result<HookResult> {
        self.current_epoch = 0;
        if self.normalize_advantage {
            advantages.normalize();
        }
        agent.module.set_grad_clipping(self.gradient_clipping);
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryContainer<Tensor = <R2lCandleLearningModule as r2l_core::policies::ModuleWithValueFunction>::InferenceTensor>>(
            &mut self,
            buffers: &[B],
            agent: &mut PPOParams<R2lCandleLearningModule>,
    ) -> anyhow::Result<HookResult>{
        self.current_epoch += 1;
        let target_kl_exceeded = if let Some(target_kl) = &mut self.target_kl {
            target_kl.target_kl_exceeded()
        } else {
            false
        };
        let should_stop = self.current_epoch == self.total_epochs || target_kl_exceeded;
        if should_stop {
            let mut total_rewards: f32 = 0.;
            let mut total_episodes: usize = 0;
            for buffer in buffers {
                total_rewards += buffer.rewards().sum::<f32>();
                total_episodes += buffer.dones().filter(|x| *x).count();
            }
            self.report.avarage_reward = total_rewards / total_episodes as f32;
            self.report.std = agent.module.get_policy().std().unwrap();
            self.report.learning_rate = agent.module.policy_learning_rate();
            self.tx.send(std::mem::take(&mut self.report)).unwrap();
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        agent: &mut PPOParams<R2lCandleLearningModule>,
        losses: &mut PolicyValuesLosses,
        data: &PPOBatchData<Tensor>,
    ) -> anyhow::Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = agent.module.get_policy().entropy().unwrap();
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
        let approx_kl = data.approx_kl()?;
        self.report.collect_batch_data(BatchStats {
            clip_fraction: data.clip_fraction(agent.clip_range)?,
            policy_loss: losses.policy_loss.to_scalar()?,
            entropy_loss: entropy_loss.to_scalar()?,
            value_loss: losses.value_loss.to_scalar()?,
            approx_kl,
        });
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

// impl<B: burn::tensor::backend::AutodiffBackend, T: BurnModuleWithValueFunction<B>> PPOHooksTrait<T>
//     for PPOHook
// {
// }
