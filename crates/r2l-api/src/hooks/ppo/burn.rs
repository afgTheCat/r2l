use crate::hooks::ppo::PPOHook;
use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use r2l_agents::{HookResult, burn_agents::ppo::BurnPPOHooksTrait};
use r2l_burn_lm::learning_module::BurnPolicy;
use r2l_core::sampler::buffer::TrajectoryContainer;

impl<B: AutodiffBackend, D: BurnPolicy<B>> BurnPPOHooksTrait<B, D> for PPOHook {
    fn before_learning_hook<
        T: TrajectoryContainer<Tensor = burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    >(
        &mut self,
        agent: &mut r2l_agents::burn_agents::ppo::BurnPPOCore<B, D>,
        _rollout_buffers: &[T],
        advantages: &mut r2l_core::utils::rollout_buffer::Advantages,
        _returns: &mut r2l_core::utils::rollout_buffer::Returns,
    ) -> anyhow::Result<r2l_agents::HookResult> {
        self.current_epoch = 0;
        if self.normalize_advantage {
            advantages.normalize();
        }
        if let Some(max_grad_norm) = self.gradient_clipping {
            agent
                .lm
                .set_grad_clipping(GradientClipping::Norm(max_grad_norm));
        }
        Ok(HookResult::Continue)
    }

    fn rollout_hook<
        T: TrajectoryContainer<Tensor = burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    >(
        &mut self,
        _agent: &mut r2l_agents::burn_agents::ppo::BurnPPOCore<B, D>,
        _rollout_buffers: &[T],
    ) -> candle_core::Result<HookResult> {
        self.current_epoch += 1;
        let should_stop = self.current_epoch == self.total_epochs;
        if should_stop {
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        agent: &mut r2l_agents::burn_agents::ppo::BurnPPOCore<B, D>,
        losses: &mut r2l_burn_lm::learning_module::PolicyValuesLosses<B>,
        data: &r2l_agents::burn_agents::ppo::PPOBatchData<B>,
    ) -> candle_core::Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = agent.lm.model.distr.entropy().unwrap();
        let entropy_loss = entropy.neg() * self.entropy_coeff;
        losses.apply_entropy(entropy_loss);
        if let Some(target_kl) = self.target_kl {
            let ratio: Vec<f32> = data.ratio.to_data().to_vec().unwrap();
            let log_ratio: Vec<f32> = data.logp_diff.to_data().to_vec().unwrap();
            let approx_kl = ratio
                .iter()
                .zip(log_ratio.iter())
                .map(|(ratio, log_ratio)| (ratio - 1.) - log_ratio)
                .sum::<f32>()
                / ratio.len() as f32;
            if approx_kl > 1.5 * target_kl {
                Ok(HookResult::Break)
            } else {
                Ok(HookResult::Continue)
            }
        } else {
            Ok(HookResult::Continue)
        }
    }
}
