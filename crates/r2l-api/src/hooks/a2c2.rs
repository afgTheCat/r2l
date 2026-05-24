use anyhow::Result;
use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    a2c2::{A2CBatchData, A2CHook, A2CParams},
};
use r2l_burn::learning_module::{
    BurnPolicy, PolicyValueLosses as BurnPolicyValueLosses,
    PolicyValueModule as BurnPolicyValueModule,
};
use r2l_candle::learning_module::{
    PolicyValueLosses as CandlePolicyValueLosses, PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{
    HookResult, buffers::gen_buffer::TrajectoryBatchT, models::Policy,
    on_policy::learning_module::OnPolicyLearningModule,
};

use crate::hooks::a2c::{A2CBatchStats, DefaultA2CHook, DefaultA2CHookReporter};

/// Default training hook used by the A2C2 builder path.
///
/// This hook applies the standard A2C2 training behavior: optional advantage
/// normalization, optional value-loss weighting, optional entropy
/// regularization, optional gradient clipping, and optional rollout reporting
/// through [`A2CBatchStats`](crate::A2CBatchStats) and
/// [`crate::A2CStats`].
pub type DefaultA2CHook2<T = ()> = DefaultA2CHook<T>;

impl DefaultA2CHookReporter {
    fn update_average_reward2<T: r2l_core::tensor::R2lTensor, B: TrajectoryBatchT<T>>(
        &mut self,
        batches: &[B],
    ) {
        let mut completed_episode_rewards = vec![];
        for (running_reward, batch) in self
            .unfinished_episode_rewards
            .iter_mut()
            .zip(batches.iter())
        {
            for (reward, done) in batch.rewards().iter().copied().zip(
                batch
                    .terminated()
                    .iter()
                    .zip(batch.truncated().iter())
                    .map(|(terminated, truncated)| *terminated || *truncated),
            ) {
                *running_reward += reward;
                if done {
                    completed_episode_rewards.push(*running_reward);
                    *running_reward = 0.;
                }
            }
        }

        if !completed_episode_rewards.is_empty() {
            self.latest_average_reward = completed_episode_rewards.iter().sum::<f32>()
                / completed_episode_rewards.len() as f32;
        }
        self.report.average_reward = self.latest_average_reward;
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> A2CHook<BurnPolicyValueModule<B, D>>
    for DefaultA2CHook2<BurnPolicyValueModule<B, D>>
{
    fn before_learning_hook<
        C: TrajectoryBatchT<<BurnPolicyValueModule<B, D> as OnPolicyLearningModule>::InferenceTensor>,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueModule<B, D>,
        _buffers: &[C],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        if self.normalize_advantage {
            advantages.normalize();
        }
        if let Some(max_grad_norm) = self.gradient_clipping {
            module.set_grad_clipping(GradientClipping::Norm(max_grad_norm));
        }
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueModule<B, D>,
        losses: &mut BurnPolicyValueLosses<B>,
        data: &A2CBatchData<burn::Tensor<B, 1>>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let entropy_loss = entropy.neg() * self.entropy_coeff;
        if let Some(DefaultA2CHookReporter { report, .. }) = &mut self.reporter {
            report.collect_batch_data(A2CBatchStats {
                policy_loss: losses.policy_loss.to_data().to_vec::<f32>().unwrap()[0],
                entropy_loss: entropy_loss.to_data().to_vec::<f32>().unwrap()[0],
                value_loss: losses.value_loss.to_data().to_vec::<f32>().unwrap()[0],
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss);
        }
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<
        C: TrajectoryBatchT<<BurnPolicyValueModule<B, D> as OnPolicyLearningModule>::InferenceTensor>,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueModule<B, D>,
        buffers: &[C],
    ) -> Result<HookResult> {
        if let Some(reporter) = &mut self.reporter {
            reporter.update_average_reward2(buffers);
            reporter.report.std = module.policy().std().ok();
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report();
        }
        Ok(HookResult::Continue)
    }
}

impl A2CHook<CandlePolicyValueModule> for DefaultA2CHook2<CandlePolicyValueModule> {
    fn before_learning_hook<
        B: TrajectoryBatchT<<CandlePolicyValueModule as OnPolicyLearningModule>::InferenceTensor>,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueModule,
        _buffers: &[B],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        if self.normalize_advantage {
            advantages.normalize();
        }
        module.set_grad_clipping(self.gradient_clipping);
        Ok(HookResult::Continue)
    }

    fn batch_hook(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueModule,
        losses: &mut CandlePolicyValueLosses,
        data: &A2CBatchData<candle_core::Tensor>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
        if let Some(DefaultA2CHookReporter { report, .. }) = &mut self.reporter {
            report.collect_batch_data(A2CBatchStats {
                policy_loss: losses.policy_loss.to_scalar()?,
                entropy_loss: entropy_loss.to_scalar()?,
                value_loss: losses.value_loss.to_scalar()?,
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss)?;
        }
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<B: TrajectoryBatchT<candle_core::Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueModule,
        buffers: &[B],
    ) -> Result<HookResult> {
        if let Some(reporter) = &mut self.reporter {
            reporter.update_average_reward2(buffers);
            reporter.report.std = module.policy().std().ok();
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report();
        }
        Ok(HookResult::Continue)
    }
}
