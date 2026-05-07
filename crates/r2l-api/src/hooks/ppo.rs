use std::{marker::PhantomData, sync::mpsc::Sender};

use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    ppo::{PPOBatchData, PPOHook, PPOParams},
};
use r2l_burn::learning_module::{
    BurnPolicy, PolicyValueLosses as BurnPolicyValueLosses,
    PolicyValueModule as BurnPolicyValueModule,
};
use r2l_candle::learning_module::{
    PolicyValueLosses as CandlePolicyValueLosses, PolicyValueModule as CandlePolicyValueModule,
};
use r2l_core::{
    HookResult, buffers::TrajectoryContainer, models::Policy,
    on_policy::learning_module::OnPolicyLearningModule,
};

use crate::utils::{fmt_stat, mean};

/// Per-batch training statistics emitted by the default PPO hook.
///
/// Each value corresponds to a single optimization batch processed within one
/// PPO epoch.
#[derive(Debug, Clone)]
pub struct PPOBatchStats {
    /// Fraction of samples whose probability ratio exceeded the clip range.
    pub clip_fraction: f32,
    /// Entropy regularization term computed for the batch.
    pub entropy_loss: f32,
    /// Policy loss computed for the batch.
    pub policy_loss: f32,
    /// Approximate KL divergence tracked for early stopping and reporting.
    pub approx_kl: f32,
    /// Value-function loss computed for the batch.
    pub value_loss: f32,
}

/// Aggregated statistics emitted by the default PPO hook after a learning pass.
///
/// A report contains all collected [`PPOBatchStats`] for the rollout together
/// with rollout-level summaries such as average reward and learning rate.
#[derive(Default, Debug, Clone)]
pub struct PPOStats {
    /// Rollout index to which the stats belong to
    pub rollout_idx: usize,
    /// Batch-level statistics collected across PPO epochs for the rollout.
    pub batch_stats: Vec<PPOBatchStats>,
    /// Current action-distribution standard deviation when available.
    pub std: Option<f32>,
    /// Average completed-episode reward observed across the active env set.
    pub average_reward: f32,
    /// Current policy optimizer learning rate.
    pub learning_rate: f64,
    /// Clip range
    pub clip_range: f32,
}

impl PPOStats {
    pub fn entropy_loss(&self) -> f32 {
        mean(
            &self
                .batch_stats
                .iter()
                .map(|s| s.entropy_loss)
                .collect::<Vec<_>>(),
        )
    }

    pub fn value_loss(&self) -> f32 {
        mean(
            &self
                .batch_stats
                .iter()
                .map(|s| s.value_loss)
                .collect::<Vec<_>>(),
        )
    }

    pub fn policy_loss(&self) -> f32 {
        mean(
            &self
                .batch_stats
                .iter()
                .map(|s| s.policy_loss)
                .collect::<Vec<_>>(),
        )
    }

    pub fn clip_fraction(&self) -> f32 {
        mean(
            &self
                .batch_stats
                .iter()
                .map(|s| s.clip_fraction)
                .collect::<Vec<_>>(),
        )
    }
}

impl std::fmt::Display for PPOStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows = [
            ("Average reward", fmt_stat(self.average_reward)),
            ("Clip fraction", fmt_stat(self.clip_fraction())),
            ("Policy gradient loss", fmt_stat(self.policy_loss())),
            ("Entropy loss", fmt_stat(self.entropy_loss())),
            ("Value loss", fmt_stat(self.value_loss())),
            ("Learning rate", fmt_stat(self.learning_rate as f32)),
            (
                "Standard deviation",
                self.std.map(|std| std.to_string()).unwrap_or("n/a".into()),
            ),
        ];

        let key_width = rows.iter().map(|(key, _)| key.len()).max().unwrap_or(0);

        writeln!(f, "PPO stats (rollout {})", self.rollout_idx)?;
        writeln!(f, "{:-<1$}", "", key_width + 15)?;

        for (key, value) in rows {
            writeln!(f, "{key:<key_width$} | {value}")?;
        }

        Ok(())
    }
}

impl PPOStats {
    /// Appends one batch report to this rollout report.
    pub fn collect_batch_data(&mut self, batch_stats: PPOBatchStats) {
        self.batch_stats.push(batch_stats);
    }
}

pub(crate) struct TargetKl {
    pub target: f32,
    pub target_exceeded: bool,
}

impl TargetKl {
    pub fn target_kl_exceeded(&mut self) -> bool {
        std::mem::take(&mut self.target_exceeded)
    }
}

pub(crate) struct DefaultPPOHookReporter {
    report: PPOStats,
    tx: Option<Sender<PPOStats>>,
    log_progress: bool,
    unfinished_episode_rewards: Vec<f32>,
    latest_average_reward: f32,
}

impl DefaultPPOHookReporter {
    pub fn new(tx: Option<Sender<PPOStats>>, log_progress: bool, n_envs: usize) -> Option<Self> {
        if tx.is_some() || log_progress {
            Some(Self {
                report: PPOStats::default(),
                log_progress,
                tx: tx,
                unfinished_episode_rewards: vec![0.; n_envs],
                latest_average_reward: 0.,
            })
        } else {
            None
        }
    }
}

impl DefaultPPOHookReporter {
    fn update_average_reward<T: TrajectoryContainer>(&mut self, buffers: &[T]) {
        let mut completed_episode_rewards = vec![];
        for (running_reward, buffer) in self
            .unfinished_episode_rewards
            .iter_mut()
            .zip(buffers.iter())
        {
            for (reward, done) in buffer.rewards().zip(buffer.dones()) {
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

    fn send_report(&mut self, rollout_idx: usize) {
        let progress = std::mem::replace(
            &mut self.report,
            PPOStats {
                rollout_idx,
                ..Default::default()
            },
        );
        if self.log_progress {
            println!("{progress}");
        }
        if let Some(tx) = &self.tx {
            tx.send(progress).unwrap();
        }
        self.report.average_reward = self.latest_average_reward;
    }
}

/// Default training hook used by [`PPOAgentBuilder`](crate::PPOAgentBuilder).
///
/// This hook applies the crate's standard PPO training behavior: advantage
/// normalization when enabled, repeated PPO epochs, optional value-loss
/// weighting, optional entropy regularization, optional gradient clipping,
/// optional target-KL early stopping, and optional rollout reporting through
/// [`PPOStats`].
///
/// The generic parameter tracks the concrete learning-module backend and is not
/// usually named directly by callers.
pub struct DefaultPPOHook<T = ()> {
    pub(crate) normalize_advantage: bool,
    pub(crate) total_epochs: usize,
    pub(crate) entropy_coeff: f32,
    pub(crate) vf_coeff: Option<f32>,
    pub(crate) target_kl: Option<TargetKl>,
    pub(crate) gradient_clipping: Option<f32>,
    pub(crate) current_epoch: usize,
    pub(crate) reporter: Option<DefaultPPOHookReporter>,
    pub(crate) rollout_idx: usize,
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PPOHook<BurnPolicyValueModule<B, D>>
    for DefaultPPOHook<BurnPolicyValueModule<B, D>>
{
    fn before_learning_hook<
        T: TrajectoryContainer<Tensor = burn::Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    >(
        &mut self,
        _params: &mut PPOParams,
        module: &mut BurnPolicyValueModule<B, D>,
        _buffers: &[T],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        self.current_epoch = 0;
        self.rollout_idx += 1;
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
        params: &mut PPOParams,
        module: &mut BurnPolicyValueModule<B, D>,
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
            if let Some(reporter) = &mut self.reporter {
                reporter.update_average_reward(buffers);
                reporter.report.std = module.policy().std().ok();
                reporter.report.learning_rate = module.policy_learning_rate();
                reporter.report.clip_range = params.clip_range;
                reporter.send_report(self.rollout_idx);
            }
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        params: &mut PPOParams,
        module: &mut BurnPolicyValueModule<B, D>,
        losses: &mut BurnPolicyValueLosses<B>,
        data: &PPOBatchData<burn::Tensor<B, 1>>,
    ) -> anyhow::Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations).unwrap();
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
            report.collect_batch_data(PPOBatchStats {
                clip_fraction,
                policy_loss: losses.policy_loss.to_data().to_vec::<f32>().unwrap()[0],
                entropy_loss: entropy_loss.to_data().to_vec::<f32>().unwrap()[0],
                value_loss: losses.value_loss.to_data().to_vec::<f32>().unwrap()[0],
                approx_kl,
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss);
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

impl PPOHook<CandlePolicyValueModule> for DefaultPPOHook<CandlePolicyValueModule> {
    fn before_learning_hook<B: TrajectoryContainer<Tensor = candle_core::Tensor>>(
        &mut self,
        _params: &mut PPOParams,
        module: &mut CandlePolicyValueModule,
        _buffers: &[B],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        self.rollout_idx += 1;
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
        module: &mut CandlePolicyValueModule,
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
            if let Some(reporter) = &mut self.reporter {
                reporter.update_average_reward(buffers);
                reporter.report.std = module.policy().std().ok();
                reporter.report.learning_rate = module.policy_learning_rate();
                reporter.send_report(self.rollout_idx);
            }
            Ok(HookResult::Break)
        } else {
            Ok(HookResult::Continue)
        }
    }

    fn batch_hook(
        &mut self,
        params: &mut PPOParams,
        module: &mut CandlePolicyValueModule,
        losses: &mut CandlePolicyValueLosses,
        data: &PPOBatchData<candle_core::Tensor>,
    ) -> anyhow::Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations).unwrap();
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
            report.collect_batch_data(PPOBatchStats {
                clip_fraction,
                policy_loss: losses.policy_loss.to_scalar()?,
                entropy_loss: entropy_loss.to_scalar()?,
                value_loss: losses.value_loss.to_scalar()?,
                approx_kl,
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss)?;
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
