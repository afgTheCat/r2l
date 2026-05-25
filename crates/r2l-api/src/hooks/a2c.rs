use std::{marker::PhantomData, sync::mpsc::Sender};

use anyhow::Result;
use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    a2c::{A2CBatchData, A2CHook, A2CParams},
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

use crate::utils::{fmt_stat, mean};

/// Per-batch training statistics emitted by the default A2C hook.
///
/// Each value corresponds to a single optimization batch processed during one
/// A2C learning pass.
#[derive(Debug, Clone)]
pub struct A2CBatchStats {
    /// Entropy regularization term computed for the batch.
    pub entropy_loss: f32,
    /// Policy-gradient loss computed for the batch.
    pub policy_loss: f32,
    /// Value-function loss computed for the batch.
    pub value_loss: f32,
}

/// Aggregated statistics emitted by the default A2C hook after a learning pass.
///
/// A report contains all collected [`A2CBatchStats`] for the rollout together
/// with rollout-level summaries such as average reward and learning rate.
#[derive(Default, Debug, Clone)]
pub struct A2CStats {
    /// Rollout index to which the stats belong to
    pub rollout_idx: usize,
    /// Batch-level statistics collected during the most recent learning pass.
    pub batch_stats: Vec<A2CBatchStats>,
    /// Current action-distribution standard deviation when available.
    pub std: Option<f32>,
    /// Average completed-episode reward observed across the active env set.
    pub average_reward: f32,
    /// Current policy optimizer learning rate.
    pub learning_rate: f64,
}

impl A2CStats {
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

    /// Appends one batch report to this rollout report.
    pub fn collect_batch_data(&mut self, batch_stats: A2CBatchStats) {
        self.batch_stats.push(batch_stats);
    }
}

impl std::fmt::Display for A2CStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows = [
            ("Average reward", fmt_stat(self.average_reward)),
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

        writeln!(f, "A2C stats (rollout {})", self.rollout_idx)?;
        writeln!(f, "{:-<1$}", "", key_width + 15)?;

        for (key, value) in rows {
            writeln!(f, "{key:<key_width$} | {value}")?;
        }

        Ok(())
    }
}

pub(crate) struct DefaultA2CHookReporter {
    pub(crate) rollout_idx: usize,
    pub(crate) report: A2CStats,
    pub(crate) tx: Option<Sender<A2CStats>>,
    pub(crate) log_progress: bool,
    pub(crate) unfinished_episode_rewards: Vec<f32>,
    pub(crate) latest_average_reward: f32,
}

impl DefaultA2CHookReporter {
    pub fn new(tx: Option<Sender<A2CStats>>, log_progress: bool, n_envs: usize) -> Option<Self> {
        if tx.is_some() || log_progress {
            Some(Self {
                rollout_idx: 0,
                report: A2CStats::default(),
                tx,
                log_progress,
                unfinished_episode_rewards: vec![0.; n_envs],
                latest_average_reward: 0.,
            })
        } else {
            None
        }
    }

    pub(crate) fn send_report(&mut self) {
        self.rollout_idx += 1;
        let progress = std::mem::replace(
            &mut self.report,
            A2CStats {
                rollout_idx: self.rollout_idx,
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

impl DefaultA2CHookReporter {
    fn update_average_reward<T: r2l_core::tensor::R2lTensor, B: TrajectoryBatchT<T>>(
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

/// Default training hook used by [`A2CAgentBuilder`](crate::A2CAgentBuilder).
///
/// This hook applies the crate's standard A2C training behavior:
/// advantage normalization when enabled, optional value-loss weighting,
/// optional entropy regularization, optional gradient clipping, and optional
/// rollout reporting through [`A2CStats`].
///
/// The generic parameter tracks the concrete learning-module backend and is not
/// usually named directly by callers.
pub struct DefaultA2CHook<T = ()> {
    pub(crate) normalize_advantage: bool,
    pub(crate) entropy_coeff: f32,
    pub(crate) vf_coeff: Option<f32>,
    pub(crate) gradient_clipping: Option<f32>,
    pub(crate) reporter: Option<DefaultA2CHookReporter>,
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> A2CHook<BurnPolicyValueModule<B, D>>
    for DefaultA2CHook<BurnPolicyValueModule<B, D>>
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
            reporter.update_average_reward(buffers);
            reporter.report.std = module.policy().std().ok();
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report();
        }
        Ok(HookResult::Continue)
    }
}

impl A2CHook<CandlePolicyValueModule> for DefaultA2CHook<CandlePolicyValueModule> {
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
            reporter.update_average_reward(buffers);
            reporter.report.std = module.policy().std().ok();
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report();
        }
        Ok(HookResult::Continue)
    }
}
