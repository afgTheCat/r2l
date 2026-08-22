use std::{marker::PhantomData, sync::mpsc::Sender};

use burn::{grad_clipping::GradientClipping, tensor::backend::AutodiffBackend};
use candle_core::Tensor;
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    a2c::{A2CBatchData, A2CHook, A2CParams},
};
use r2l_burn::learning_module::{
    BurnPolicy, PolicyValueLearner as BurnPolicyValueLearner,
    PolicyValueLosses as BurnPolicyValueLosses,
};
use r2l_candle::learning_module::{
    PolicyValueLearner as CandlePolicyValueLearner, PolicyValueLosses as CandlePolicyValueLosses,
};
use r2l_core::{
    HookResult,
    buffers::TrajectoryBatch,
    error::{Error, ResourceInterrupted, Result},
    models::Policy,
    on_policy::learning_module::OnPolicyLearner,
    tensor::R2lTensor,
};

use crate::utils::{fmt_stat, mean};

/// Training statistics for a single A2C optimization minibatch.
///
/// These statistics are collected during one A2C learning pass and reported by
/// the default A2C hook.
#[derive(Debug, Clone)]
pub struct A2CMinibatchStats {
    /// Entropy regularization term computed for the batch.
    pub entropy_loss: f32,
    /// Policy-gradient loss computed for the batch.
    pub policy_loss: f32,
    /// Value-function loss computed for the batch.
    pub value_loss: f32,
}

/// Training statistics for a single A2C rollout and its learning pass.
///
/// These statistics include the collected [`A2CMinibatchStats`] together with
/// rollout-level summaries such as average reward and learning rate.
#[derive(Default, Debug, Clone)]
pub struct A2CRolloutStats {
    /// Planned number of rollouts, when it can be determined before training.
    pub total_rollouts: Option<usize>,
    /// Rollout index to which the stats belong to
    pub rollout_idx: usize,
    /// Minibatch statistics collected during the most recent learning pass.
    pub minibatch_stats: Vec<A2CMinibatchStats>,
    /// Current action-distribution standard deviation when available.
    pub std: Option<f32>,
    /// Average completed-episode reward observed across the active env set.
    pub average_reward: f32,
    /// Current policy optimizer learning rate.
    pub learning_rate: f64,
}

impl A2CRolloutStats {
    /// Returns the mean entropy loss across collected minibatches.
    #[must_use]
    pub fn entropy_loss(&self) -> f32 {
        mean(
            &self
                .minibatch_stats
                .iter()
                .map(|s| s.entropy_loss)
                .collect::<Vec<_>>(),
        )
    }

    /// Returns the mean value loss across collected minibatches.
    #[must_use]
    pub fn value_loss(&self) -> f32 {
        mean(
            &self
                .minibatch_stats
                .iter()
                .map(|s| s.value_loss)
                .collect::<Vec<_>>(),
        )
    }

    /// Returns the mean policy loss across collected minibatches.
    #[must_use]
    pub fn policy_loss(&self) -> f32 {
        mean(
            &self
                .minibatch_stats
                .iter()
                .map(|s| s.policy_loss)
                .collect::<Vec<_>>(),
        )
    }

    fn collect_minibatch(&mut self, minibatch_stats: A2CMinibatchStats) {
        self.minibatch_stats.push(minibatch_stats);
    }
}

impl std::fmt::Display for A2CRolloutStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows = [
            ("Average reward", fmt_stat(self.average_reward)),
            ("Policy gradient loss", fmt_stat(self.policy_loss())),
            ("Entropy loss", fmt_stat(self.entropy_loss())),
            ("Value loss", fmt_stat(self.value_loss())),
            ("Learning rate", fmt_stat(self.learning_rate as f32)),
            (
                "Standard deviation",
                self.std.map_or("n/a".into(), |std| std.to_string()),
            ),
        ];

        let key_width = rows.iter().map(|(key, _)| key.len()).max().unwrap_or(0);

        match self.total_rollouts {
            Some(total_rollouts) => {
                writeln!(
                    f,
                    "A2C stats (rollout {}/{total_rollouts})",
                    self.rollout_idx
                )?;
            }
            None => writeln!(f, "A2C stats (rollout {}/?)", self.rollout_idx)?,
        }
        writeln!(f, "{:-<1$}", "", key_width + 15)?;

        for (key, value) in rows {
            writeln!(f, "{key:<key_width$} | {value}")?;
        }

        Ok(())
    }
}

pub(crate) struct A2CRolloutReporter {
    pub(crate) rollout_idx: usize,
    pub(crate) report: A2CRolloutStats,
    pub(crate) tx: Option<Sender<A2CRolloutStats>>,
    pub(crate) log_progress: bool,
    pub(crate) unfinished_episode_rewards: Vec<f32>,
    pub(crate) latest_average_reward: f32,
}

impl A2CRolloutReporter {
    pub(crate) fn new(
        tx: Option<Sender<A2CRolloutStats>>,
        log_progress: bool,
        n_envs: usize,
        total_rollouts: Option<usize>,
    ) -> Option<Self> {
        if tx.is_some() || log_progress {
            Some(Self {
                rollout_idx: 0,
                report: A2CRolloutStats {
                    total_rollouts,
                    ..Default::default()
                },
                tx,
                log_progress,
                unfinished_episode_rewards: vec![0.; n_envs],
                latest_average_reward: 0.,
            })
        } else {
            None
        }
    }

    pub(crate) fn send_report(&mut self, total_rollouts: Option<usize>) -> Result<()> {
        self.rollout_idx += 1;
        self.report.rollout_idx = self.rollout_idx;
        self.report.total_rollouts = total_rollouts;
        let progress = std::mem::take(&mut self.report);
        if self.log_progress {
            println!("{progress}");
        }
        if let Some(tx) = &self.tx {
            tx.send(progress).map_err(|error| {
                Error::ResourceInterrupted(ResourceInterrupted {
                    resource: "A2C rollout reporter".into(),
                    details: error.to_string(),
                })
            })?;
        }
        self.report.average_reward = self.latest_average_reward;
        Ok(())
    }
}

impl A2CRolloutReporter {
    fn update_average_reward<T: r2l_core::tensor::R2lTensor, B: TrajectoryBatch<T>>(
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

/// Learning behavior for A2C optimization.
///
/// This hook applies the crate's standard A2C training behavior:
/// advantage normalization when enabled, optional value-loss weighting,
/// optional entropy regularization, optional gradient clipping, and optional
/// rollout reporting through [`A2CRolloutStats`].
///
/// The generic parameter tracks the concrete learner backend and is not
/// usually named directly by callers.
pub struct A2CLearningHook<T = ()> {
    pub(crate) normalize_advantage: bool,
    pub(crate) entropy_coeff: f32,
    pub(crate) vf_coeff: Option<f32>,
    pub(crate) gradient_clipping: Option<f32>,
    pub(crate) reporter: Option<A2CRolloutReporter>,
    pub(crate) total_rollouts: Option<usize>,
    pub(crate) _lm: PhantomData<T>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> A2CHook<BurnPolicyValueLearner<B, D>>
    for A2CLearningHook<BurnPolicyValueLearner<B, D>>
{
    fn before_learning_hook<
        C: TrajectoryBatch<<BurnPolicyValueLearner<B, D> as OnPolicyLearner>::InferenceTensor>,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueLearner<B, D>,
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
        module: &mut BurnPolicyValueLearner<B, D>,
        losses: &mut BurnPolicyValueLosses<B>,
        data: &A2CBatchData<burn::Tensor<B, 1>>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let entropy_loss = entropy.neg() * self.entropy_coeff;
        if let Some(A2CRolloutReporter { report, .. }) = &mut self.reporter {
            report.collect_minibatch(A2CMinibatchStats {
                policy_loss: losses.policy_loss.to_vec()?[0],
                entropy_loss: entropy_loss.to_vec()?[0],
                value_loss: losses.value_loss.to_vec()?[0],
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(entropy_loss);
        }
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<
        C: TrajectoryBatch<<BurnPolicyValueLearner<B, D> as OnPolicyLearner>::InferenceTensor>,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut BurnPolicyValueLearner<B, D>,
        buffers: &[C],
    ) -> Result<HookResult> {
        if let Some(reporter) = &mut self.reporter {
            reporter.update_average_reward(buffers);
            reporter.report.std = module.policy().std()?;
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report(self.total_rollouts)?;
        }
        Ok(HookResult::Continue)
    }
}

impl A2CHook<CandlePolicyValueLearner> for A2CLearningHook<CandlePolicyValueLearner> {
    fn before_learning_hook<
        B: TrajectoryBatch<<CandlePolicyValueLearner as OnPolicyLearner>::InferenceTensor>,
    >(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueLearner,
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
        module: &mut CandlePolicyValueLearner,
        losses: &mut CandlePolicyValueLosses,
        data: &A2CBatchData<candle_core::Tensor>,
    ) -> Result<HookResult> {
        losses.set_vf_coeff(self.vf_coeff);
        let entropy = module.policy().entropy(&data.observations)?;
        let device = entropy.device();
        let entropy_loss = (Tensor::full(self.entropy_coeff, (), device)? * entropy.neg()?)?;
        if let Some(A2CRolloutReporter { report, .. }) = &mut self.reporter {
            report.collect_minibatch(A2CMinibatchStats {
                policy_loss: losses.policy_loss.to_scalar()?,
                entropy_loss: entropy_loss.to_scalar()?,
                value_loss: losses.value_loss.to_scalar()?,
            });
        }
        if self.entropy_coeff != 0. {
            losses.add_entropy_loss(&entropy_loss)?;
        }
        Ok(HookResult::Continue)
    }

    fn after_learning_hook<B: TrajectoryBatch<candle_core::Tensor>>(
        &mut self,
        _params: &mut A2CParams,
        module: &mut CandlePolicyValueLearner,
        buffers: &[B],
    ) -> Result<HookResult> {
        if let Some(reporter) = &mut self.reporter {
            reporter.update_average_reward(buffers);
            reporter.report.std = module.policy().std()?;
            reporter.report.learning_rate = module.policy_learning_rate();
            reporter.send_report(self.total_rollouts)?;
        }
        Ok(HookResult::Continue)
    }
}
