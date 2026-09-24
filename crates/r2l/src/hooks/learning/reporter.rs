use std::{fmt::Display, sync::mpsc::Sender};

use itertools::izip;
use r2l_core::{
    buffers::TrajectoryBatch,
    error::{Error, ResourceInterrupted, Result},
    tensor::R2lTensor,
    utils::slice_mean,
};

use super::stats::{A2CMinibatchStats, A2CRolloutStats, PPOMinibatchStats, PPORolloutStats};

/// Shared reward tracking and delivery, retaining the existing statistics payloads.
pub(crate) enum RolloutReporter<R> {
    Disabled,
    Enabled(EnabledRolloutReporter<R>),
}

pub(crate) struct EnabledRolloutReporter<R> {
    report: R,
    tx: Option<Sender<R>>,
    log_progress: bool,
    unfinished_episode_rewards: Vec<f32>,
    latest_average_reward: f32,
}

impl<R> RolloutReporter<R> {
    /// Whether reporting statistics should be computed.
    pub(super) fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled(_))
    }
}

impl<R: Default + Display> RolloutReporter<R> {
    /// Creates a reporter when logging or channel delivery is enabled.
    ///
    /// # Arguments
    ///
    /// * `tx` - Optional channel receiving each rollout's statistics.
    /// * `log_progress` - Whether to print rollout statistics.
    /// * `n_envs` - Number of environment reward streams to track.
    pub(crate) fn new(tx: Option<Sender<R>>, log_progress: bool, n_envs: usize) -> Self {
        if tx.is_none() && !log_progress {
            return Self::Disabled;
        }
        Self::Enabled(EnabledRolloutReporter {
            report: R::default(),
            tx,
            log_progress,
            unfinished_episode_rewards: vec![0.; n_envs],
            latest_average_reward: 0.,
        })
    }
}

impl<R: Default + Display> EnabledRolloutReporter<R> {
    fn update_average_reward<T: R2lTensor, B: TrajectoryBatch<T>>(&mut self, batches: &[B]) {
        let mut completed_episode_rewards = vec![];
        for (running_reward, batch) in self.unfinished_episode_rewards.iter_mut().zip(batches) {
            for (reward, done) in izip!(batch.iter_rewards(), batch.iter_dones()) {
                *running_reward += reward;
                if done {
                    completed_episode_rewards.push(*running_reward);
                    *running_reward = 0.;
                }
            }
        }
        if !completed_episode_rewards.is_empty() {
            self.latest_average_reward = slice_mean(&completed_episode_rewards);
        }
    }

    fn send_report(&mut self) -> Result<()> {
        let report = std::mem::take(&mut self.report);
        if self.log_progress {
            println!("{report}");
        }
        if let Some(tx) = &self.tx {
            tx.send(report).map_err(|error| {
                Error::ResourceInterrupted(ResourceInterrupted {
                    resource: "on-policy rollout reporter".into(),
                    details: error.to_string(),
                })
            })?;
        }
        Ok(())
    }
}

impl RolloutReporter<A2CRolloutStats> {
    /// Records available minibatch statistics when enabled.
    ///
    /// # Arguments
    ///
    /// * `stats` - Statistics computed for this minibatch, or `None` if collection was skipped.
    pub(super) fn record_batch(&mut self, stats: Option<A2CMinibatchStats>) {
        if let (Self::Enabled(reporter), Some(stats)) = (self, stats) {
            reporter.report.minibatch_stats.push(stats);
        }
    }

    /// Updates episode rewards and delivers the rollout report when enabled.
    ///
    /// # Arguments
    ///
    /// * `batches` - Collected trajectories, in environment order, processed once per rollout.
    /// * `completed_rollouts` - Number of completed collections used as the report index.
    /// * `total_rollouts` - Planned number of collections, if known.
    /// * `std` - Current action-distribution standard deviation, if available.
    /// * `learning_rate` - Policy learning rate used for this rollout.
    pub(super) fn report<T: R2lTensor, B: TrajectoryBatch<T>>(
        &mut self,
        batches: &[B],
        completed_rollouts: usize,
        total_rollouts: Option<usize>,
        std: Option<f32>,
        learning_rate: f64,
    ) -> Result<()> {
        let Self::Enabled(reporter) = self else {
            return Ok(());
        };
        reporter.update_average_reward(batches);
        reporter.report.rollout_idx = completed_rollouts;
        reporter.report.total_rollouts = total_rollouts;
        reporter.report.average_reward = reporter.latest_average_reward;
        reporter.report.std = std;
        reporter.report.learning_rate = learning_rate;
        reporter.send_report()
    }
}

impl RolloutReporter<PPORolloutStats> {
    /// Records available minibatch statistics when enabled.
    ///
    /// # Arguments
    ///
    /// * `stats` - Loss statistics, or `None` if collection was skipped.
    /// * `clip_fraction` - Fraction of samples outside the PPO clipping range.
    /// * `approx_kl` - Estimated KL divergence for this minibatch.
    pub(super) fn record_batch(
        &mut self,
        stats: Option<A2CMinibatchStats>,
        clip_fraction: f32,
        approx_kl: f32,
    ) {
        if let (Self::Enabled(reporter), Some(stats)) = (self, stats) {
            reporter.report.minibatch_stats.push(PPOMinibatchStats {
                policy_loss: stats.policy_loss,
                entropy_loss: stats.entropy_loss,
                value_loss: stats.value_loss,
                clip_fraction,
                approx_kl,
            });
        }
    }

    /// Updates episode rewards and delivers the rollout report when enabled.
    ///
    /// # Arguments
    ///
    /// * `batches` - Collected trajectories, in environment order, processed once per rollout.
    /// * `completed_rollouts` - Number of completed collections used as the report index.
    /// * `total_rollouts` - Planned number of collections, if known.
    /// * `std` - Current action-distribution standard deviation, if available.
    /// * `learning_rate` - Policy learning rate used for this rollout.
    /// * `clip_range` - PPO clipping range used for this rollout.
    pub(super) fn report<T: R2lTensor, B: TrajectoryBatch<T>>(
        &mut self,
        batches: &[B],
        completed_rollouts: usize,
        total_rollouts: Option<usize>,
        std: Option<f32>,
        learning_rate: f64,
        clip_range: f32,
    ) -> Result<()> {
        let Self::Enabled(reporter) = self else {
            return Ok(());
        };
        reporter.update_average_reward(batches);
        reporter.report.rollout_idx = completed_rollouts;
        reporter.report.total_rollouts = total_rollouts;
        reporter.report.average_reward = reporter.latest_average_reward;
        reporter.report.std = std;
        reporter.report.learning_rate = learning_rate;
        reporter.report.clip_range = clip_range;
        reporter.send_report()
    }
}
