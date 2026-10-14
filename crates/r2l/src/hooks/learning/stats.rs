use crate::utils::{fmt_stat, mean};

fn fmt_rollout_stats(
    f: &mut std::fmt::Formatter<'_>,
    algorithm: &str,
    rollout_idx: usize,
    total_rollouts: Option<usize>,
    rows: &[(&str, String)],
) -> std::fmt::Result {
    let key_width = rows.iter().map(|(key, _)| key.len()).max().unwrap_or(0);
    match total_rollouts {
        Some(total_rollouts) => {
            writeln!(
                f,
                "{algorithm} stats (rollout {rollout_idx}/{total_rollouts})"
            )?;
        }
        None => writeln!(f, "{algorithm} stats (rollout {rollout_idx}/?)")?,
    }
    writeln!(f, "{:-<1$}", "", key_width + 15)?;
    for (key, value) in rows {
        writeln!(f, "{key:<key_width$} | {value}")?;
    }
    Ok(())
}

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
        mean(self.minibatch_stats.iter().map(|s| s.entropy_loss))
    }

    /// Returns the mean value loss across collected minibatches.
    #[must_use]
    pub fn value_loss(&self) -> f32 {
        mean(self.minibatch_stats.iter().map(|s| s.value_loss))
    }

    /// Returns the mean policy loss across collected minibatches.
    #[must_use]
    pub fn policy_loss(&self) -> f32 {
        mean(self.minibatch_stats.iter().map(|s| s.policy_loss))
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

        fmt_rollout_stats(f, "A2C", self.rollout_idx, self.total_rollouts, &rows)
    }
}

/// Training statistics for a single PPO optimization minibatch.
///
/// These statistics are collected during one PPO epoch and reported by the
/// default PPO hook.
#[derive(Debug, Clone)]
pub struct PPOMinibatchStats {
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

/// Training statistics for a single PPO rollout and its learning pass.
///
/// These statistics include the [`PPOMinibatchStats`] collected across PPO
/// epochs together with rollout-level summaries such as average reward and
/// learning rate.
#[derive(Default, Debug, Clone)]
pub struct PPORolloutStats {
    /// Planned number of rollouts, when it can be determined before training.
    pub total_rollouts: Option<usize>,
    /// Rollout index to which the stats belong.
    pub rollout_idx: usize,
    /// Minibatch statistics collected across PPO epochs for the rollout.
    pub minibatch_stats: Vec<PPOMinibatchStats>,
    /// Current action-distribution standard deviation when available.
    pub std: Option<f32>,
    /// Average completed-episode reward observed across the active env set.
    pub average_reward: f32,
    /// Current policy optimizer learning rate.
    pub learning_rate: f64,
    /// PPO clip range used during the rollout.
    pub clip_range: f32,
}

impl PPORolloutStats {
    /// Returns the mean entropy loss across all collected batch stats.
    #[must_use]
    pub fn entropy_loss(&self) -> f32 {
        mean(self.minibatch_stats.iter().map(|s| s.entropy_loss))
    }

    /// Returns the mean value loss across all collected batch stats.
    #[must_use]
    pub fn value_loss(&self) -> f32 {
        mean(self.minibatch_stats.iter().map(|s| s.value_loss))
    }

    /// Returns the mean policy loss across all collected batch stats.
    #[must_use]
    pub fn policy_loss(&self) -> f32 {
        mean(self.minibatch_stats.iter().map(|s| s.policy_loss))
    }

    /// Returns the mean clip fraction across all collected batch stats.
    #[must_use]
    pub fn clip_fraction(&self) -> f32 {
        mean(self.minibatch_stats.iter().map(|s| s.clip_fraction))
    }
}

impl std::fmt::Display for PPORolloutStats {
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
                self.std.map_or("n/a".into(), |std| std.to_string()),
            ),
        ];

        fmt_rollout_stats(f, "PPO", self.rollout_idx, self.total_rollouts, &rows)
    }
}
