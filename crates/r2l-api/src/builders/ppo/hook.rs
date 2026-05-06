use std::{marker::PhantomData, sync::mpsc::Sender};

use crate::hooks::ppo::{DefaultPPOHook, DefaultPPOHookReporter, PPOStats, TargetKl};

/// Builder for the default PPO training hook.
///
/// This builder controls the hook behavior used by
/// [`PPOAgentBuilder`](crate::PPOAgentBuilder), including PPO epoch count,
/// advantage normalization, target KL handling, gradient clipping, and
/// optional reporting.
#[derive(Debug, Clone)]
pub struct DefaultPPOHookBuilder {
    normalize_advantage: bool,
    total_epochs: usize,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    target_kl: Option<f32>,
    gradient_clipping: Option<f32>,
    n_envs: usize,
    tx: Option<Sender<PPOStats>>,
}

impl DefaultPPOHookBuilder {
    /// Creates a default PPO hook builder.
    ///
    /// `n_envs` is used when reporting rollout statistics.
    pub fn new(n_envs: usize) -> Self {
        Self {
            normalize_advantage: true,
            total_epochs: 10,
            entropy_coeff: 0.,
            vf_coeff: None,
            target_kl: None,
            gradient_clipping: None,
            n_envs,
            tx: None,
        }
    }

    /// Enables or disables advantage normalization before learning.
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.normalize_advantage = normalize_advantage;
        self
    }

    /// Sets the maximum number of PPO epochs per rollout.
    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.total_epochs = total_epochs;
        self
    }

    /// Sets the entropy coefficient added during optimization.
    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.entropy_coeff = entropy_coeff;
        self
    }

    /// Sets the optional value-function loss coefficient.
    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.vf_coeff = vf_coeff;
        self
    }

    /// Sets the optional target KL threshold used for early stopping.
    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.target_kl = target_kl;
        self
    }

    /// Sets the optional gradient clipping threshold used during learning.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.gradient_clipping = gradient_clipping;
        self
    }

    /// Installs a channel used to emit [`PPOStats`](crate::PPOStats).
    pub fn with_tx(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.tx = tx;
        self
    }

    /// Builds the default PPO hook.
    pub fn build<T>(self) -> DefaultPPOHook<T> {
        DefaultPPOHook {
            normalize_advantage: self.normalize_advantage,
            total_epochs: self.total_epochs,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            target_kl: self.target_kl.map(|target| TargetKl {
                target,
                target_exceeded: false,
            }),
            gradient_clipping: self.gradient_clipping,
            current_epoch: 0,
            reporter: self
                .tx
                .map(|tx| DefaultPPOHookReporter::new(tx, self.n_envs)),
            _lm: PhantomData,
        }
    }
}
