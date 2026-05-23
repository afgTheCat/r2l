use std::{marker::PhantomData, sync::mpsc::Sender};

use crate::hooks::ppo2::{DefaultPPO2Hook, DefaultPPO2HookReporter, PPO2Stats, TargetKl};

#[derive(Debug, Clone)]
pub struct DefaultPPO2HookBuilder {
    normalize_advantage: bool,
    log_progress: bool,
    total_epochs: usize,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    target_kl: Option<f32>,
    gradient_clipping: Option<f32>,
    n_envs: usize,
    tx: Option<Sender<PPO2Stats>>,
}

impl DefaultPPO2HookBuilder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            normalize_advantage: true,
            log_progress: true,
            total_epochs: 10,
            entropy_coeff: 0.,
            vf_coeff: None,
            target_kl: None,
            gradient_clipping: None,
            n_envs,
            tx: None,
        }
    }

    pub fn with_log_progress(mut self, log_progress: bool) -> Self {
        self.log_progress = log_progress;
        self
    }

    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.normalize_advantage = normalize_advantage;
        self
    }

    pub fn with_total_epochs(mut self, total_epochs: usize) -> Self {
        self.total_epochs = total_epochs;
        self
    }

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.entropy_coeff = entropy_coeff;
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.vf_coeff = vf_coeff;
        self
    }

    pub fn with_target_kl(mut self, target_kl: Option<f32>) -> Self {
        self.target_kl = target_kl;
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.gradient_clipping = gradient_clipping;
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<PPO2Stats>>) -> Self {
        self.tx = tx;
        self
    }

    pub fn build<T>(self) -> DefaultPPO2Hook<T> {
        DefaultPPO2Hook {
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
            rollout_idx: 0,
            reporter: DefaultPPO2HookReporter::new(self.tx, self.log_progress, self.n_envs),
            _lm: PhantomData,
        }
    }
}
