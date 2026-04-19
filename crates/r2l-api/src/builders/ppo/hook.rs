use std::{marker::PhantomData, sync::mpsc::Sender};

use crate::hooks::ppo::{DefaultPPOHook, DefaultPPOHookReporter, PPOStats, TargetKl};

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

    pub fn with_tx(mut self, tx: Option<Sender<PPOStats>>) -> Self {
        self.tx = tx;
        self
    }

    pub fn normalize_advantage(&self) -> bool {
        self.normalize_advantage
    }

    pub fn total_epochs(&self) -> usize {
        self.total_epochs
    }

    pub fn entropy_coeff(&self) -> f32 {
        self.entropy_coeff
    }

    pub fn vf_coeff(&self) -> Option<f32> {
        self.vf_coeff
    }

    pub fn target_kl(&self) -> Option<f32> {
        self.target_kl
    }

    pub fn gradient_clipping(&self) -> Option<f32> {
        self.gradient_clipping
    }

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
