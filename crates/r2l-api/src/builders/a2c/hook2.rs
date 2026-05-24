use std::{marker::PhantomData, sync::mpsc::Sender};

use crate::{
    hooks::a2c::{A2CStats, DefaultA2CHookReporter},
    hooks::a2c2::DefaultA2CHook2,
};

#[derive(Debug, Clone)]
pub struct DefaultA2CHook2Builder {
    normalize_advantage: bool,
    log_progress: bool,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    gradient_clipping: Option<f32>,
    n_envs: usize,
    tx: Option<Sender<A2CStats>>,
}

impl DefaultA2CHook2Builder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            n_envs,
            normalize_advantage: false,
            log_progress: true,
            entropy_coeff: 0.,
            vf_coeff: None,
            gradient_clipping: None,
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

    pub fn with_entropy_coeff(mut self, entropy_coeff: f32) -> Self {
        self.entropy_coeff = entropy_coeff;
        self
    }

    pub fn with_vf_coeff(mut self, vf_coeff: Option<f32>) -> Self {
        self.vf_coeff = vf_coeff;
        self
    }

    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.gradient_clipping = gradient_clipping;
        self
    }

    pub fn with_reporter(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.tx = tx;
        self
    }

    pub fn build<T>(self) -> DefaultA2CHook2<T> {
        DefaultA2CHook2 {
            normalize_advantage: self.normalize_advantage,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            gradient_clipping: self.gradient_clipping,
            reporter: DefaultA2CHookReporter::new(self.tx, self.log_progress, self.n_envs),
            _lm: PhantomData,
        }
    }
}
