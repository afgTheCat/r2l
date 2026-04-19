use std::{marker::PhantomData, sync::mpsc::Sender};

use crate::hooks::a2c::{A2CStats, DefaultA2CHook, DefaultA2CHookReporter};

#[derive(Debug, Clone)]
pub struct DefaultA2CHookBuilder {
    normalize_advantage: bool,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    gradient_clipping: Option<f32>,
    n_envs: usize,
    tx: Option<Sender<A2CStats>>,
}

impl DefaultA2CHookBuilder {
    pub fn new(n_envs: usize) -> Self {
        Self {
            n_envs,
            normalize_advantage: false,
            entropy_coeff: 0.,
            vf_coeff: None,
            gradient_clipping: None,
            tx: None,
        }
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

    pub fn with_tx(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.tx = tx;
        self
    }

    pub fn build<T>(self) -> DefaultA2CHook<T> {
        DefaultA2CHook {
            normalize_advantage: self.normalize_advantage,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            gradient_clipping: self.gradient_clipping,
            reporter: self
                .tx
                .map(|tx| DefaultA2CHookReporter::new(tx, self.n_envs)),
            _lm: PhantomData,
        }
    }
}
