use std::{marker::PhantomData, sync::mpsc::Sender};

use crate::hooks::a2c::{A2CStats, DefaultA2CHook, DefaultA2CHookReporter};

/// Builder for the default A2C training hook.
///
/// This builder controls the hook behavior used by
/// [`A2CAgentBuilder`](crate::A2CAgentBuilder), including advantage
/// normalization, loss coefficients, gradient clipping, and optional reporting.
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
    /// Creates a default A2C hook builder.
    ///
    /// `n_envs` is used when reporting rollout statistics.
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

    /// Enables or disables advantage normalization before learning.
    pub fn with_normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.normalize_advantage = normalize_advantage;
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

    /// Sets the optional gradient clipping threshold used during learning.
    pub fn with_gradient_clipping(mut self, gradient_clipping: Option<f32>) -> Self {
        self.gradient_clipping = gradient_clipping;
        self
    }

    /// Installs a channel used to emit [`A2CStats`](crate::A2CStats).
    pub fn with_tx(mut self, tx: Option<Sender<A2CStats>>) -> Self {
        self.tx = tx;
        self
    }

    /// Builds the default A2C hook.
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
