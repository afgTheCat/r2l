pub mod burn;
pub mod candle;

use std::sync::mpsc::Sender;

// All the hooks that the user might want to adjust
#[derive(Debug, Clone)]
pub struct PPOHookBuilder {
    normalize_advantage: bool,
    total_epochs: usize,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    target_kl: Option<f32>,
    gradient_clipping: Option<f32>,
}

impl Default for PPOHookBuilder {
    fn default() -> Self {
        Self {
            normalize_advantage: true,
            total_epochs: 10,
            entropy_coeff: 0.,
            vf_coeff: None,
            target_kl: None,
            gradient_clipping: None,
        }
    }
}

impl PPOHookBuilder {
    pub fn new() -> Self {
        Self::default()
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

    pub fn build(self, tx: Sender<PPOStats>) -> PPOHook {
        PPOHook {
            normalize_advantage: self.normalize_advantage,
            total_epochs: self.total_epochs,
            entropy_coeff: self.entropy_coeff,
            vf_coeff: self.vf_coeff,
            target_kl: self.target_kl,
            gradient_clipping: self.gradient_clipping,
            current_epoch: 0,
            report: PPOStats::default(),
            tx,
        }
    }
}

pub struct BatchStats {
    pub clip_fracion: f32,
    pub entropy_loss: f32,
    pub policy_loss: f32,
    pub approx_kl: f32,
    pub value_loss: f32,
}

#[derive(Default)]
pub struct PPOStats {
    pub batch_stats: Vec<BatchStats>,
    pub std: f32,
    pub avarage_reward: f32,
    pub learning_rate: f64,
}

impl PPOStats {
    fn collect_batch_data(&mut self, batch_stats: BatchStats) {
        self.batch_stats.push(batch_stats);
    }
}

pub struct PPOHook {
    normalize_advantage: bool,
    total_epochs: usize,
    entropy_coeff: f32,
    vf_coeff: Option<f32>,
    target_kl: Option<f32>,
    gradient_clipping: Option<f32>,
    current_epoch: usize,
    report: PPOStats,
    tx: Sender<PPOStats>,
}
