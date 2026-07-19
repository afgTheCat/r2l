use anyhow::Result;
use r2l_core::{
    env::{Env, EnvDescription, Snapshot, Space},
    tensor::{R2lTensor, TensorData},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const SUM_BOUND: f32 = 5.;
const ACTION_BOUND: f32 = 100.;
const DEFAULT_MAX_STEPS: usize = 16;

/// Partially observable control task whose hidden sum must be reconstructed
/// from the sequence of applied deltas.
pub struct SumTargetEnv {
    target: f32,
    sum: f32,
    current_term: f32,
    step: usize,
    max_steps: usize,
    rng: StdRng,
}

impl SumTargetEnv {
    /// Creates an environment with a fixed episode horizon.
    pub fn new(max_steps: usize) -> Self {
        assert!(max_steps > 0, "max steps must be positive");
        Self {
            target: 0.,
            sum: 0.,
            current_term: 0.,
            step: 0,
            max_steps,
            rng: StdRng::seed_from_u64(0),
        }
    }

    fn observation(&self, previous_delta: f32) -> TensorData {
        TensorData::from_vec(vec![self.current_term, previous_delta, self.target])
    }
}

impl Default for SumTargetEnv {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_STEPS)
    }
}

impl Env for SumTargetEnv {
    type Tensor = TensorData;

    fn reset(&mut self, seed: u64) -> Result<Self::Tensor> {
        self.rng = StdRng::seed_from_u64(seed);
        self.target = self.rng.random_range(-5. ..5.);
        self.sum = 0.; // NOTE: could be random
        self.current_term = self.rng.random_range(-5. ..5.);
        self.step = 0;
        Ok(self.observation(0.))
    }

    fn step(&mut self, action: Self::Tensor) -> Result<Snapshot<Self::Tensor>> {
        let coeff = action.to_vec()[0].clamp(-ACTION_BOUND, ACTION_BOUND);
        let previous_sum = self.sum;
        self.sum = (self.sum + self.current_term * coeff).clamp(-SUM_BOUND, SUM_BOUND);
        let applied_delta = self.sum - previous_sum;
        self.current_term = self.rng.random_range(-5. ..5.);
        self.step += 1;
        let reward = -((self.target - self.sum).abs());
        Ok(Snapshot {
            state: self.observation(applied_delta),
            terminated: false,
            truncated: self.step >= self.max_steps,
            reward,
        })
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
        EnvDescription::new(
            Space::Box {
                min: Some(TensorData::from_vec(vec![-5., -10., -5.])),
                max: Some(TensorData::from_vec(vec![5., 10., 5.])),
                shape: vec![3],
            },
            Space::Box {
                min: Some(TensorData::from_vec(vec![-ACTION_BOUND])),
                max: Some(TensorData::from_vec(vec![ACTION_BOUND])),
                shape: vec![1],
            },
        )
    }
}
