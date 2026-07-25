//! Rollout samplers for `r2l` on-policy algorithms.
//!
//! [`R2lSampler`] stores raw observations and rewards, while
//! [`R2lNormalizedSampler`] can normalize observations before exposing
//! trajectories. Both support inline and threaded environment workers.

mod direct;
mod normalized;

pub use direct::worker::WorkerPool;
pub use direct::{R2lSampler, R2lSamplerCore, SamplerHook, SamplerHookResult};
pub use normalized::{
    NormalizedSamplerHook, NormalizerMode, R2lNormalizedSampler, R2lNormalizedSamplerCore,
    clipped_normalizer::ClippedNormalizer,
};

/// Execution strategy used by the sampler.
///
/// This controls whether environment workers run inline in the current thread
/// or in dedicated background threads.
#[derive(Debug, Clone, Copy)]
pub enum SamplerExecutionMode {
    /// Run sampler workers inline in a local vector on the current thread.
    Vec,
    /// Run sampler workers in dedicated background threads.
    Thread,
}

/// Bound used for one rollout collection request per environment.
#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    /// Collect until each selected environment completes `n_episodes`.
    EpisodeBound {
        /// Number of completed episodes required per environment.
        n_episodes: usize,
    },
    /// Collect a fixed number of steps from each selected environment.
    StepBound {
        /// Number of steps required per environment.
        n_steps: usize,
    },
}
