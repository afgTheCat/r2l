//! Rollout samplers for `r2l` on-policy algorithms.
//!
//! [`DirectSampler`] lets workers write directly to trajectory buffers, while
//! [`StagedSampler`] receives transitions from workers and can transform them
//! before writing them. Both support single-threaded and multi-threaded
//! environment workers.

mod direct;
mod staged;

pub use direct::worker::WorkerPool;
pub use direct::{DirectSampler, DirectSamplerCore, DirectSamplerHook, SamplerHookResult};
use serde::{Deserialize, Serialize};
pub use staged::{StagedSampler, StagedSamplerCore, StagedSamplerHook};

/// Execution strategy used by the sampler.
///
/// This controls whether environment workers run inline in the current thread
/// or in dedicated background threads.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SamplerExecutionMode {
    /// Run sampler workers inline in a local vector on the current thread.
    SingleThreaded,
    /// Run sampler workers in dedicated background threads.
    MultiThreaded,
}

/// Bound used for one rollout collection request per environment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
