//! Rollout samplers for `r2l` on-policy algorithms.
//!
//! [`RolloutSampler`] supports direct collection into trajectory buffers and
//! staged collection with shared observation normalization. Both strategies
//! support inline and threaded environment workers through the same hook interface.

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

mod sampler;

pub use sampler::{RolloutSampler, SamplerCore, SamplerHook};

/// Instruction returned by a [`SamplerHook`] during rollout collection.
pub enum SamplerHookResult {
    /// Finish the current rollout.
    Stop,
    /// Collect data up to the supplied bound, then invoke the hook again.
    Bound(RolloutMode),
}

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
        n_episodes: NonZeroUsize,
    },
    /// Collect a fixed number of steps from each selected environment.
    StepBound {
        /// Number of steps required per environment.
        n_steps: NonZeroUsize,
    },
}

impl RolloutMode {
    /// Collects a fixed number of completed episodes per environment.
    ///
    /// # Arguments
    ///
    /// * `n_episodes` - Number of completed episodes required from each environment.
    ///
    /// # Panics
    ///
    /// Panics if `n_episodes` is zero.
    #[must_use]
    pub fn episode_bound(n_episodes: usize) -> Self {
        Self::EpisodeBound {
            n_episodes: NonZeroUsize::new(n_episodes)
                .expect("rollout episodes must be greater than zero"),
        }
    }

    /// Collects a fixed number of steps per environment.
    ///
    /// # Arguments
    ///
    /// * `n_steps` - Number of steps required from each environment.
    ///
    /// # Panics
    ///
    /// Panics if `n_steps` is zero.
    #[must_use]
    pub fn step_bound(n_steps: usize) -> Self {
        Self::StepBound {
            n_steps: NonZeroUsize::new(n_steps).expect("rollout steps must be greater than zero"),
        }
    }

    #[must_use]
    pub fn is_step_bound(&self) -> bool {
        matches!(self, Self::StepBound { .. })
    }

    #[must_use]
    pub fn is_episode_bound(&self) -> bool {
        matches!(self, Self::EpisodeBound { .. })
    }
}
