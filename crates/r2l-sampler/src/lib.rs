mod direct;
mod normalized;

pub use direct::worker::WorkerPool;
pub use direct::{R2lSampler, R2lSamplerCore, SamplerHook, SamplerHookResult};
pub use normalized::{NormalizerMode, R2lNormalizedSampler, clipped_normalizer::ClippedNormalizer};

/// Execution strategy used by the sampler.
///
/// This controls whether environment workers run inline in the current thread
/// or in dedicated background threads.
pub enum SamplerExecutionMode {
    /// Run sampler workers inline in a local vector on the current thread.
    Vec,
    /// Run sampler workers in dedicated background threads.
    Thread,
}

#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}
