mod direct;
mod normalized;

pub use direct::worker::WorkerPool;
pub use direct::{R2lSampler, SamplerExecutionMode, SamplerHook, SamplerHookResult};
pub use normalized::Coordinator;

#[derive(Debug, Clone, Copy)]
pub enum RolloutMode {
    EpisodeBound { n_episodes: usize },
    StepBound { n_steps: usize },
}
