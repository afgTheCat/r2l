use std::marker::PhantomData;

use r2l_core::env::Env;
use r2l_sampler::{
    DirectSamplerCore, DirectSamplerHook, SamplerHookResult, StagedSamplerCore, StagedSamplerHook,
};

use super::progress::SharedTrainingProgress;
use crate::utils::RewardNormalizer;

enum Phase {
    Collect,
    Stop,
}

/// Marker for a sampler configured to collect a fixed number of steps.
pub struct Steps;
/// Marker for a sampler configured to collect a fixed number of episodes.
pub struct Episodes;

/// Controls collection and optional reward normalization using shared progress.
pub struct SamplerHook<E: Env, Bound> {
    progress: SharedTrainingProgress,
    reward_normalizer: Option<RewardNormalizer>,
    phase: Phase,
    _p: PhantomData<(E, Bound)>,
}

/// Step-bound specialization of the shared sampler hook.
pub type StepBoundHook<E> = SamplerHook<E, Steps>;
/// Episode-bound specialization of the shared sampler hook.
pub type EpisodeBoundHook<E> = SamplerHook<E, Episodes>;

impl<E: Env, Bound> SamplerHook<E, Bound> {
    /// Creates a sampler hook using the shared collection configuration.
    ///
    /// # Arguments
    ///
    /// * `progress` - Progress and collection settings shared with the training hooks.
    /// * `reward_normalizer` - Optional normalization for step-bound rollouts.
    pub(crate) fn new(
        progress: SharedTrainingProgress,
        reward_normalizer: Option<RewardNormalizer>,
    ) -> Self {
        Self {
            progress,
            reward_normalizer,
            phase: Phase::Collect,
            _p: PhantomData,
        }
    }

    fn reset_state(&mut self) {
        self.phase = Phase::Collect;
        if let Some(normalizer) = &mut self.reward_normalizer {
            normalizer.reset_returns();
        }
    }

    fn next_result(&mut self, normalize: impl FnOnce(&mut RewardNormalizer)) -> SamplerHookResult {
        let mode = self.progress.borrow().rollout_mode();
        match self.phase {
            Phase::Collect => {
                self.phase = Phase::Stop;
                SamplerHookResult::Bound(mode)
            }
            Phase::Stop => {
                self.phase = Phase::Collect;
                if mode.is_step_bound()
                    && let Some(normalizer) = &mut self.reward_normalizer
                {
                    normalize(normalizer);
                }
                SamplerHookResult::Stop
            }
        }
    }
}

impl<E: Env, Bound> DirectSamplerHook for SamplerHook<E, Bound> {
    type E = E;

    fn reset(&mut self) {
        self.reset_state();
    }

    fn hook(&mut self, core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        self.next_result(|normalizer| {
            let mut buffers = core.buffers_mut().lock().unwrap();
            normalizer.normalize(&mut buffers);
        })
    }
}

impl<E: Env, Bound> StagedSamplerHook for SamplerHook<E, Bound> {
    type E = E;

    fn reset(&mut self) {
        self.reset_state();
    }

    fn hook(&mut self, core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult {
        self.next_result(|normalizer| normalizer.normalize(core.buffers_mut()))
    }
}
