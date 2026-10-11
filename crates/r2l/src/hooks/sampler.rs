use std::marker::PhantomData;

use r2l_core::env::Env;
use r2l_sampler::{RolloutMode, SamplerCore, SamplerHook as SamplerHookTrait, SamplerHookResult};

use crate::utils::RewardNormalizer;

enum Phase {
    Collect,
    Stop,
}

/// Marker for a sampler configured to collect a fixed number of steps.
pub struct Steps;
/// Marker for a sampler configured to collect a fixed number of episodes.
pub struct Episodes;

/// Controls collection with a fixed rollout mode and optional reward normalization.
pub struct SamplerHook<E: Env, Bound> {
    rollout_mode: RolloutMode,
    reward_normalizer: Option<RewardNormalizer>,
    phase: Phase,
    _p: PhantomData<(E, Bound)>,
}

/// Step-bound specialization of the sampler hook.
pub type StepBoundHook<E> = SamplerHook<E, Steps>;
/// Episode-bound specialization of the sampler hook.
pub type EpisodeBoundHook<E> = SamplerHook<E, Episodes>;

impl<E: Env, Bound> SamplerHook<E, Bound> {
    /// Creates a sampler hook with its own collection bound.
    ///
    /// # Arguments
    ///
    /// * `rollout_mode` - Step or episode bound applied to each environment per rollout.
    /// * `reward_normalizer` - Optional normalization for step-bound rollouts.
    pub(crate) fn new(
        rollout_mode: RolloutMode,
        reward_normalizer: Option<RewardNormalizer>,
    ) -> Self {
        Self {
            rollout_mode,
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
        match self.phase {
            Phase::Collect => {
                self.phase = Phase::Stop;
                SamplerHookResult::Bound(self.rollout_mode)
            }
            Phase::Stop => {
                self.phase = Phase::Collect;
                if self.rollout_mode.is_step_bound()
                    && let Some(normalizer) = &mut self.reward_normalizer
                {
                    normalize(normalizer);
                }
                SamplerHookResult::Stop
            }
        }
    }
}

impl<E: Env + 'static, Bound> SamplerHookTrait for SamplerHook<E, Bound> {
    type E = E;

    fn reset(&mut self) {
        self.reset_state();
    }

    fn hook(&mut self, core: &mut SamplerCore<Self::E>) -> SamplerHookResult {
        self.next_result(|normalizer| normalizer.normalize(core.buffers_mut().as_mut()))
    }
}
