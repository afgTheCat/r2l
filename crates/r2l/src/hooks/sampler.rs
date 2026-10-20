use std::marker::PhantomData;

use r2l_core::env::Env;
use r2l_sampler::{RolloutMode, SamplerCore, SamplerHook as SamplerHookTrait, SamplerHookResult};

use crate::utils::RewardNormalizer;

enum Phase {
    Collect,
    Stop,
}

/// Controls collection with a fixed rollout mode and optional reward normalization.
pub struct SamplerHook<E: Env> {
    rollout_mode: RolloutMode,
    phase: Phase,
    pub(crate) reward_normalizer: Option<RewardNormalizer>,
    _p: PhantomData<E>,
}

impl<E: Env> SamplerHook<E> {
    /// Creates a sampler hook with its own collection bound.
    ///
    /// # Arguments
    ///
    /// * `rollout_mode` - Step or episode bound applied to each environment per rollout.
    /// * `reward_normalizer` - Optional normalization applied to collected training rewards.
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
}

impl<E: Env + 'static> SamplerHookTrait for SamplerHook<E> {
    type E = E;

    fn reset(&mut self) {
        self.phase = Phase::Collect;
        if let Some(normalizer) = &mut self.reward_normalizer {
            normalizer.reset_returns();
        }
    }

    fn hook(&mut self, core: &mut SamplerCore<Self::E>) -> SamplerHookResult {
        match self.phase {
            Phase::Collect => {
                self.phase = Phase::Stop;
                SamplerHookResult::Bound(self.rollout_mode)
            }
            Phase::Stop => {
                self.phase = Phase::Collect;
                if let Some(normalizer) = &mut self.reward_normalizer {
                    normalizer.normalize(core.buffers_mut().as_mut());
                }
                SamplerHookResult::Stop
            }
        }
    }
}

/// Rollout sampler with step or episode bounds and optional reward normalization.
pub type OnPolicySampler<E> = r2l_sampler::RolloutSampler<SamplerHook<E>>;
