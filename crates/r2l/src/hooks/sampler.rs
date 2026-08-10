use std::marker::PhantomData;

use r2l_core::{env::Env, tensor::R2lTensor};
use r2l_sampler::{
    DirectSamplerCore, DirectSamplerHook, RolloutMode, SamplerHookResult, StagedSamplerCore,
    StagedSamplerHook,
};

use crate::utils::RewardNormalizer;

/// Sampler hook that requests rollout collection until a fixed number of
/// episodes has been scheduled.
///
/// The hook returns an episode-bound rollout mode once, then returns
/// [`SamplerHookResult::Stop`] on the next call so the outer sampler loop can
/// hand the collected data off for training.
pub struct EpisodeBoundHook<E: Env> {
    num_episodes: usize,
    episodes_scheduled: usize,
    _p: PhantomData<E>,
}

impl<E: Env> EpisodeBoundHook<E> {
    /// Creates an episode-bound sampler hook.
    #[must_use]
    pub fn new(num_episodes: usize) -> Self {
        Self {
            num_episodes,
            episodes_scheduled: 0,
            _p: PhantomData,
        }
    }

    fn next_result(&mut self) -> SamplerHookResult {
        if self.episodes_scheduled == self.num_episodes {
            self.episodes_scheduled = 0;
            SamplerHookResult::Stop
        } else {
            self.episodes_scheduled = self.num_episodes;
            SamplerHookResult::Bound(RolloutMode::EpisodeBound {
                n_episodes: self.num_episodes,
            })
        }
    }
}

impl<E: Env> DirectSamplerHook for EpisodeBoundHook<E> {
    type E = E;

    fn hook(&mut self, _core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        self.next_result()
    }
}

impl<E: Env<Tensor: R2lTensor>> StagedSamplerHook for EpisodeBoundHook<E> {
    type E = E;

    fn hook(&mut self, _core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult {
        self.next_result()
    }
}

/// Sampler hook that requests rollout collection until a fixed number of steps
/// has been scheduled.
///
/// When configured with a reward normalizer, the hook normalizes a completed
/// rollout before handing it to the agent.
pub struct StepBoundHook<E: Env<Tensor: R2lTensor>> {
    num_steps: usize,
    steps_scheduled: usize,
    reward_normalizer: Option<RewardNormalizer>,
    _p: PhantomData<E>,
}

impl<E: Env<Tensor: R2lTensor>> StepBoundHook<E> {
    /// Creates a step-bound sampler hook.
    #[must_use]
    pub fn new(num_steps: usize, reward_normalizer: Option<RewardNormalizer>) -> Self {
        Self {
            num_steps,
            steps_scheduled: 0,
            reward_normalizer,
            _p: PhantomData,
        }
    }

    fn next_result(&mut self) -> SamplerHookResult {
        if self.steps_scheduled == self.num_steps {
            self.steps_scheduled = 0;
            SamplerHookResult::Stop
        } else {
            self.steps_scheduled = self.num_steps;
            SamplerHookResult::Bound(RolloutMode::StepBound {
                n_steps: self.num_steps,
            })
        }
    }
}

impl<E: Env<Tensor: R2lTensor>> DirectSamplerHook for StepBoundHook<E> {
    type E = E;

    fn hook(&mut self, core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        if self.steps_scheduled == self.num_steps
            && let Some(normalizer) = &mut self.reward_normalizer
        {
            let mut buffers = core.buffers_mut().lock().unwrap();
            normalizer.normalize(&mut buffers);
        }
        self.next_result()
    }

    fn reset(&mut self) {
        self.steps_scheduled = 0;
        if let Some(normalizer) = &mut self.reward_normalizer {
            normalizer.reset_returns();
        }
    }
}

impl<E: Env<Tensor: R2lTensor>> StagedSamplerHook for StepBoundHook<E> {
    type E = E;

    fn hook(&mut self, core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult {
        if self.steps_scheduled == self.num_steps
            && let Some(normalizer) = &mut self.reward_normalizer
        {
            normalizer.normalize(core.buffers_mut());
        }
        self.next_result()
    }

    fn reset(&mut self) {
        self.steps_scheduled = 0;
        if let Some(normalizer) = &mut self.reward_normalizer {
            normalizer.reset_returns();
        }
    }
}
