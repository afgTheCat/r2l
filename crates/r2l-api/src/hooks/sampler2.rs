use std::marker::PhantomData;

use bimodal_array::ArrayHandle;
use r2l_core::{buffers::buffer::NewBuffer, env::Env};
use r2l_sampler::{
    RolloutMode,
    sampler2::{SamplerHook2, SamplerHookResult},
};

pub struct EpisodeBoundHook<E: Env> {
    num_episodes: usize,
    episodes_scheduled: usize,
    _p: PhantomData<E>,
}

impl<E: Env> EpisodeBoundHook<E> {
    pub fn new(num_episodes: usize) -> Self {
        Self {
            num_episodes,
            episodes_scheduled: 0,
            _p: PhantomData,
        }
    }
}

impl<E: Env> SamplerHook2 for EpisodeBoundHook<E> {
    type E = E;

    fn hook(
        &mut self,
        _buffer: &mut ArrayHandle<NewBuffer<<Self::E as Env>::Tensor>>,
    ) -> SamplerHookResult {
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

pub struct StepBoundHook<E: Env> {
    num_steps: usize,
    steps_scheduled: usize,
    _p: PhantomData<E>,
}

impl<E: Env> StepBoundHook<E> {
    pub fn new(num_steps: usize) -> Self {
        Self {
            num_steps,
            steps_scheduled: 0,
            _p: PhantomData,
        }
    }
}

impl<E: Env> SamplerHook2 for StepBoundHook<E> {
    type E = E;

    fn hook(
        &mut self,
        _buffer: &mut ArrayHandle<NewBuffer<<Self::E as Env>::Tensor>>,
    ) -> SamplerHookResult {
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
