// Should be shared between the
// - on policy hook
// - ppo/a2c hooks
// - sampler hooks

use std::{marker::PhantomData, sync::Mutex};

use r2l_core::{
    env::Env,
    models::Actor,
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
};
use r2l_sampler::{
    DirectSamplerCore, DirectSamplerHook, RolloutMode, SamplerHookResult, StagedSamplerCore,
    StagedSamplerHook,
};

use crate::utils::RewardNormalizer;

#[derive(Debug, Clone, Copy)]
enum TrainingState {
    RolloutBound {
        total_rollouts: usize,
        current_rollouts: usize,
    },
    TotalSteps {
        total_steps: usize,
        current_steps: usize,
    },
}

enum SamplerBound {
    EpisodeBound { num_episodes: usize },
    StepBound { num_speps: usize },
}

pub struct Coordinator {
    training_state: TrainingState,
    rollout_mode: RolloutMode,
}

impl Coordinator {
    pub fn update_rollout_progress<A: Agent, S: Sampler>(
        &mut self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) -> f64 {
        todo!()
    }

    // TODO: progress in terms of what?
    pub fn progress_remaining(&self) -> f64 {
        todo!()
    }
}

enum Phase {
    Collect,
    Stop,
}

struct SamplerHookImpl<E: Env> {
    // TODO: maybe we do not need Mutex here, and can use references?
    coordinator: Mutex<Coordinator>,
    reward_normalizer: Option<RewardNormalizer>,
    phase: Phase,
    _p: PhantomData<E>,
}

impl<E: Env> SamplerHookImpl<E> {
    fn calculate_bound(&self) -> RolloutMode {
        todo!()
    }
}

impl<E: Env> DirectSamplerHook for SamplerHookImpl<E> {
    type E = E;

    fn reset(&mut self) {
        self.phase = Phase::Collect;
        if let Some(normalizer) = &mut self.reward_normalizer {
            normalizer.reset_returns();
        }
    }

    fn hook(&mut self, core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        let coordinator = self.coordinator.lock().unwrap();
        let mode = coordinator.rollout_mode.clone();
        match self.phase {
            Phase::Stop => {
                self.phase = Phase::Collect;
                if mode.is_step_bound()
                    && let Some(reward_normalizer) = &mut self.reward_normalizer
                {
                    let mut buffers = core.buffers_mut().lock().unwrap();
                    reward_normalizer.normalize(&mut buffers);
                }
                SamplerHookResult::Stop
            }
            Phase::Collect => {
                self.phase = Phase::Stop;
                SamplerHookResult::Bound(mode)
            }
        }
    }
}

impl<E: Env> StagedSamplerHook for SamplerHookImpl<E> {
    type E = E;

    fn reset(&mut self) {
        self.phase = Phase::Collect;
        if let Some(normalizer) = &mut self.reward_normalizer {
            normalizer.reset_returns();
        }
    }

    fn hook(&mut self, core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult {
        let coordinator = self.coordinator.lock().unwrap();
        let mode = coordinator.rollout_mode.clone();
        match self.phase {
            Phase::Stop => {
                self.phase = Phase::Collect;
                if mode.is_step_bound()
                    && let Some(reward_normalizer) = &mut self.reward_normalizer
                {
                    let mut buffers = core.buffers_mut();
                    reward_normalizer.normalize(&mut buffers);
                }
                SamplerHookResult::Stop
            }
            Phase::Collect => {
                self.phase = Phase::Stop;
                SamplerHookResult::Bound(mode)
            }
        }
    }
}
