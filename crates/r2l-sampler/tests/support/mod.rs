use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use r2l_core::{
    env::{Env, EnvDescription, Snapshot, Space},
    error::Error,
    models::Actor,
    tensor::{R2lTensor, VecTensor},
};
use r2l_sampler::{
    DirectSamplerCore, DirectSamplerHook, RolloutMode, SamplerHookResult, StagedSamplerCore,
    StagedSamplerHook,
};

#[derive(Clone, Copy)]
pub enum EpisodeEnd {
    Terminated,
    Truncated,
}

pub struct TestEnv {
    step: usize,
    episode_len: usize,
    episode_end: EpisodeEnd,
    fail_at_step: Option<usize>,
    reset_count: Arc<AtomicUsize>,
}

impl TestEnv {
    pub fn new(
        episode_len: usize,
        episode_end: EpisodeEnd,
        fail_at_step: Option<usize>,
        reset_count: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            step: 0,
            episode_len,
            episode_end,
            fail_at_step,
            reset_count,
        }
    }
}

impl Env for TestEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor, Error> {
        self.step = 0;
        self.reset_count.fetch_add(1, Ordering::SeqCst);
        Ok(VecTensor::from_vec(vec![0.0]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>, Error> {
        self.step += 1;
        if self.fail_at_step == Some(self.step) {
            return Err(Error::InvalidState {
                operation: "test environment step".into(),
                details: "configured failure".into(),
            });
        }
        let done = self.step == self.episode_len;
        let (terminated, truncated) = match self.episode_end {
            EpisodeEnd::Terminated => (done, false),
            EpisodeEnd::Truncated => (false, done),
        };
        Ok(Snapshot::new(
            VecTensor::from_vec(vec![self.step as f32]),
            self.step as f32,
            terminated,
            truncated,
        ))
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: vec![1],
            },
            Space::Discrete(1),
        )
    }
}

#[derive(Clone)]
pub struct ConstantActor;

impl Actor for ConstantActor {
    type Tensor = VecTensor;

    fn action(&self, _observation: Self::Tensor) -> Result<Self::Tensor, Error> {
        Ok(VecTensor::from_vec(vec![0.0]))
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor, Error> {
        self.action(observation)
    }
}

pub struct OneBoundHook {
    bound: RolloutMode,
    scheduled: bool,
}

impl OneBoundHook {
    pub fn steps(n_steps: usize) -> Self {
        Self {
            bound: RolloutMode::StepBound { n_steps },
            scheduled: false,
        }
    }

    pub fn episodes(n_episodes: usize) -> Self {
        Self {
            bound: RolloutMode::EpisodeBound { n_episodes },
            scheduled: false,
        }
    }

    fn next(&mut self) -> SamplerHookResult {
        if self.scheduled {
            self.scheduled = false;
            SamplerHookResult::Stop
        } else {
            self.scheduled = true;
            SamplerHookResult::Bound(self.bound)
        }
    }
}

impl DirectSamplerHook for OneBoundHook {
    type E = TestEnv;

    fn hook(&mut self, _core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        self.next()
    }

    fn reset(&mut self) {
        self.scheduled = false;
    }
}

impl StagedSamplerHook for OneBoundHook {
    type E = TestEnv;

    fn hook(&mut self, _core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult {
        self.next()
    }

    fn reset(&mut self) {
        self.scheduled = false;
    }
}

#[derive(Debug, PartialEq)]
pub struct OwnedTrajectory {
    pub states: Vec<f32>,
    pub next_states: Vec<f32>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
}

impl OwnedTrajectory {
    pub fn from_view(view: &r2l_core::buffers::buffer::TrajectoryView<'_, VecTensor>) -> Self {
        Self {
            states: view
                .states
                .iter()
                .map(|state| state.to_vec().unwrap()[0])
                .collect(),
            next_states: view
                .next_states
                .iter()
                .map(|state| state.to_vec().unwrap()[0])
                .collect(),
            rewards: view.rewards.to_vec(),
            terminated: view.terminated.to_vec(),
            truncated: view.truncated.to_vec(),
        }
    }
}
