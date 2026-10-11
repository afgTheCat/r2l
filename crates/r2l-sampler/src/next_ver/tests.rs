use std::{
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use r2l_core::{
    env::{EnvDescription, Space},
    running_mean::RunningMeanStd,
    tensor::VecTensor,
};

use super::*;

struct TestEnv {
    step: usize,
    episode_len: usize,
    fail_reset: bool,
    fail_step: bool,
    drops: Arc<AtomicUsize>,
    _local: Rc<()>,
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl Env for TestEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<VecTensor> {
        if self.fail_reset {
            return Err(Error::invalid_state(
                "reset test environment",
                "reset failed",
            ));
        }
        self.step = 0;
        Ok(VecTensor::from_vec(vec![0.0]))
    }

    fn step(&mut self, _action: VecTensor) -> Result<Snapshot<VecTensor>> {
        if std::mem::take(&mut self.fail_step) {
            return Err(Error::invalid_state("step test environment", "step failed"));
        }
        self.step += 1;
        Ok(Snapshot::new(
            VecTensor::from_vec(vec![self.step as f32]),
            1.0,
            false,
            self.step == self.episode_len,
        ))
    }

    fn env_description(&self) -> EnvDescription<VecTensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: vec![1].into(),
            },
            Space::Discrete(1),
        )
    }
}

#[derive(Clone)]
struct TestActor;

impl Actor for TestActor {
    type Tensor = VecTensor;

    fn action(&self, observation: VecTensor) -> Result<VecTensor> {
        Ok(observation)
    }

    fn mode_action(&self, observation: VecTensor) -> Result<VecTensor> {
        self.action(observation)
    }
}

struct TestHook {
    bound: RolloutMode,
    collected: bool,
    resets: usize,
}

impl TestHook {
    fn new(bound: RolloutMode) -> Self {
        Self {
            bound,
            collected: false,
            resets: 0,
        }
    }
}

impl ReSamplerHook for TestHook {
    type E = TestEnv;

    fn hook(&mut self, pool: &mut Pool<TestEnv>) -> SamplerHookResult {
        if !self.collected {
            self.collected = true;
            SamplerHookResult::Bound(self.bound)
        } else {
            self.collected = false;
            for buffer in pool.buffers_mut().as_mut() {
                buffer
                    .rewards_mut()
                    .iter_mut()
                    .for_each(|reward| *reward *= 2.0);
            }
            SamplerHookResult::Stop
        }
    }

    fn reset(&mut self) {
        self.collected = false;
        self.resets += 1;
    }
}

impl ReSampler<TestEnv, TestHook> {
    fn for_test(
        staged: bool,
        execution_mode: SamplerExecutionMode,
        bound: RolloutMode,
        fail_reset: bool,
        fail_step: bool,
        drops: Arc<AtomicUsize>,
    ) -> Result<Self> {
        let builders = EnvBuilderType::heterogeneous(
            (2..=3)
                .map(|episode_len| {
                    let drops = drops.clone();
                    move || {
                        Ok(TestEnv {
                            step: 0,
                            episode_len,
                            // Fail on the second worker to exercise partial construction cleanup.
                            fail_reset: fail_reset && episode_len == 3,
                            fail_step: fail_step && episode_len == 2,
                            drops: drops.clone(),
                            _local: Rc::new(()),
                        })
                    }
                })
                .collect(),
        )?;
        let hook = TestHook::new(bound);
        if staged {
            let normalizer = Normalizer::new(
                NormalizerMode::ReadOnly,
                RunningMeanStd::build(
                    VecTensor::from_vec(vec![1.0]),
                    VecTensor::from_vec(vec![1.0]),
                    1.0,
                ),
                None,
            );
            Self::build_staged(&builders, hook, execution_mode, normalizer)
        } else {
            Self::build_direct(&builders, hook, execution_mode)
        }
    }
}

const EXECUTION_MODES: [SamplerExecutionMode; 2] = [
    SamplerExecutionMode::SingleThreaded,
    SamplerExecutionMode::MultiThreaded,
];

#[test]
fn collection_preserves_state_and_reset_restores_hook_and_observations() {
    for staged in [false, true] {
        for execution_mode in EXECUTION_MODES {
            let drops = Arc::new(AtomicUsize::new(0));
            let mut sampler = ReSampler::for_test(
                staged,
                execution_mode,
                RolloutMode::step_bound(1),
                false,
                false,
                drops.clone(),
            )
            .unwrap();
            let initial = if staged { -1.0 } else { 0.0 };
            for expected in [initial, initial + 1.0] {
                sampler.collect_rollouts(TestActor).unwrap();
                let views = sampler.trajectory_views();
                for view in views.as_ref() {
                    assert_eq!(view.states.len(), 1);
                    assert_eq!(view.states[0].to_vec().unwrap(), [expected]);
                    assert_eq!(view.actions[0].to_vec().unwrap(), [expected]);
                    assert_eq!(view.rewards, [2.0]);
                }
            }
            sampler.reset_all_envs().unwrap();
            assert_eq!(sampler.hook.resets, 1);
            assert!(
                sampler
                    .trajectory_views()
                    .as_ref()
                    .iter()
                    .all(|view| view.states.is_empty())
            );
            sampler.collect_rollouts(TestActor).unwrap();
            assert_eq!(
                sampler.trajectory_views().as_ref()[0].states[0]
                    .to_vec()
                    .unwrap(),
                [initial]
            );
            drop(sampler);
            assert_eq!(drops.load(Ordering::Relaxed), 2);
        }
    }
}

#[test]
fn episode_bounds_preserve_terminal_observations_and_stop_completed_workers() {
    for staged in [false, true] {
        for execution_mode in EXECUTION_MODES {
            let mut sampler = ReSampler::for_test(
                staged,
                execution_mode,
                RolloutMode::episode_bound(2),
                false,
                false,
                Arc::default(),
            )
            .unwrap();
            sampler.collect_rollouts(TestActor).unwrap();
            let views = sampler.trajectory_views();
            for (idx, view) in views.as_ref().iter().enumerate() {
                let episode_len = idx + 2;
                assert_eq!(view.states.len(), episode_len * 2);
                assert_eq!(view.episode_terminations(), 2);
                assert!(view.terminated.iter().all(|terminated| !terminated));
                let offset = if staged { -1.0 } else { 0.0 };
                assert_eq!(
                    view.next_states[episode_len - 1].to_vec().unwrap(),
                    [episode_len as f32 + offset]
                );
                assert_eq!(view.states[episode_len].to_vec().unwrap(), [offset]);
            }
        }
    }
}

#[test]
fn initialization_errors_propagate_and_release_started_workers() {
    for staged in [false, true] {
        for execution_mode in EXECUTION_MODES {
            let drops = Arc::new(AtomicUsize::new(0));
            let result = ReSampler::for_test(
                staged,
                execution_mode,
                RolloutMode::step_bound(1),
                true,
                false,
                drops.clone(),
            );
            let error = result.err().expect("initial reset should fail");
            assert!(error.to_string().contains("reset failed"));
            assert_eq!(drops.load(Ordering::Relaxed), 2);
        }
    }
}

#[test]
fn collection_errors_allow_an_explicit_reset_and_retry() {
    for staged in [false, true] {
        for execution_mode in EXECUTION_MODES {
            let mut sampler = ReSampler::for_test(
                staged,
                execution_mode,
                RolloutMode::step_bound(2),
                false,
                true,
                Arc::default(),
            )
            .unwrap();
            let error = sampler.collect_rollouts(TestActor).unwrap_err();
            assert!(error.to_string().contains("step failed"));
            sampler.reset_all_envs().unwrap();
            sampler.collect_rollouts(TestActor).unwrap();
            assert!(
                sampler
                    .trajectory_views()
                    .as_ref()
                    .iter()
                    .all(|view| view.states.len() == 2)
            );
        }
    }
}

#[test]
fn full_and_indexed_steps_apply_the_same_running_normalization() {
    for execution_mode in EXECUTION_MODES {
        let mut samplers = [(); 2].map(|()| {
            let mut sampler = ReSampler::for_test(
                true,
                execution_mode,
                RolloutMode::step_bound(5),
                false,
                false,
                Arc::default(),
            )
            .unwrap();
            let Pool::Staged(pool) = &mut sampler.pool else {
                unreachable!()
            };
            pool.obs_normalizer = pool.obs_normalizer.with_mode(NormalizerMode::Update);
            sampler.reset_all_envs().unwrap();
            sampler.pool.set_policy(&TestActor).unwrap();
            sampler
        });
        let [full, indexed] = &mut samplers;
        let (Pool::Staged(full), Pool::Staged(indexed)) = (&mut full.pool, &mut indexed.pool)
        else {
            unreachable!()
        };
        let mut episodes = [0; 2];
        for _ in 0..5 {
            full.step().unwrap();
            indexed.step_indexed(&[0, 1], &mut episodes).unwrap();
        }
        for (full, indexed) in full.buffers.iter().zip(&indexed.buffers) {
            let full = full.to_trajectory_view();
            let indexed = indexed.to_trajectory_view();
            for (actual, expected) in full
                .states
                .iter()
                .chain(full.next_states)
                .zip(indexed.states.iter().chain(indexed.next_states))
            {
                assert_eq!(actual.to_vec().unwrap(), expected.to_vec().unwrap());
            }
        }
    }
}
