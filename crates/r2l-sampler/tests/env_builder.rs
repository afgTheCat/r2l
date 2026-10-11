use std::{
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use r2l_core::{
    env::{Env, EnvBuilder, EnvBuilderType, EnvDescription, Snapshot, Space},
    error::Error,
    on_policy::algorithm::Sampler,
    tensor::VecTensor,
};
use r2l_sampler::{
    DirectSampler, DirectSamplerCore, DirectSamplerHook, SamplerExecutionMode, SamplerHookResult,
    StagedSampler, StagedSamplerCore, StagedSamplerHook,
};

// Environments are constructed on their worker threads and need not be Send or Clone.
struct TestEnv {
    _local: Rc<()>,
}

impl TestEnv {
    fn new() -> Self {
        Self {
            _local: Rc::new(()),
        }
    }
}

impl Env for TestEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor, Error> {
        Ok(VecTensor::new(vec![0.0], vec![1])?)
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>, Error> {
        Ok(Snapshot::new(
            VecTensor::new(vec![0.0], vec![1])?,
            0.0,
            false,
            false,
        ))
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
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

struct StopHook;

impl DirectSamplerHook for StopHook {
    type E = TestEnv;

    fn hook(&mut self, _core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        SamplerHookResult::Stop
    }
}

impl StagedSamplerHook for StopHook {
    type E = TestEnv;

    fn hook(&mut self, _core: &mut StagedSamplerCore<Self::E>) -> SamplerHookResult {
        SamplerHookResult::Stop
    }
}

#[test]
fn samplers_build_from_a_shared_env_builder() {
    for execution_mode in [
        SamplerExecutionMode::SingleThreaded,
        SamplerExecutionMode::MultiThreaded,
    ] {
        let build_count = Arc::new(AtomicUsize::new(0));
        let env_builder: Arc<dyn EnvBuilder<Env = TestEnv>> = {
            let build_count = build_count.clone();
            Arc::new(move || {
                build_count.fetch_add(1, Ordering::Relaxed);
                Ok(TestEnv::new())
            })
        };

        let mut direct =
            DirectSampler::build_from_env_builder(env_builder.clone(), 2, StopHook, execution_mode)
                .unwrap();
        direct.reset_all_envs().unwrap();
        let mut staged =
            StagedSampler::build_from_env_builder(env_builder, 3, StopHook, execution_mode, None)
                .unwrap();
        staged.reset_all_envs().unwrap();

        assert_eq!(build_count.load(Ordering::Relaxed), 5);
    }
}

struct CountingBuilder {
    build_count: Arc<AtomicUsize>,
}

impl EnvBuilder for CountingBuilder {
    type Env = TestEnv;

    fn build_env(&self) -> Result<TestEnv, Error> {
        self.build_count.fetch_add(1, Ordering::Relaxed);
        Ok(TestEnv::new())
    }

    fn env_description(&self) -> Result<EnvDescription<VecTensor>, Error> {
        Ok(TestEnv::new().env_description())
    }
}

#[test]
fn samplers_build_from_different_factory_types_for_the_same_environment() {
    for execution_mode in [
        SamplerExecutionMode::SingleThreaded,
        SamplerExecutionMode::MultiThreaded,
    ] {
        let first_count = Arc::new(AtomicUsize::new(0));
        let second_count = Arc::new(AtomicUsize::new(0));
        let closure_count = second_count.clone();
        let builders = EnvBuilderType::heterogeneous_shared(vec![
            Arc::new(CountingBuilder {
                build_count: first_count.clone(),
            }),
            Arc::new(move || {
                closure_count.fetch_add(1, Ordering::Relaxed);
                Ok(TestEnv::new())
            }),
        ])
        .unwrap();
        assert_eq!(builders.num_envs(), 2);
        builders.env_description().unwrap();
        assert_eq!(first_count.load(Ordering::Relaxed), 0);
        assert!(builders.build_idx(2).is_err());

        let mut direct = DirectSampler::build(builders.clone(), StopHook, execution_mode);
        direct.reset_all_envs().unwrap();
        let mut staged =
            StagedSampler::build_with_obs_normalizer(&builders, StopHook, execution_mode, None)
                .unwrap();
        staged.reset_all_envs().unwrap();

        assert_eq!(first_count.load(Ordering::Relaxed), 2);
        assert_eq!(second_count.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn builder_collections_reject_empty_inputs() {
    assert!(EnvBuilderType::homogeneous(|| Ok(TestEnv::new()), 0).is_err());
    assert!(EnvBuilderType::homogeneous_shared(Arc::new(|| Ok(TestEnv::new())), 0).is_err());
    assert!(EnvBuilderType::<TestEnv>::heterogeneous(Vec::<CountingBuilder>::new()).is_err());
    assert!(EnvBuilderType::<TestEnv>::heterogeneous_shared(vec![]).is_err());
    let builders = EnvBuilderType::heterogeneous(vec![|| Ok(TestEnv::new())]).unwrap();
    assert_eq!(builders.num_envs(), 1);
    builders.build_idx(0).unwrap();
}
