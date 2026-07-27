use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use r2l_core::{
    env::{
        Env, EnvBuilder, EnvBuilderType, EnvDescription, Snapshot, Space,
        normalizer::{ClippedNormalizer, NormalizerMode},
    },
    models::Actor,
    on_policy::algorithm::Sampler,
    tensor::TensorData,
};
use r2l_sampler::{
    RolloutMode, SamplerExecutionMode, SamplerHookResult,
    staged2::{StagedSampler2, StagedSampler2Hook, StagedSamplerCore2, WorkerPool2},
};

struct TerminalEnv {
    reset_count: Arc<AtomicUsize>,
}

impl Env for TerminalEnv {
    type Tensor = TensorData;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor> {
        self.reset_count.fetch_add(1, Ordering::Relaxed);
        Ok(TensorData::from_vec(vec![1.0]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>> {
        Ok(Snapshot::new(
            TensorData::from_vec(vec![99.0]),
            1.0,
            true,
            false,
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
struct TerminalEnvBuilder {
    reset_count: Arc<AtomicUsize>,
}

impl EnvBuilder for TerminalEnvBuilder {
    type Env = TerminalEnv;

    fn build_env(&self) -> Result<Self::Env> {
        Ok(TerminalEnv {
            reset_count: self.reset_count.clone(),
        })
    }
}

#[derive(Clone)]
struct TestActor;

impl Actor for TestActor {
    type Tensor = TensorData;

    fn action(&self, _observation: Self::Tensor) -> Result<Self::Tensor> {
        Ok(TensorData::from_vec(vec![1.0]))
    }
}

struct OneStepHook(bool);

impl StagedSampler2Hook for OneStepHook {
    type E = TerminalEnv;

    fn hook(&mut self, _core: &mut StagedSamplerCore2<Self::E>) -> SamplerHookResult {
        if self.0 {
            SamplerHookResult::Stop
        } else {
            self.0 = true;
            SamplerHookResult::Bound(RolloutMode::StepBound { n_steps: 1 })
        }
    }

    fn reset(&mut self) {
        self.0 = false;
    }
}

fn env_builder(reset_count: Arc<AtomicUsize>) -> EnvBuilderType<TerminalEnvBuilder> {
    EnvBuilderType::homogeneous(TerminalEnvBuilder { reset_count }, 1)
}

#[test]
fn worker_pool_preserves_terminal_observation_until_explicit_reset() {
    for execution_mode in [
        SamplerExecutionMode::SingleThreaded,
        SamplerExecutionMode::MultiThreaded,
    ] {
        let reset_count = Arc::new(AtomicUsize::new(0));
        let mut pool = WorkerPool2::build(env_builder(reset_count.clone()), execution_mode);
        pool.set_policy(TestActor);

        let memory = pool.step().pop().unwrap();

        assert!(memory.is_done());
        assert_eq!(memory.next_state.data, vec![99.0]);
        assert_eq!(pool.current_states()[0].data, vec![99.0]);
        assert_eq!(reset_count.load(Ordering::Relaxed), 1);
        pool.shutdown();
    }
}

#[test]
fn staged_sampler_resets_after_buffering_terminal_observation() {
    for execution_mode in [
        SamplerExecutionMode::SingleThreaded,
        SamplerExecutionMode::MultiThreaded,
    ] {
        let reset_count = Arc::new(AtomicUsize::new(0));
        let normalizer = ClippedNormalizer::new(NormalizerMode::ReadOnly, 100.0, vec![1]);
        let mut sampler = StagedSampler2::build(
            env_builder(reset_count.clone()),
            OneStepHook(false),
            execution_mode,
            normalizer,
        );

        sampler.collect_rollouts(TestActor);
        {
            let views = sampler.trajectory_views();
            let view = &views.as_ref()[0];
            assert_eq!(view.terminated, &[true]);
            assert!(view.next_states[0].data[0] > 90.0);
        }
        assert_eq!(reset_count.load(Ordering::Relaxed), 2);
        sampler.shutdown();
    }
}
