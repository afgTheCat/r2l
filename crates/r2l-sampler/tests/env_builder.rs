use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    tensor::TensorData,
};
use r2l_sampler::{
    DirectSampler, DirectSamplerCore, DirectSamplerHook, SamplerExecutionMode, SamplerHookResult,
    StagedSampler, StagedSamplerCore, StagedSamplerHook,
};

struct TestEnv;

impl Env for TestEnv {
    type Tensor = TensorData;

    fn reset(&mut self, seed: u64) -> std::result::Result<Self::Tensor, r2l_core::error::Error> {
        Ok(TensorData::new(vec![0.0], vec![1]))
    }

    fn step(
        &mut self,
        action: Self::Tensor,
    ) -> std::result::Result<Snapshot<Self::Tensor>, r2l_core::error::Error> {
        Ok(Snapshot::new(
            TensorData::new(vec![0.0], vec![1]),
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
                shape: vec![1],
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
    let build_count = Arc::new(AtomicUsize::new(0));
    let env_builder: Arc<dyn EnvBuilder<Env = TestEnv>> = {
        let build_count = build_count.clone();
        Arc::new(move || {
            build_count.fetch_add(1, Ordering::Relaxed);
            Ok(TestEnv)
        })
    };

    let _direct = DirectSampler::build_from_env_builder(
        env_builder.clone(),
        2,
        StopHook,
        SamplerExecutionMode::SingleThreaded,
    );
    let _staged = StagedSampler::build_from_env_builder(
        env_builder,
        3,
        StopHook,
        SamplerExecutionMode::SingleThreaded,
        None,
    );

    assert_eq!(build_count.load(Ordering::Relaxed), 5);
}
