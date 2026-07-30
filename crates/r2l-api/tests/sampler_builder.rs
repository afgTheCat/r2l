use anyhow::Result;
use r2l_api::{
    BuiltSampler, DefaultSamplerBuilder, DirectSampler, Env, EnvBuilder, EnvDescription,
    OnPolicyAlgorithmBuilder, PPOAlgorithmBuilder, PPOCandleAgentBuilder, SamplerBuilder,
    SamplerExecutionMode, Snapshot, Space, StepBoundHook, TensorData,
};

struct TestEnv;

impl Env for TestEnv {
    type Tensor = TensorData;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor> {
        Ok(TensorData::from_vec(vec![0.0]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>> {
        Ok(Snapshot::new(
            TensorData::from_vec(vec![0.0]),
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

#[derive(Clone)]
struct TestEnvBuilder;

impl EnvBuilder for TestEnvBuilder {
    type Env = TestEnv;

    fn build_env(&self) -> Result<Self::Env> {
        Ok(TestEnv)
    }
}

struct CustomSamplerBuilder(DefaultSamplerBuilder<TestEnvBuilder>);

impl SamplerBuilder for CustomSamplerBuilder {
    type EnvBuilder = TestEnvBuilder;
    type Sampler = DirectSampler<TestEnv, StepBoundHook<TestEnv>>;

    fn env_builder(&self) -> &r2l_core::env::EnvBuilderType<Self::EnvBuilder> {
        SamplerBuilder::env_builder(&self.0)
    }

    fn with_execution_mode(self, execution_mode: SamplerExecutionMode) -> Self {
        Self(SamplerBuilder::with_execution_mode(self.0, execution_mode))
    }

    fn build(self) -> BuiltSampler<Self::Sampler> {
        SamplerBuilder::build(self.0)
    }
}

#[test]
fn on_policy_builder_accepts_a_custom_sampler_builder() {
    let sampler_builder = CustomSamplerBuilder(DefaultSamplerBuilder::new(TestEnvBuilder, 1));
    let agent_builder = PPOCandleAgentBuilder::new(1);

    OnPolicyAlgorithmBuilder::from_sampler_and_agent_builder(sampler_builder, agent_builder)
        .build()
        .unwrap();
}

#[test]
fn on_policy_builder_uses_the_shared_build_path_for_normalized_sampling() {
    PPOAlgorithmBuilder::new(TestEnvBuilder, 1)
        .with_observation_normalizer(Some(10.0))
        .build()
        .unwrap();
}
