use anyhow::Result;
use r2l_api::{
    ClippedNormalizer, DirectInferenceBuilder, Env, EnvDescription, Inference2Builder,
    NormalizerBuilder, NormalizerMode, PPOAlgorithmBuilder, Snapshot, Space,
    StagedInferenceBuilder, TensorData,
};

struct TestEnv;

impl Env for TestEnv {
    type Tensor = TensorData;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor> {
        Ok(TensorData::from_vec(vec![0.0, 0.0]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>> {
        Ok(Snapshot::new(
            TensorData::from_vec(vec![0.0, 0.0]),
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
                shape: vec![2],
            },
            Space::Discrete(2),
        )
    }
}

#[test]
fn derives_serializable_direct_inference_builder() {
    let algorithm_builder = PPOAlgorithmBuilder::gym("CartPole-v1", 4);

    let inference_builder = algorithm_builder.inference_builder();
    let serialized = yaml_serde::to_string(&inference_builder).unwrap();
    let _: Inference2Builder<DirectInferenceBuilder> = yaml_serde::from_str(&serialized).unwrap();
    let _inference = inference_builder.build(TestEnv).unwrap();

    assert!(!serialized.contains("CartPole-v1"));
    assert!(serialized.contains("Direct"));
}

#[test]
fn derives_serializable_staged_inference_builder() {
    let algorithm_builder =
        PPOAlgorithmBuilder::gym("CartPole-v1", 4).with_observation_normalizer(Some(10.0));

    let inference_builder = algorithm_builder.inference_builder();
    let serialized = yaml_serde::to_string(&inference_builder).unwrap();
    let _: Inference2Builder<StagedInferenceBuilder> = yaml_serde::from_str(&serialized).unwrap();
    let normalizer: ClippedNormalizer<TensorData> =
        ClippedNormalizer::build(NormalizerMode::ReadOnly, 10.0, vec![2]);
    let normalizer_builder = NormalizerBuilder::from_normalizer(normalizer);
    let _inference = inference_builder
        .build(TestEnv, normalizer_builder)
        .unwrap();

    assert!(!serialized.contains("CartPole-v1"));
    assert!(serialized.contains("Staged"));
}
