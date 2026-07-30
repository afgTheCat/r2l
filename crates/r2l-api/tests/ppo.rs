use std::path::PathBuf;

use r2l_api::{InferenceArtifacts, LearningSchedule, PPOAlgorithmBuilder, StepHookBound};
use r2l_gym::GymEnv;

const ENV_NAME: &str = "Pendulum-v1";

#[test]
fn ppo_inference() {
    let inference_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ppo-inference");
    let hidden_layers = vec![64, 64];
    let ppo_builder = PPOAlgorithmBuilder::gym(ENV_NAME, 10)
        .with_inference_dir(&inference_dir)
        .with_seed(0)
        .with_policy_hidden_layers(hidden_layers.clone())
        .with_clip_range(0.2)
        .with_entropy_coeff(0.)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_rollout_bound(StepHookBound::new(1024))
        .with_observation_normalizer(Some(10.0))
        .with_total_epochs(10)
        .with_learning_schedule(LearningSchedule::rollout_bound(30));
    let mut ppo = ppo_builder.build().unwrap();
    ppo.train().unwrap();

    let inference_artifacts = InferenceArtifacts::load(inference_dir).unwrap();
    let env = GymEnv::new(ENV_NAME, Some("human".to_owned())).unwrap();
    let mut inference = inference_artifacts.build(env).unwrap();
    for _ in 0..10 {
        loop {
            let snapshot = inference.step().unwrap();
            if snapshot.terminated || snapshot.truncated {
                break;
            }
        }
        inference.reset().unwrap();
    }
}
