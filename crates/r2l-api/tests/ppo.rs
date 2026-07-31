use std::path::PathBuf;

use r2l_api::{
    BestActorEvaluatorConfig, InferenceArtifacts, LearningSchedule, PPOAlgorithmBuilder,
};
use r2l_gym::GymEnv;

const ENV_NAME: &str = "Pendulum-v1";

#[test]
fn ppo_inference() {
    // build stage
    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ppo-inference");
    let evaluator_config = BestActorEvaluatorConfig::new(&output_dir);
    let mut ppo = PPOAlgorithmBuilder::gym(ENV_NAME, 10)
        .with_evaluator(evaluator_config)
        .with_policy_hidden_layers(vec![64, 64])
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_learning_schedule(LearningSchedule::rollout_bound(30))
        .build()
        .unwrap();

    // training
    ppo.train().unwrap();

    // inference stage
    let inference_artifacts = InferenceArtifacts::load(output_dir).unwrap();
    let env = GymEnv::new(ENV_NAME, Some("human".to_owned())).unwrap();
    let mut inference = inference_artifacts.build(env).unwrap();
    for _ in 0..10 {
        inference.run_episode();
    }
}
