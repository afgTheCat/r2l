use r2l_api::{
    EvaluationSettings, InferenceArtifacts, LearningSchedule, PPOAlgorithmBuilder,
    SamplerExecutionMode::{MultiThreaded, SingleThreaded},
    TrainingArtifactsConfig,
};
use r2l_gym::GymEnv;

const ENV_NAME: &str = "Pendulum-v1";
const ARTIFACT_DIR: &str = "runs/pendulum";

fn main() {
    // Train the agent and persist the best policy for inference.
    let artifacts_config = TrainingArtifactsConfig::new(ARTIFACT_DIR);
    let mut ppo = PPOAlgorithmBuilder::gym(ENV_NAME, 10)
        .with_training_artifacts(artifacts_config)
        .with_policy_hidden_layers(vec![64, 64])
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_learning_schedule(LearningSchedule::rollout_bound(30))
        .build()
        .unwrap();
    ppo.train().unwrap();

    // Reload the artifacts later without rebuilding the policy by hand.
    let inference_artifacts = InferenceArtifacts::load(ARTIFACT_DIR).unwrap();
    let env = GymEnv::new(ENV_NAME, Some("human".to_owned())).unwrap();
    let mut inference = inference_artifacts.build(env).unwrap();
    for _ in 0..4 {
        inference.run_episode();
    }
}
