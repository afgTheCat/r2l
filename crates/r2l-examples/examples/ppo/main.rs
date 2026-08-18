// ANCHOR: ppo
use r2l::{GymEnv, InferenceRunner, PPOBuilder, TrainingArtifactsConfig, TrainingLimit};

const ENV_NAME: &str = "Pendulum-v1";
const ARTIFACT_DIR: &str = "runs/pendulum";

fn main() -> anyhow::Result<()> {
    // Train the agent and persist the best policy for inference.
    let artifacts_config = TrainingArtifactsConfig::new(ARTIFACT_DIR);
    let mut ppo = PPOBuilder::gym(ENV_NAME, 10)?
        .with_training_artifacts(artifacts_config)
        .with_policy_hidden_layers(vec![64, 64])
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_training_limit(TrainingLimit::rollouts(30))
        .build()?;
    ppo.train()?;

    // Reload the artifacts later without rebuilding the policy by hand.
    let env = GymEnv::new(ENV_NAME, Some("human".to_owned()))?;
    let mut inference = InferenceRunner::load(ARTIFACT_DIR, env)?;
    for _ in 0..4 {
        inference.run_episode()?;
    }
    Ok(())
}
// ANCHOR_END: ppo
