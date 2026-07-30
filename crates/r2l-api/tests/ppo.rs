use std::path::PathBuf;

use r2l_api::{LearningSchedule, PPOAlgorithmBuilder, StepHookBound};

const ENV_NAME: &str = "Pendulum-v1";

#[test]
fn ppo() {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ppo.safetensor");
    let hidden_layers = vec![64, 64];
    let ppo_builder = PPOAlgorithmBuilder::gym(ENV_NAME, 10)
        .with_burn()
        .with_seed(0)
        .with_policy_hidden_layers(hidden_layers.clone())
        .with_clip_range(0.2)
        .with_entropy_coeff(0.)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_rollout_bound(StepHookBound::new(1024))
        .with_total_epochs(10)
        .with_learning_schedule(LearningSchedule::rollout_bound(30));
    let mut ppo = ppo_builder.build().unwrap();
    ppo.train().unwrap();
}
