// ANCHOR: ppo
use std::path::PathBuf;

use burn::backend::NdArray;
use burn_store::SafetensorsStore;
use r2l_api::{
    Evaluator, LearningSchedule, PPOAlgorithmBuilder, SamplerExecutionMode, StepHookBound,
};
use r2l_burn::distributions::diagonal_distribution::DiagGaussianDistribution;

const ENV_NAME: &str = "Pendulum-v1";

fn main() {
    let best_model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ppo.safetensor");
    let hidden_layers = vec![64, 64];
    let ppo_builder = PPOAlgorithmBuilder::gym(ENV_NAME, 10)
        .with_burn()
        .with_policy_hidden_layers(hidden_layers.clone())
        .with_clip_range(0.2)
        .with_entropy_coeff(0.)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_rollout_bound(StepHookBound::new(1024))
        .with_total_epochs(10)
        .with_learning_schedule(LearningSchedule::rollout_bound(30))
        .with_evaluator_best_actor_path(best_model_path.clone());
    let mut ppo = ppo_builder.build().unwrap();
    ppo.train().unwrap();

    // If we later decide to use the learned model, we can do so by importing it.
    let mut store = SafetensorsStore::from_file(best_model_path);
    let distribution = DiagGaussianDistribution::<NdArray>::from_store(&mut store);
    let (episodes, environments) = (10, 10);
    let mut evaluator =
        Evaluator::gym(ENV_NAME, episodes, environments, SamplerExecutionMode::Vec);
    let results = evaluator.eval(distribution);
    let total_rewards = results
        .as_ref()
        .iter()
        .map(|tr| tr.rewards.iter().sum::<f32>())
        .sum::<f32>();
    println!(
        "Average rewards recieved: {}",
        total_rewards / (episodes * environments) as f32
    );
}
// ANCHOR_END: ppo
