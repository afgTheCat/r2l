// ANCHOR: ppo
use r2l_api::{LearningSchedule, PPOAlgorithmBuilder, StepTrajectoryBound};

fn main() {
    let ppo_builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 10)
        .with_burn()
        .with_clip_range(0.2)
        .with_entropy_coeff(0.)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_bound(StepTrajectoryBound::new(1024))
        .with_total_epochs(10)
        .with_learning_schedule(LearningSchedule::rollout_bound(5))
        .with_evaluator_eval_path("/home/gabor/projects/r2l/model");
    let mut ppo = ppo_builder.build().unwrap();
    ppo.train().unwrap();
}
// ANCHOR_END: ppo
