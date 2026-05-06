use r2l_api::{LearningSchedule, PPOAlgorithmBuilder, StepTrajectoryBound};

fn main() {
    let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
        .with_clip_range(0.2)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_total_epochs(10)
        .with_bound(StepTrajectoryBound::new(1024))
        .with_learning_schedule(LearningSchedule::total_step_bound(100000));
    let mut algo = builder.build().unwrap();
    algo.train().unwrap();
}
