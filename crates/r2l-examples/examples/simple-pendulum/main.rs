use r2l_api::{LearningSchedule2, PPO2AlgorithmBuilder, StepHookBound};

fn main() {
    let builder = PPO2AlgorithmBuilder::gym("Pendulum-v1", 4)
        .with_clip_range(0.2)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_total_epochs(10)
        .with_rollout_bound(StepHookBound::new(1024))
        .with_learning_schedule(LearningSchedule2::total_step_bound(100000));
    let mut algo = builder.build().unwrap();
    algo.train().unwrap();
}
