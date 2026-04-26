// ANCHOR: on_policy
use r2l_api::{
    LearningSchedule, Location, PPOAlgorithmBuilder, StepTrajectoryBound,
};
use r2l_gym::GymEnvBuilder;

fn main() {
    let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
    let algo_builder = PPOAlgorithmBuilder::new(gym_env_builder, 10)
        .with_location(Location::Thread)
        .with_bound(StepTrajectoryBound::new(1000))
        .with_normalize_advantage(true)
        .with_learning_schedule(LearningSchedule::rollout_bound(10))
        .with_learning_schedule(LearningSchedule::total_step_bound(1000));
    let mut algo = algo_builder.build().unwrap();
    algo.train().unwrap();
}
// ANCHOR_END: on_policy
