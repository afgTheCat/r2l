// ANCHOR: on_policy
use r2l_api::{LearningSchedule2, PPO2AlgorithmBuilder, SamplerExecutionMode, StepHookBound};
use r2l_gym::GymEnvBuilder;

fn main() {
    let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
    let algo_builder = PPO2AlgorithmBuilder::new(gym_env_builder, 10)
        .with_execution_mode(SamplerExecutionMode::Thread)
        .with_execution_mode(SamplerExecutionMode::Vec)
        .with_rollout_bound(StepHookBound::new(1000))
        .with_normalize_advantage(true)
        .with_learning_schedule(LearningSchedule2::rollout_bound(10))
        .with_learning_schedule(LearningSchedule2::total_step_bound(1000));
    let mut algo = algo_builder.build().unwrap();
    algo.train().unwrap();
}
// ANCHOR_END: on_policy
