// ANCHOR: on_policy
use r2l_api::{
    Location, StepTrajectoryBound,
    builders::{
        on_policy::OnPolicyAlgorithmBuilder, ppo::agent::PPOAgentBuilder, sampler::SamplerBuilder,
    },
    hooks::on_policy::LearningSchedule,
};
use r2l_gym::GymEnvBuilder;

fn main() {
    let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
    let sampler_builder = SamplerBuilder::<GymEnvBuilder>::new(gym_env_builder, 10)
        .with_location(Location::Thread)
        .with_bound(StepTrajectoryBound::new(1000));
    let agent_builder = PPOAgentBuilder::new(10).with_normalize_advantage(true);
    let algo_builder =
        OnPolicyAlgorithmBuilder::from_sampler_and_agent_builder(sampler_builder, agent_builder)
            .with_learning_schedule(LearningSchedule::rollout_bound(10))
            .with_learning_schedule(LearningSchedule::total_step_bound(1000));
    let mut algo = algo_builder.build().unwrap();
    algo.train().unwrap();
}
// ANCHOR_END: on_policy
