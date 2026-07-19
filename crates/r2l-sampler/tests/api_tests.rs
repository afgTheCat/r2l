use burn::backend::NdArray;
use r2l_api::{AgentBuilder, EnvBuilder, PPOAgentBuilder, SamplerBuilder};
use r2l_burn::distributions::recurrent_categorical::RecurrentCategoricalDistribution;
use r2l_core::{
    models::Actor,
    on_policy::algorithm::{Agent, Sampler},
    prelude::ActorWrapper,
};
use r2l_gym::GymEnvBuilder;

const N_ENVS: usize = 10;
const ENV: &str = "";
type BACKEND = NdArray;

// TODO:this is too much code, just to collect some rollouts. We should simplify this a whole lot
// in the future.
#[test]
fn non_recursive_sampler_testing() {
    let env_builder = GymEnvBuilder::new(ENV);
    let env_description = env_builder.env_description().unwrap();
    let action_space = env_description.action_space.clone();
    let observation_size = env_description.observation_size();
    let agent = PPOAgentBuilder::new(N_ENVS)
        .build(observation_size, action_space, None)
        .unwrap();
    let actor = agent.actor();
    let actor = ActorWrapper::new(actor);
    // TODO: this is stupid that we need to specify this!
    let sampler_builder: SamplerBuilder<GymEnvBuilder, _, _> =
        SamplerBuilder::new(env_builder, N_ENVS);
    let mut sampler = sampler_builder.build();
    sampler.collect_rollouts(actor);
}

#[test]
fn recursive_sampler_testing() {
    let env_builder = GymEnvBuilder::new(ENV);
    let actor: RecurrentCategoricalDistribution<BACKEND> =
        RecurrentCategoricalDistribution::build(&vec![32, 32]);
    let actor = ActorWrapper::new(actor);
    let sampler_builder: SamplerBuilder<GymEnvBuilder, _, _> =
        SamplerBuilder::new(env_builder, N_ENVS);
    let mut sampler = sampler_builder
        .build_with_state::<<RecurrentCategoricalDistribution<BACKEND> as Actor>::State>();
    sampler.collect_rollouts(actor);
}
