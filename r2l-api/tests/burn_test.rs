use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamWConfig,
};
use r2l_agents::burn_agents::ppo::{BurnPPO, BurnPPOCore, EmptyBurnPPOHooks};
use r2l_api::builders::sampler::{EnvBuilderType, SamplerType};
use r2l_burn_lm::{
    distributions::categorical_distribution::CategoricalDistribution,
    learning_module::{ParalellActorCriticLM, ParalellActorModel},
    sequential::Sequential,
};
use r2l_core::Algorithm;
use r2l_core::{
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
    sampler::NewSampler,
};
use r2l_gym::GymEnv;
use std::sync::Arc;

type MyAutoDiffBackend = Autodiff<NdArray>;

fn build_sampler() -> NewSampler<GymEnv> {
    let sampler_type = SamplerType {
        capacity: 2048,
        hook_options: Default::default(),
        env_pool_type: Default::default(),
    };
    sampler_type.build_with_builder_type(EnvBuilderType::EnvBuilder {
        builder: Arc::new("CartPole-v1".to_owned()),
        n_envs: 1,
    })
}

fn build_algo() -> OnPolicyAlgorithm<
    NewSampler<GymEnv>,
    BurnPPO<MyAutoDiffBackend, CategoricalDistribution<MyAutoDiffBackend>>,
    DefaultOnPolicyAlgorightmsHooks,
> {
    let sampler = build_sampler();
    let env_description = sampler.env_description();
    let logits_layer = [
        env_description.observation_space.size(),
        32,
        32,
        env_description.action_space.size(),
    ];
    let value_net_layers = [env_description.observation_space.size(), 32, 32, 1];
    let distr: CategoricalDistribution<MyAutoDiffBackend> =
        CategoricalDistribution::build(&logits_layer);
    let value_net = Sequential::build(&value_net_layers);
    let paralell_actor_model = ParalellActorModel::new(distr, value_net);
    let paralell_actor_critic_lm =
        ParalellActorCriticLM::new(paralell_actor_model, AdamWConfig::new().init());
    let ppo_core = BurnPPOCore::new(paralell_actor_critic_lm, 0.1, 100, 0.99, 0.99);
    let agent = BurnPPO::new(ppo_core, Box::new(EmptyBurnPPOHooks));
    let op_hooks = DefaultOnPolicyAlgorightmsHooks::new(LearningSchedule::RolloutBound {
        total_rollouts: 100,
        current_rollout: 0,
    });
    OnPolicyAlgorithm {
        sampler,
        hooks: op_hooks,
        agent,
    }
}

#[test]
fn test_burn_rl() {
    let mut algo = build_algo();
    algo.train();
}
