use candle_core::Device;
use candle_nn::ParamsAdamW;
use r2l_api::{PPOAgentBuilder, SamplerBuilder};
use r2l_gym::GymEnvBuilder;
use r2l_sampler::{EpisodeTrajectoryBound, Location, StepTrajectoryBound};

fn main() {
    let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
    let sampler_builder = SamplerBuilder::<GymEnvBuilder>::new(gym_env_builder, 10)
        .with_location(Location::Vec)
        .with_location(Location::Thread)
        .with_bound(EpisodeTrajectoryBound::new(10))
        .with_bound(StepTrajectoryBound::new(1000));
    let sampler = sampler_builder.build();

    let agents = PPOAgentBuilder::new(10)
        .with_burn()
        .with_candle(Device::Cpu)
        .with_entropy_coeff(0.1)
        .with_vf_coeff(Some(0.1))
        .with_target_kl(Some(0.5))
        .with_gradient_clipping(Some(1.))
        .with_clip_range(0.5)
        .with_gamma(0.98)
        .with_lambda(0.9)
        .with_sample_size(32)
        .with_policy_hidden_layers(vec![32, 32])
        .with_value_hidden_layers(vec![32, 32])
        .with_learning_rate(3e-4)
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_epsilon(1e-5)
        .with_weight_decay(1e-4)
        .with_joint(
            None,
            ParamsAdamW {
                lr: 3e-4,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-5,
                weight_decay: 1e-4,
            },
        );
}
