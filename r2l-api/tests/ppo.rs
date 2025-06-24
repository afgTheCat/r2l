use candle_core::Device;
use candle_core::Result;
use r2l_core::{
    env::RolloutMode,
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm, OnPolicyHooks},
};
use r2l_gym::gym_dummy_vec_env;

// #[test]
// fn ppo_testing() -> Result<()> {
//     let device = Device::Cpu;
//     let env_pool = gym_dummy_vec_env("CartPole-v1", &device, 16)?;
//     let learning_schedule = LearningSchedule::TotalStepBound {
//         total_steps: 100000,
//         current_step: 0,
//     };
//     let mut algo = OnPolicyAlgorithm {
//         env_pool,
//         agent,
//         rollout_mode: RolloutMode::StepBound { n_steps: 1024 },
//         learning_schedule,
//         hooks: OnPolicyHooks::default(),
//     };
//     Ok(())
// }
