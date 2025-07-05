use candle_core::{Device, Result};
use r2l_api::{
    builders::{
        env_pool::{EvaluatorNormalizerOptions, EvaluatorOptions, NormalizerOptions},
        on_policy_algo::OnPolicyAlgorithmBuilder,
    },
    run_gym_episodes,
};
use r2l_core::agents::Agent;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};
use r2l_gym::GymEnv;

#[test]
fn ppo2_cart_pole() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo2("CartPole-v1".into());
    ppo_builder.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 500000,
        current_step: 0,
    });
    let eval_env = GymEnv::new(&"CartPole-v1", None, &Device::Cpu)?;
    let evaluator = EvaluatorOptions {
        eval_freq: 1000,
        ..EvaluatorOptions::default()
    }
    .build(eval_env, ppo_builder.env_pool_builder.n_envs);
    let eval_res = evaluator.eval_res();
    ppo_builder.set_eval(evaluator);
    let mut ppo = ppo_builder.build()?;
    ppo.train()?;
    println!("eval res: {:?}", eval_res.lock().unwrap());
    run_gym_episodes("CartPole-v1", 10, ppo.agent.distribution())?;
    Ok(())
}

#[test]
fn ppo2_cart_pole_normalize() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo2("CartPole-v1".into());
    ppo_builder.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 500000,
        current_step: 0,
    });

    let eval_env = GymEnv::new(&"CartPole-v1", None, &Device::Cpu)?;
    let evaluator_options = EvaluatorOptions {
        eval_freq: 1000,
        ..EvaluatorOptions::default()
    };
    let normalizer_options = NormalizerOptions::default();
    let eval_normalizer = EvaluatorNormalizerOptions {
        evaluator_options,
        normalizer_options,
    }
    .build(eval_env, ppo_builder.env_pool_builder.n_envs, Device::Cpu);
    let eval_res = eval_normalizer.evaluator.eval_res();
    ppo_builder.set_eval_normalize(eval_normalizer);
    let mut ppo = ppo_builder.build()?;
    ppo.train()?;
    println!("eval res: {:?}", eval_res.lock().unwrap());
    run_gym_episodes("CartPole-v1", 10, ppo.agent.distribution())?;
    Ok(())
}
