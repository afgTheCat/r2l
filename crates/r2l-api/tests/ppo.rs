use anyhow::Result;
use candle_core::Device;
use r2l_api::builders::on_policy_algo2::OnPolicyAlgorithmBuilder;
use r2l_api::builders::sampler::EnvPoolType;
use r2l_api::builders::sampler_hooks2::{EvaluatorNormalizerOptions, EvaluatorOptions};
use r2l_api::test_utils::run_gym_episodes;
use r2l_core::agents::Agent;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};
use r2l_gym::GymEnvBuilder;

const NUM_ENVIRONMENTS: usize = 10;

#[test]
fn ppo_acrobat() -> Result<()> {
    // TODO: separate OnPolicyAlgorithmBuilder into ppo and a2c
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("Acrobot-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo_pendulum1() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("Pendulum-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo_pendulum2() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("Pendulum-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo_cart_pole1() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("CartPole-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.policy())?;
    Ok(())
}

#[test]
fn ppo_cart_pole2() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("CartPole-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.policy())?;
    Ok(())
}

#[test]
fn ppo_cart_pole3() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    ppo_builder.set_env_pool_type(EnvPoolType::ThreadStep);
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("CartPole-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.policy())?;
    Ok(())
}

#[test]
fn ppo_pendulum3() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    ppo_builder.set_env_pool_type(EnvPoolType::ThreadStep);
    ppo_builder.set_n_step(2048);
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("Pendulum-v1"), 1)?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo2_cart_pole() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(500000));
    let evaluator_opts = EvaluatorOptions {
        eval_freq: 1000,
        ..EvaluatorOptions::default()
    };
    let eval_res = evaluator_opts.results.clone();
    ppo_builder.set_hook_options(EvaluatorNormalizerOptions::evaluator(
        evaluator_opts,
        Device::Cpu,
    ));
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("CartPole-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    println!("eval res: {:?}", eval_res.lock().unwrap());
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.policy())?;
    Ok(())
}

// Evaluator and normalizer at the same time
#[test]
fn ppo2_cart_pole_normalize() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(500000));
    let evaluator_options = EvaluatorOptions {
        eval_freq: 1000,
        ..EvaluatorOptions::default()
    };
    let eval_res = evaluator_options.results.clone();
    let eval_normalizer = EvaluatorNormalizerOptions::evaluator(evaluator_options, Device::Cpu);
    ppo_builder.set_hook_options(eval_normalizer);
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("CartPole-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    println!("eval res: {:?}", eval_res.lock().unwrap());
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.policy())?;
    Ok(())
}

#[test]
fn ppo2_cart_pole_seq_only() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(500000));
    let mut ppo = ppo_builder.build(GymEnvBuilder::new("CartPole-v1"), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.policy())?;
    Ok(())
}
