use candle_core::Result;
use r2l_api::builders::env_pool::VecPoolType;
use r2l_api::builders::on_policy_algo::OnPolicyAlgorithmBuilder;
use r2l_api::test_utils::run_gym_episodes;
use r2l_core::agents::Agent;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};

const NUM_ENVIRONMENTS: usize = 10;

#[test]
fn ppo_acrobat() -> Result<()> {
    // TODO: separate OnPolicyAlgorithmBuilder into ppo and a2c
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build("Acrobot-v1".to_owned(), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo_pendulum() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build("Pendulum-v1".to_owned(), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo_cart_pole1() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    let mut ppo = ppo_builder.build("CartPole-v1".to_owned(), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, ppo.agent.distribution())?;
    Ok(())
}

#[test]
fn ppo_cart_pole2() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(5000000));
    ppo_builder.env_pool_type = VecPoolType::Dummy2;
    let mut ppo = ppo_builder.build("CartPole-v1".to_owned(), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, ppo.agent.distribution())?;
    Ok(())
}
