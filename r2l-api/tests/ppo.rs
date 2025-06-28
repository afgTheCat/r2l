use candle_core::Result;
use r2l_api::{
    builders::{env_pool::EvaluatorNormalizerOptions, on_policy_algo::OnPolicyAlgorithmBuilder},
    run_gym_episodes,
};
use r2l_core::agents::Agent;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};

#[test]
fn ppo_acrobat() -> Result<()> {
    let mut ppo = OnPolicyAlgorithmBuilder::ppo("Acrobot-v1".into());
    ppo.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 5000000,
        current_step: 0,
    });
    let mut ppo = ppo.build()?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo_pendulum() -> Result<()> {
    let mut ppo = OnPolicyAlgorithmBuilder::ppo("Pendulum-v1".into());
    ppo.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 5000000,
        current_step: 0,
    });
    let mut ppo = ppo.build()?;
    ppo.train()?;
    Ok(())
}

#[test]
fn ppo_cart_pole() -> Result<()> {
    let mut ppo = OnPolicyAlgorithmBuilder::ppo("CartPole-v1".into());
    ppo.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 5000000,
        current_step: 0,
    });
    // ppo.set_normalize(EvaluatorNormalizerOptions::default());
    let mut ppo = ppo.build()?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, ppo.agent.distribution())?;
    Ok(())
}
