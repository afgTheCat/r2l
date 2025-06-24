use candle_core::Result;
use r2l_api::builders::on_policy_algo::OnPolicyAlgorithmBuilder;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};

#[test]
fn ppo_testing() -> Result<()> {
    let mut ppo = OnPolicyAlgorithmBuilder::ppo("Pendulum-v1".into());
    ppo.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 100000,
        current_step: 0,
    });
    let mut ppo = ppo.build()?;
    ppo.train()?;
    Ok(())
}
