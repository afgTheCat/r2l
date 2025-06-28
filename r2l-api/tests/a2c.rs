use candle_core::Result;
use r2l_api::builders::on_policy_algo::OnPolicyAlgorithmBuilder;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};

#[test]
fn a2c_acrobat_test() -> Result<()> {
    let mut a2c = OnPolicyAlgorithmBuilder::a2c("Acrobot-v1".into());
    a2c.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 500000,
        current_step: 0,
    });
    let mut a2c = a2c.build()?;
    a2c.train()?;
    Ok(())
}

#[test]
fn a2c_cartpoole_test() -> Result<()> {
    let mut a2c = OnPolicyAlgorithmBuilder::a2c("CartPole-v1".into());
    a2c.set_learning_schedule(LearningSchedule::TotalStepBound {
        total_steps: 500000,
        current_step: 0,
    });
    let mut a2c = a2c.build()?;
    a2c.train()?;
    Ok(())
}
