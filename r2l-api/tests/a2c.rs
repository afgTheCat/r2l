use candle_core::Result;
use r2l_api::builders::on_policy_algo::OnPolicyAlgorithmBuilder;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};

#[test]
fn a2c_acrobat_test() -> Result<()> {
    let mut a2c = OnPolicyAlgorithmBuilder::a2c();
    a2c.set_learning_schedule(LearningSchedule::total_step_bound(500000));
    let mut a2c = a2c.build("Acrobot-v1".to_owned(), 10)?;
    a2c.train()?;
    Ok(())
}

#[test]
fn a2c_cartpoole_test() -> Result<()> {
    let mut a2c = OnPolicyAlgorithmBuilder::a2c();
    a2c.set_learning_schedule(LearningSchedule::total_step_bound(500000));
    let mut a2c = a2c.build("CartPole-v1".to_owned(), 10)?;
    a2c.train()?;
    Ok(())
}
