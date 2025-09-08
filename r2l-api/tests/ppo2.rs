use anyhow::Result;
use r2l_api::{
    builders::{
        on_policy_algo2::OnPolicyAlgorithmBuilder,
        sampler_hooks2::{EvaluatorNormalizerOptions, EvaluatorOptions},
    },
    test_utils::run_gym_episodes,
};
use r2l_core::agents::Agent;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};

const NUM_ENVIRONMENTS: usize = 10;

// Evaluator only test
#[test]
fn ppo2_cart_pole() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(500000));
    let evaluator_opts = EvaluatorOptions {
        eval_freq: 1000,
        ..EvaluatorOptions::default()
    };
    let eval_res = evaluator_opts.results.clone();
    ppo_builder.set_hook_options(EvaluatorNormalizerOptions::evaluator(evaluator_opts));
    let mut ppo = ppo_builder.build("CartPole-v1".to_owned(), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    println!("eval res: {:?}", eval_res.lock().unwrap());
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.distribution())?;
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
    let eval_normalizer = EvaluatorNormalizerOptions::evaluator(evaluator_options);
    ppo_builder.set_hook_options(eval_normalizer);
    let mut ppo = ppo_builder.build("CartPole-v1".to_owned(), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    println!("eval res: {:?}", eval_res.lock().unwrap());
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.distribution())?;
    Ok(())
}

#[test]
fn ppo2_cart_pole_seq_only() -> Result<()> {
    let mut ppo_builder = OnPolicyAlgorithmBuilder::ppo();
    ppo_builder.set_learning_schedule(LearningSchedule::total_step_bound(500000));
    let mut ppo = ppo_builder.build("CartPole-v1".to_owned(), NUM_ENVIRONMENTS)?;
    ppo.train()?;
    run_gym_episodes("CartPole-v1", 10, &ppo.agent.distribution())?;
    Ok(())
}
