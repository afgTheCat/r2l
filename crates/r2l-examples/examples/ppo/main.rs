// ANCHOR: ppo
use r2l::{Error, GymEnv, InferenceRunner, PPOBuilder, TrainingArtifactsConfig, TrainingLimit};

fn main() -> Result<(), Error> {
    const ENV_NAME: &str = "Pendulum-v1";

    // Path where the training artifacts are going to be stored. Training artifacts could include:
    // - Parameters of the model that was trained + the weights as a safetensor and the optional obs normalizer serialized
    // - Measurements on how the trained model perfoms after each training round
    // - Measurements on how long parts of the trainig run took
    const ARTIFACT_DIR: &str = "runs/pendulum";
    let artifacts_config = TrainingArtifactsConfig::new(ARTIFACT_DIR);

    // An environmnet builder how environments are to be constructed. Environment construction can
    // be elaborate (especially when working with external dependencies), so r2l opts to not pass
    // the environment directly (the environment would have to be Send for multi sampling), but
    // instead accepts anything that implements the `EnvBuilder` trait. Simplest example is just a
    // function/closure that returns the Env.
    let env_builder = || GymEnv::new(ENV_NAME, None);

    // The algorightm is constructed through a PPOBuilder. For A2C, the A2Cbuilder would be
    // equivalent. Builders expose a lot of common parameter setters. To check all the options, you
    // check https://docs.rs/r2l/latest/r2l/type.PPOBuilder.html.
    let mut ppo = PPOBuilder::new(env_builder, 10)?
        .with_training_artifacts(artifacts_config)
        .with_policy_hidden_layers(vec![64, 64])
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_training_limit(TrainingLimit::rollouts(30))
        .build()?;

    // This kicks off and finishes training.
    ppo.train()?;

    // Once training in done, the training artifacts as serialized. You can reuse the trained model
    // by constructing an InferenceRunner. InferenceRunner can single step or run episodes on the
    // environment it recieves.
    let env = GymEnv::new(ENV_NAME, Some("human".to_owned()))?;
    let mut inference = InferenceRunner::load_from_env(ARTIFACT_DIR, env)?;
    for _ in 0..10 {
        inference.run_episode()?;
    }

    Ok(())
}
// ANCHOR_END: ppo
