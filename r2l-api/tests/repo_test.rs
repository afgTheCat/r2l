use std::sync::Mutex;

use candle_core::{Device, Result};
use once_cell::sync::Lazy;
use r2l_agents::ppo::PPO;
use r2l_agents::ppo::hooks::PPOHooks;
use r2l_api::builders::agents::ppo::PPOBuilder;
use r2l_api::builders::env_pool::VecPoolType;
use r2l_core::Algorithm;
use r2l_core::agents::Agent;
use r2l_core::distributions::Distribution;
use r2l_core::env::{Env, EnvPool, EnvPoolType, RolloutMode};
use r2l_core::on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm, OnPolicyHooks};
use r2l_core::policies::PolicyKind;
use r2l_core::rng::RNG;
use r2l_core::utils::rollout_buffer::{Advantages, RolloutBuffer};
use r2l_gym::GymEnv;
use rand::Rng;

const ENV_NAME: &str = "Pendulum-v1";

#[derive(Default)]
struct AppData {
    current_epoch: usize,   // to control the learning
    total_epochs: usize,    // to control the learning
    current_rollout: usize, // does not need it any more
    total_rollouts: usize,  // does not need it any more
    target_kl: f32,         // to control the learning
}

static SHARED_APP_DATA: Lazy<Mutex<AppData>> = Lazy::new(|| {
    let app_data = AppData::default();
    Mutex::new(app_data)
});

enum AfterLearningHookResult {
    ShouldStop,
    ShouldContinue,
}

fn after_learning_hook_inner(
    policy: &mut PolicyKind,
) -> candle_core::Result<AfterLearningHookResult> {
    let mut app_data = SHARED_APP_DATA.lock().unwrap();
    app_data.current_epoch += 1;
    let should_stop = app_data.current_epoch == app_data.total_epochs;
    if should_stop {
        // snapshot the learned things, API can be much better
        app_data.current_rollout += 1;
        // the std after learning
        Ok(AfterLearningHookResult::ShouldStop)
    } else {
        Ok(AfterLearningHookResult::ShouldContinue)
    }
}

fn before_learning_hook(
    rollout_buffers: &mut Vec<RolloutBuffer>,
    advantages: &mut Advantages,
) -> candle_core::Result<bool> {
    let mut app_data = SHARED_APP_DATA.lock().unwrap();
    app_data.current_epoch = 0;
    let mut total_rewards: f32 = 0.;
    let mut total_episodes: usize = 0;
    for rb in rollout_buffers {
        total_rewards += rb.rewards.iter().sum::<f32>();
        total_episodes += rb.dones.iter().filter(|x| **x).count();
    }
    advantages.normalize();
    Ok(false)
}

fn batch_hook() -> candle_core::Result<bool> {
    // TODO: this breaks the computation graph. We need to explore our options here. The most
    // reasonable choice seems to be that we switch up our hook interface by not only allowing
    // booleans to be returned, but that seems a lot of work right now
    // *policy_loss = PolicyLoss(policy_loss.add(&entropy_loss)?);

    // TODO: this seems to slow down the learning process quite a bit. Maybe there is an issue with
    // the learning rate?
    // let approx_kl = (batch_data
    //     .ratio
    //     .exp()?
    //     .sub(&Tensor::ones_like(&batch_data.ratio)?))?
    // .sub(&batch_data.ratio)?
    // .mean_all()?
    // .to_scalar::<f32>()?;
    // Ok(approx_kl > 1.5 * app_data.target_kl)
    Ok(false)
}

fn train_algo(device: &Device, env_pool: impl EnvPool) -> Result<PPO<PolicyKind>> {
    let total_rollouts = 2;
    {
        let mut app_data = SHARED_APP_DATA.lock().unwrap();
        app_data.total_epochs = 2;
        app_data.target_kl = 0.01;
        app_data.total_rollouts = total_rollouts;
        app_data.current_rollout = 0;
    }
    let after_learning_hook =
        move |policy: &mut PolicyKind| match after_learning_hook_inner(policy)? {
            AfterLearningHookResult::ShouldStop => Ok(true),
            AfterLearningHookResult::ShouldContinue => Ok(false),
        };
    let mut builder = PPOBuilder::default();
    builder.sample_size = 64;
    let mut agent = builder.build(&device, &env_pool.env_description())?;
    agent.hooks = PPOHooks::empty()
        .add_before_learning_hook(before_learning_hook)
        .add_batching_hook(batch_hook)
        .add_rollout_hook(after_learning_hook);
    let mut algo = OnPolicyAlgorithm {
        env_pool,
        agent,
        learning_schedule: LearningSchedule::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        },
        rollout_mode: RolloutMode::StepBound { n_steps: 1024 },
        hooks: OnPolicyHooks::default(),
    };
    algo.train()?;
    Ok(algo.agent)
}

fn env_pool_3(device: &Device) -> EnvPoolType<GymEnv> {
    VecPoolType::Dummy
        .build(device, ENV_NAME.to_owned(), 10)
        .unwrap()
}

fn evaluate_agent(device: &Device, agent: PPO<PolicyKind>) -> Result<f32> {
    let mut total_rewards = 0.;
    let ep_count = 10;
    for _ in 0..ep_count {
        let mut env = GymEnv::new(ENV_NAME, None, &device)?;
        let seed = RNG.with_borrow_mut(|rng| rng.random());
        let mut state = env.reset(seed)?;
        let (mut action, _) = agent.distribution().get_action(&state.unsqueeze(0)?)?;
        while let (next_state, reward, false, false) = env.step(&action)? {
            total_rewards += reward;
            state = next_state;
            // TODO: unsqueeze seems too much here
            let (next_action, _) = agent.distribution().get_action(&state.unsqueeze(0)?)?;
            action = next_action;
        }
    }
    Ok(total_rewards / ep_count as f32)
}

fn train_algo3(device: &Device) -> Result<PPO<PolicyKind>> {
    let env_pool = env_pool_3(device);
    train_algo(device, env_pool)
}

#[test]
fn repro_test() -> Result<()> {
    let device = Device::new_cuda(0)?;
    device.set_seed(0)?;
    let agent = train_algo3(&device)?;
    let eval = evaluate_agent(&device, agent)?;
    println!("{eval}");
    Ok(())
}
