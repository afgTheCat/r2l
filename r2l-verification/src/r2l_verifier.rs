use crate::{parse_config::EnvConfig, python_verifier::PythonResult};
use candle_core::{DType, Device, Tensor};
use r2l_agents::{AgentKind, a2c::builder::A2CBuilder, ppo::builder::PPOBuilder};
use r2l_core::{
    Algorithm,
    env::{
        RolloutMode,
        dummy_vec_env::{DummyVecEnvWithEvaluator, Evaluator},
    },
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm, OnPolicyHooks},
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};
use r2l_gym::GymEnv;

fn r2l_verify(env_config: &EnvConfig) {
    let device = Device::Cpu;
    let args = &env_config.args;
    let config = &env_config.config;
    let eval_freq = args.get("eval_freq").unwrap().parse().unwrap();
    let eval_episodes = args.get("eval_episodes").unwrap().parse().unwrap();
    let env_name = args.get("env").unwrap();
    let n_timesteps: f64 = config.get("n_timesteps").unwrap().parse().unwrap();
    let evaluator = Evaluator {
        env: GymEnv::new(env_name, None, &device).unwrap(),
        eval_episodes,
    };
    let n_envs = config.get("n_envs").unwrap().parse().unwrap();
    let env = (0..n_envs)
        .map(|_| GymEnv::new(env_name, None, &device).unwrap())
        .collect::<Vec<_>>();
    let input_dim = env[0].observation_size();
    let out_dim = env[0].action_size().or(Some(env[0].action_dim())).unwrap();
    let env_pool = DummyVecEnvWithEvaluator {
        buffers: vec![RolloutBuffer::default(); n_envs],
        env,
        evaluator,
        eval_freq,
        eval_step: 0,
        obs_rms: RunningMeanStd::new((n_envs, input_dim), device.clone()), // we will need the obs shape
        ret_rms: RunningMeanStd::new(n_envs, device.clone()),
        clip_obs: 10.,
        clip_rew: 10.,
        epsilon: 1e-8,
        returns: Tensor::zeros(n_envs, DType::F32, &device).unwrap(),
        gamma: 0.99,
    };
    let learning_schedule = LearningSchedule::TotalStepBound {
        total_steps: n_timesteps as usize,
        current_step: 0,
    };
    let algo = args.get("algo").unwrap();
    let (agent, rollout_mode) = match algo.as_str() {
        "ppo" => {
            let ppo = PPOBuilder::default().build().unwrap();
            let agent = AgentKind::PPO(ppo);
            // sb3 defaults to 2048 as n_steps
            let rollout_mode = RolloutMode::StepBound { n_steps: 2048 };
            (agent, rollout_mode)
        }
        "a2c" => {
            let a2c = A2CBuilder {
                input_dim,
                out_dim,
                ..Default::default()
            }
            .build()
            .unwrap();
            let agent = AgentKind::A2C(a2c);
            // sb3 defaults to 5 as n_steps
            let rollout_mode = RolloutMode::StepBound { n_steps: 5 };
            (agent, rollout_mode)
        }
        _ => unreachable!(),
    };
    let mut algo = OnPolicyAlgorithm {
        env_pool,
        agent,
        learning_schedule,
        rollout_mode,
        hooks: OnPolicyHooks::default(),
    };
    algo.train().unwrap();
}

pub fn r2l_verify_results(py_res: &[PythonResult]) {
    for res in py_res {
        r2l_verify(&res.env_config)
    }
}

#[cfg(test)]
mod test {
    use crate::{parse_config::parse_config_files, r2l_verifier::r2l_verify};

    #[test]
    fn acrobat_verification() {
        let configs = parse_config_files().unwrap();
        let a2c_config = configs.into_iter().find(|c| c.model == "a2c").unwrap();
        let acrobat_config = a2c_config
            .envs
            .iter()
            .find(|e| {
                let s = e.args.get("env").unwrap();
                s == "Acrobot-v1"
            })
            .unwrap();
        r2l_verify(acrobat_config);
    }
}
