use crate::{parse_config::EnvConfig, python_verifier::PythonResult};
use candle_core::Device;
use r2l_agents::AgentKind;
use r2l_api::{
    builders::{
        agents::{a2c::A2C3Builder, ppo::PPO3Builder},
        env_pool::{
            EvaluatorNormalizerOptions, EvaluatorOptions, NormalizerOptions,
            SequentialEnvHookTypes, VecPoolType,
        },
    },
    hooks::on_policy_algo_hooks::LoggerTrainingHook,
};
use r2l_core::{
    Algorithm,
    env::{EnvPool, RolloutMode},
    on_policy_algorithm::{LearningSchedule, OnPolicyAlgorithm2, OnPolicyHooks},
};

fn r2l_verify(env_config: &EnvConfig) {
    let device = Device::Cpu;
    let args = &env_config.args;
    let config = &env_config.config;
    let eval_freq = args.get("eval_freq").unwrap().parse().unwrap();
    let eval_episodes = args.get("eval_episodes").unwrap().parse().unwrap();
    let env_name = args.get("env").unwrap();
    let n_timesteps: f64 = config.get("n_timesteps").unwrap().parse().unwrap();
    let n_envs = config.get("n_envs").unwrap().parse().unwrap();
    println!("{}", eval_freq);
    let (evaluator_options, results) = EvaluatorOptions::new(eval_episodes, eval_freq, 0);
    let normalizer_options = NormalizerOptions::new(1e-8, 0.99, 10., 10.);
    let eval_normalizer_options =
        EvaluatorNormalizerOptions::new(evaluator_options, normalizer_options);
    let env_pool = VecPoolType::Sequential(SequentialEnvHookTypes::EvaluatorNormalizer {
        options: eval_normalizer_options,
    })
    .build(&device, env_name.clone(), n_envs)
    .unwrap();
    let learning_schedule = LearningSchedule::TotalStepBound {
        total_steps: n_timesteps as usize,
        current_step: 0,
    };
    let algo = args.get("algo").unwrap();
    let (agent, rollout_mode) = match algo.as_str() {
        "ppo" => {
            let ppo = PPO3Builder::default()
                .build(&device, &env_pool.env_description())
                .unwrap();
            let agent = AgentKind::PPO(ppo);
            // sb3 defaults to 2048 as n_steps
            let rollout_mode = RolloutMode::StepBound { n_steps: 2048 };
            (agent, rollout_mode)
        }
        "a2c" => {
            let a2c = A2C3Builder::default()
                .build(&device, &env_pool.env_description())
                .unwrap();
            let agent = AgentKind::A2C(a2c);
            // sb3 defaults to 5 as n_steps
            let rollout_mode = RolloutMode::StepBound { n_steps: 5 };
            (agent, rollout_mode)
        }
        _ => unreachable!(),
    };
    let mut on_policy_hooks = OnPolicyHooks::default();
    on_policy_hooks.add_training_hook(LoggerTrainingHook::default());
    let mut algo = OnPolicyAlgorithm2 {
        env_pool,
        agent,
        learning_schedule,
        rollout_mode,
        hooks: on_policy_hooks,
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
