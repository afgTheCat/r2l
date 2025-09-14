use std::sync::Arc;

use crate::{parse_config::EnvConfig, python_verifier::PythonResult};
use candle_core::Device;
use r2l_agents::AgentKind;
use r2l_api::builders::{
    agents::{a2c::A2CBuilder, ppo::PPOBuilder},
    sampler::{EnvBuilderType, EnvPoolType, SamplerType},
    sampler_hooks2::{EvaluatorNormalizerOptions, EvaluatorOptions, NormalizerOptions},
};
use r2l_core::{
    Algorithm,
    on_policy_algorithm::{DefaultOnPolicyAlgorightmsHooks, LearningSchedule, OnPolicyAlgorithm},
};
use r2l_gym::GymEnvBuilder;

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
    let (evaluator_options, _results) = EvaluatorOptions::new(eval_episodes, eval_freq, 0);
    let normalizer_options = NormalizerOptions::new(1e-8, 0.99, 10., 10.);
    let algo = args.get("algo").unwrap();
    let capacity = match algo.as_str() {
        // sb3 defaults to 2048 as n_steps
        "ppo" => 2048,
        // sb3 defaults to 5 as n_steps
        "a2c" => 5,
        _ => unreachable!(),
    };
    let sampler = SamplerType {
        capacity,
        env_pool_type: EnvPoolType::VecStep,
        hook_options: EvaluatorNormalizerOptions::eval_normalizer(
            evaluator_options,
            normalizer_options,
            Device::Cpu,
        ),
    }
    .build_with_builder_type(EnvBuilderType::EnvBuilder {
        builder: Arc::new(GymEnvBuilder::new(env_name)),
        n_envs,
    });
    let learning_schedule = LearningSchedule::TotalStepBound {
        total_steps: n_timesteps as usize,
        current_step: 0,
    };
    let agent = match algo.as_str() {
        "ppo" => {
            let ppo = PPOBuilder::default()
                .build(&device, &sampler.env_description())
                .unwrap();
            let agent = AgentKind::PPO(ppo);
            agent
        }
        "a2c" => {
            let a2c = A2CBuilder::default()
                .build(&device, &sampler.env_description())
                .unwrap();
            let agent = AgentKind::A2C(a2c);
            agent
        }
        _ => unreachable!(),
    };
    let mut algo = OnPolicyAlgorithm {
        sampler,
        agent,
        hooks: DefaultOnPolicyAlgorightmsHooks::new(learning_schedule),
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
