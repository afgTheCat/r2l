use crate::parse_config::ModelConfigs;

#[derive(Debug, Default)]
struct Builder {
    eval_episodes: Option<i64>,
    eval_freq: Option<i64>,
    n_timesteps: Option<i64>,
    n_eval_envs: Option<i64>,
    ent_coef: Option<f64>,
    normalize: Option<bool>, // advantage normalization
    policy: Option<String>,
    learning_rate: Option<f64>,
}

impl Builder {
    fn process_arg(mut self, arg_name: &str, arg_val: &str) -> Option<Self> {
        match arg_name {
            // we already kinda new this one
            "env_kwargs" => {
                if arg_val == "null" {
                    Some(self)
                } else {
                    None
                }
            }
            "eval_episodes" => {
                let eval_ep = arg_val.parse::<i64>().ok()?;
                self.eval_episodes = Some(eval_ep);
                Some(self)
            }
            "eval_freq" => {
                let eval_freq = arg_val.parse::<i64>().ok()?;
                self.eval_freq = Some(eval_freq);
                Some(self)
            }
            "gym_packages" => {
                if arg_val == "[]" {
                    Some(self)
                } else {
                    None
                }
            }
            "n_eval_envs" => {
                let n_eval_envs = arg_val.parse().ok()?;
                self.n_eval_envs = Some(n_eval_envs);
                Some(self)
            }
            "optimize_hyperparameters" => {
                if arg_val == "false" {
                    Some(self)
                } else {
                    None
                }
            }
            // we want to use the n_timesteps from the config.yml
            "n_timesteps" => {
                let timesteps: i64 = arg_val.parse().ok()?;
                if timesteps == -1 { Some(self) } else { None }
            }
            "trained_agent" => {
                if arg_val == "''" {
                    Some(self)
                } else {
                    None
                }
            }
            "hyperparams" => {
                if arg_val == "null" {
                    Some(self)
                } else {
                    None
                }
            }
            "algo" // we already know this
            | "pruner" // pruner is only used when hyperparameter optimization is turned on
            | "device" // device sholt not matter
            | "env" // we already know this
            | "sampler" // sampler is only used when hyperparameter optimization is turned on
            | "log_folder" // logging does not change evaluation
            | "log_interval" // logging does not change evaluation
            | "n_trials" // only if we optimize hyperparameters
            | "save_freq" // saving does not change training
            | "save_replay_buffer" // saving does not change training
            | "seed" // random seed
            | "storage" // only seems to affect the saved file
            | "study_name" // hyperparameter optimization
            | "tensorboard_log" // we do not care about tensorboard
            | "uuid" // for logging stuff
            | "vec_env" // should not matter which env we are using
            | "verbose" // changes the verbosity
            | "truncate_last_trajectory" // only use with pre trained agents
            | "n_jobs" // only used when hyperparameter is optimized
            | "n_evaluations" // only used when hyperparameter is optimized
            | "n_startup_trials" // only used by the pruner + sampler
            | "no_optim_plots" // we don't use optim
            | "wandb_project_name" // we don't use wandb
            | "wandb_entity" // we don't use wandb
            | "track" // we don't use wandb
            | "optimization_log_path" // only when hyperparameter is optimized
            | "num_threads" // Should not matter really
            => Some(self),
            _ => {
                println!("Unhandled arg type: {arg_name}");
                None
            }
        }
    }

    fn process_config(mut self, config_name: &str, config_val: &str) -> Option<Self> {
        match config_name {
            "n_envs" => Some(self),
            "ent_coef" => {
                let ent_coef = config_val.parse().ok()?;
                self.ent_coef = Some(ent_coef);
                Some(self)
            }
            "n_timesteps" => {
                let timesteps: i64 = config_val.parse::<f64>().ok()? as i64;
                self.n_timesteps = Some(timesteps);
                Some(self)
            }
            "normalize" => {
                let normalize: bool = config_val.parse().ok()?;
                self.normalize = Some(normalize);
                Some(self)
            }
            "policy" => {
                if config_val == "MlpPolicy" {
                    self.policy = Some(config_val.to_owned());
                    Some(self)
                } else {
                    None
                }
            }
            "learning_rate" => {
                let learning_rate: f64 = config_val.parse().ok()?;
                self.learning_rate = Some(learning_rate);
                Some(self)
            }
            "env_wrapper" => {
                // TODO: env wrappers are not yet implemented
                None
            }
            _ => None,
        }
    }
}

// we probably need a discriminator on why we are rejecting a model
pub fn test_construct_configs(configs: Vec<ModelConfigs>) {
    let a2c_config = configs.into_iter().find(|c| c.model == "a2c").unwrap();
    let mut builders = vec![];
    // TODO: ehh, loop labels, whatever
    'builder_iter: for env_config in a2c_config.envs.iter() {
        let mut builder = Builder::default();
        for (arg_name, arg_val) in env_config.args.iter() {
            let Some(next_builer) = builder.process_arg(arg_name, arg_val) else {
                continue 'builder_iter;
            };
            builder = next_builer
        }
        for (config_name, config_val) in env_config.config.iter() {
            let Some(next_builer) = builder.process_config(config_name, config_val) else {
                continue 'builder_iter;
            };
            builder = next_builer
        }
        builders.push(builder);
    }

    println!("{:#?}", builders);
}
