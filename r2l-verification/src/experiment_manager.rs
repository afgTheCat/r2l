use crate::parse_config::ModelConfigs;
use once_cell::sync::Lazy;
use pyo3::{
    Bound, IntoPyObject, PyErr, PyObject, PyResult, Python,
    exceptions::PyTypeError,
    types::{PyAnyMethods, PyDict},
};
use std::{collections::HashMap, sync::Mutex};

type Converter = fn(String, Python) -> Option<PyObject>;

fn parse_env_kwarg(arg_val: String, py: Python) -> Option<PyObject> {
    if arg_val == "null" {
        Some(py.None())
    } else {
        None
    }
}

static MAP_CONFIG_TO_EXP_MANAGER: Lazy<
    Mutex<HashMap<String, fn(String, Python) -> Option<PyObject>>>,
> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("env_kwargs".to_string(), parse_env_kwarg as Converter);
    Mutex::new(m)
});

#[derive(Debug)]
struct Builder<'py> {
    eval_episodes: Option<i64>,
    eval_freq: Option<i64>,
    n_timesteps: Option<i64>,
    n_eval_envs: Option<i64>,
    ent_coef: Option<f64>,
    normalize: Option<bool>, // advantage normalization
    policy: Option<String>,
    learning_rate: Option<f64>,
    py_exp_manager_args: Bound<'py, PyDict>, // TODO: we have to prepare the python dict so that we
}

impl<'py> Builder<'py> {
    fn new(py: Python<'py>) -> Self {
        let dict = PyDict::new(py);
        Self {
            eval_episodes: Default::default(),
            eval_freq: Default::default(),
            n_timesteps: Default::default(),
            n_eval_envs: Default::default(),
            ent_coef: Default::default(),
            normalize: Default::default(),
            policy: Default::default(),
            learning_rate: Default::default(),
            py_exp_manager_args: dict,
        }
    }

    fn set_py_arg<K, V>(&mut self, key: K, value: V) -> PyResult<()>
    where
        K: IntoPyObject<'py>,
        V: IntoPyObject<'py>,
    {
        self.py_exp_manager_args.set_item(key, value)
    }

    fn process_arg(
        mut self,
        py: Python<'py>,
        arg_name: &str,
        arg_val: &str,
    ) -> PyResult<Option<Self>> {
        match arg_name {
            // we already kinda new this one
            "env_kwargs" => {
                if arg_val == "null" {
                    self.set_py_arg( "env_kwargs", py.None())?;
                    Ok(Some(self))
                } else {
                    Ok(None)
                }
            }
            "eval_episodes" => {
                let eval_ep = arg_val.parse::<i64>().map_err(PyErr::new::<PyTypeError, _>)?;
                self.eval_episodes = Some(eval_ep);
                Ok(Some(self))
            }
            "eval_freq" => {
                let eval_freq = arg_val.parse::<i64>().map_err(PyErr::new::<PyTypeError, _>)?;
                self.eval_freq = Some(eval_freq);
                self.set_py_arg( "eval_freq", eval_freq)?;
                Ok(Some(self))
            }
            "gym_packages" => {
                if arg_val == "[]" {
                    Ok(Some(self))
                } else {
                    Ok(None)
                }
            }
            "n_eval_envs" => {
                let n_eval_envs = arg_val.parse().map_err(PyErr::new::<PyTypeError, _>)?;
                self.n_eval_envs = Some(n_eval_envs);
                Ok(Some(self))
            }
            "optimize_hyperparameters" => {
                if arg_val == "false" {
                    Ok(Some(self))
                } else {
                    Ok(None)
                }
            }
            // we want to use the n_timesteps from the config.yml
            "n_timesteps" => {
                let timesteps: i64 = arg_val.parse().map_err(PyErr::new::<PyTypeError, _>)?;
                if timesteps == -1 { Ok(Some(self)) } else { Ok(None) }
            }
            "trained_agent" => {
                if arg_val == "''" {
                    Ok(Some(self))
                } else {
                    Ok(None)
                }
            }
            "hyperparams" => {
                if arg_val == "null" {
                    Ok(Some(self))
                } else {
                    Ok(None)
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
            => Ok(Some(self)),
            _ => {
                println!("Unhandled arg type: {arg_name}");
                Ok(None)
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
    Python::with_gil(|py| {
        let a2c_config = configs.into_iter().find(|c| c.model == "a2c").unwrap();
        let mut builders = vec![];
        // TODO: ehh, loop labels, whatever
        'builder_iter: for env_config in a2c_config.envs.iter() {
            let mut builder = Builder::new(py);
            for (arg_name, arg_val) in env_config.args.iter() {
                let Some(next_builer) = builder.process_arg(py, arg_name, arg_val)? else {
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
        PyResult::Ok(())
    })
    .unwrap();
}
