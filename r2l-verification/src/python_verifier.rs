use crate::parse_config::{EnvConfig, ModelConfigs};
use once_cell::sync::Lazy;
use pyo3::ffi::c_str;
use pyo3::{PyObject, prelude::*, types::PyDict};
use std::{collections::HashMap, ffi::CString};

pub type Converter = fn(&String, Python<'_>) -> Option<PyObject>;

fn parse_null(arg_val: &String, py: Python) -> Option<PyObject> {
    (arg_val == "null").then(|| py.None())
}

fn parse_str(arg_val: &String, py: Python) -> Option<PyObject> {
    Some(arg_val.to_object(py))
}

fn parse_int(arg_val: &String, py: Python) -> Option<PyObject> {
    arg_val.parse::<i64>().ok().map(|v| v.to_object(py))
}

fn parse_bool(arg_val: &String, py: Python) -> Option<PyObject> {
    arg_val.parse::<bool>().ok().map(|v| v.to_object(py))
}

static CONVERTERS: Lazy<HashMap<String, (String, Converter)>> = Lazy::new(|| {
    let mut m: HashMap<String, (String, Converter)> = HashMap::new();
    m.insert("env".into(), ("env".into(), parse_str));
    m.insert("env_kwargs".into(), ("env_kwargs".into(), parse_null));
    m.insert("hyperparams".into(), ("hyperparams".into(), parse_null));
    m.insert("storage".into(), ("storage".into(), parse_null));
    m.insert("study_name".into(), ("study_name".into(), parse_null));
    m.insert(
        "eval_episodes".into(),
        ("n_eval_episodes".into(), parse_int),
    );
    m.insert("eval_freq".into(), ("eval_freq".into(), parse_int));
    m.insert("log_interval".into(), ("log_interval".into(), parse_int));
    // TODO: This is possibly irrelevant
    m.insert("n_evaluations".into(), ("n_evaluations".into(), parse_int));
    m.insert("n_jobs".into(), ("n_jobs".into(), parse_int));
    m.insert(
        "n_startup_trials".into(),
        ("n_startup_trials".into(), parse_int),
    );
    m.insert("n_timesteps".into(), ("n_timesteps".into(), parse_int));
    m.insert("n_trials".into(), ("n_trials".into(), parse_int));
    m.insert("save_freq".into(), ("save_freq".into(), parse_int));
    m.insert("seed".into(), ("seed".into(), parse_int));
    m.insert("verbose".into(), ("verbose".into(), parse_int));
    m.insert(
        "optimize_hyperparameters".into(),
        ("optimize_hyperparameters".into(), parse_bool),
    );
    m.insert(
        "save_replay_buffer".into(),
        ("save_replay_buffer".into(), parse_bool),
    );
    m.insert(
        "truncate_last_trajectory".into(),
        ("truncate_last_trajectory".into(), parse_bool),
    );
    m.insert("pruner".into(), ("pruner".into(), parse_str));
    m.insert("sampler".into(), ("sampler".into(), parse_str));
    m.insert(
        "tensorboard_log".into(),
        ("tensorboard_log".into(), parse_str),
    );
    m.insert("trained_agent".into(), ("trained_agent".into(), parse_str));
    m.insert("uuid_str".into(), ("uuid_str".into(), parse_str));
    m.insert("vec_env".into(), ("vec_env_type".into(), parse_str));
    m
});

pub struct PythonBuilder<'py> {
    model_name: String,
    env_name: String,
    env_config: EnvConfig,
    py_dict: Bound<'py, PyDict>, // TODO: we have to prepare the python dict so that we
}

pub struct PythonResult {
    pub mean_rewards: Vec<f64>,
    pub std_rewards: Vec<f64>,
    pub env_config: EnvConfig,
}

impl<'py> PythonBuilder<'py> {
    fn new(model_name: String, env_name: String, py: Python<'py>, env_config: EnvConfig) -> Self {
        let dict = PyDict::new(py);
        Self {
            model_name,
            env_name,
            env_config,
            py_dict: dict,
        }
    }

    fn set_py_arg<K, V>(&mut self, key: K, value: V) -> PyResult<()>
    where
        K: IntoPyObject<'py>,
        V: IntoPyObject<'py>,
    {
        self.py_dict.set_item(key, value)
    }
}

fn construct_builers<'py>(
    configs: &ModelConfigs,
    py: Python<'py>,
) -> PyResult<Vec<PythonBuilder<'py>>> {
    let mut builders = vec![];
    'builder_iter: for env_config in configs.envs.iter() {
        let mut builder = PythonBuilder::new(
            configs.model.to_owned(),
            env_config.env_name.clone(),
            py,
            env_config.clone(),
        );
        // TODO: add some explanation here + better filtering, maybe separate filtering stage?
        for (arg_name, arg_val) in env_config.args.iter().filter(|(arg_name, _)| {
            ![
                "log_folder",
                "gym_packages",
                "algo",
                "num_threads",
                "uuid",
                "save_replay_buffer",
                "env",
            ]
            .contains(&arg_name.as_str())
        }) {
            if let Some((py_name, py_val)) = CONVERTERS
                .get(arg_name)
                .and_then(|(out_key, conv)| conv(arg_val, py).map(|obj| (out_key.clone(), obj)))
            {
                builder.set_py_arg(py_name, py_val)?;
            } else {
                println!("env name: {} arg name: {arg_name}", env_config.env_name);
                continue 'builder_iter;
            };
            builder.set_py_arg("config", env_config.config_file.clone())?;
        }
        builders.push(builder);
    }
    Ok(builders)
}

fn analyze_python_model<'py>(
    func: Bound<'py, PyAny>,
    builder: &PythonBuilder<'py>,
    py: Python<'py>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let exp_manager_args = PyDict::new(py);
    exp_manager_args.set_item("exp_manager_args", builder.py_dict.clone())?;
    let rewards = func.call(
        (&builder.model_name, &builder.env_name),
        Some(&exp_manager_args),
    )?;
    let mean_rewards = rewards.get_item(0)?.extract()?;
    let std_rewards = rewards.get_item(1)?.extract()?;
    Ok((mean_rewards, std_rewards))
}

pub fn python_verification_pass(configs: Vec<ModelConfigs>) -> Vec<PythonResult> {
    let mut python_results = vec![];
    Python::with_gil(|py| {
        let a2c_config = configs.into_iter().find(|c| c.model == "a2c").unwrap();
        let builders = construct_builers(&a2c_config, py)?;
        let module_path = format!(
            "{}/scripts/manually_start_zoo.py",
            env!("CARGO_MANIFEST_DIR")
        );
        let module = PyModule::from_code(
            py,
            CString::new(std::fs::read_to_string(module_path)?)?.as_c_str(),
            c_str!("manually_start_zoo.py"),
            c_str!("manually_start_zoo"),
        )?;
        let func = module.getattr("manual_training_reproduction")?;
        let (mean_rewards, std_rewards) = analyze_python_model(func, &builders[0], py)?;
        println!("{mean_rewards:?}, {std_rewards:?}");
        python_results.push(PythonResult {
            mean_rewards,
            std_rewards,
            env_config: builders[0].env_config.clone(),
        });
        PyResult::Ok(())
    })
    .unwrap();
    python_results
}
