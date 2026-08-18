//! Gymnasium-backed environment adapters for `r2l`.
//!
//! This crate provides a small bridge between Python Gymnasium environments and
//! the `r2l-core` [`Env`] / [`EnvBuilder`]
//! traits. It is primarily intended for examples and high-level algorithm
//! builders that want to train against standard Gym-style environments without
//! implementing a native Rust environment wrapper first.
//!
//! The main entry points are:
//! - [`GymEnv`], a concrete environment wrapper around a Python Gymnasium env
//! - [`GymEnvBuilder`], an [`EnvBuilder`]
//!   implementation that constructs named Gymnasium environments
//!
//! The adapter maps Gymnasium `Discrete`, `Box`, `MultiDiscrete`,
//! `MultiBinary`, `Tuple`, and `Dict` spaces into `r2l-core` space metadata.
//! Observations are converted into flat [`VecTensor`] values. Discrete
//! observations are one-hot encoded, while structured `Tuple` and `Dict`
//! observations are flattened recursively.

mod parse;

use parse::{parse_action, parse_gym_space, parse_obs};
use pyo3::{
    PyErr, PyObject, PyResult, Python,
    exceptions::PyModuleNotFoundError,
    types::{PyAnyMethods, PyDict},
};
use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    error::{EnvironmentError, Error, MissingDependency},
    tensor::VecTensor,
};

fn map_py_error_to_r2l(py: Python<'_>, operation: &str, error: PyErr) -> Error {
    if error.is_instance_of::<PyModuleNotFoundError>(py) {
        let name = error
            .value(py)
            .getattr("name")
            .and_then(|name| name.extract::<String>())
            .unwrap_or_else(|_| error.to_string());
        Error::MissingDependency(MissingDependency {
            name,
            dependency_type: "Python module".into(),
        })
    } else {
        Error::Environment(EnvironmentError {
            operation: operation.into(),
            source: Box::new(error),
        })
    }
}

// Just a wrapper around the loaded env
struct GymEnvPyhon(PyObject);

impl GymEnvPyhon {
    fn new(py: Python<'_>, name: &str, render_mode: Option<String>) -> Result<Self, Error> {
        let new = || {
            let gym = py.import("gymnasium")?;
            let kwargs = PyDict::new(py);
            if let Some(render_mode) = render_mode {
                kwargs.set_item("render_mode", render_mode)?;
            }
            let make = gym.getattr("make")?;
            let env = make.call((name,), Some(&kwargs))?;
            PyResult::Ok(GymEnvPyhon(env.into()))
        };
        new().map_err(|error| map_py_error_to_r2l(py, "build", error))
    }

    fn observation_space(&self, py: Python<'_>) -> Result<Space<VecTensor>, Error> {
        let observation_space = || {
            let gym_spaces = py.import("gymnasium.spaces")?;
            let observation_space = self.0.getattr(py, "observation_space")?.into_bound(py);
            let observation_space = parse_gym_space(&observation_space, &gym_spaces)?;
            PyResult::Ok(observation_space)
        };
        observation_space()
            .map_err(|error| map_py_error_to_r2l(py, "inspect observation space", error))
    }

    fn action_space(&self, py: Python<'_>) -> Result<Space<VecTensor>, Error> {
        let action_space = || {
            let gym_spaces = py.import("gymnasium.spaces")?;
            let action_space = self.0.getattr(py, "action_space")?.into_bound(py);
            let action_space = parse_gym_space(&action_space, &gym_spaces)?;
            PyResult::Ok(action_space)
        };
        action_space().map_err(|error| map_py_error_to_r2l(py, "inspect action space", error))
    }

    fn reset(
        &self,
        py: Python<'_>,
        seed: u64,
        observation_space: &Space<VecTensor>,
    ) -> Result<VecTensor, Error> {
        let reset = || {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let state = self.0.call_method(py, "reset", (), Some(&kwargs))?;
            let step = state.bind(py);
            parse_obs(&step.get_item(0)?, observation_space)
        };
        reset().map_err(|error| map_py_error_to_r2l(py, "reset", error))
    }

    fn step(
        &self,
        py: Python<'_>,
        action: VecTensor,
        action_space: &Space<VecTensor>,
        observation_space: &Space<VecTensor>,
    ) -> Result<Snapshot<VecTensor>, Error> {
        let step = || {
            let action = parse_action(py, &action.into_vec(), action_space)?;
            let step = self.0.call_method(py, "step", (action,), None)?;
            let step = step.bind(py);
            let next_state = parse_obs(&step.get_item(0)?, observation_space)?;
            let reward: f32 = step.get_item(1)?.extract()?;
            let terminated: bool = step.get_item(2)?.extract()?;
            let truncated: bool = step.get_item(3)?.extract()?;
            let snapshot = Snapshot::new(next_state, reward, terminated, truncated);
            PyResult::Ok(snapshot)
        };
        step().map_err(|error| map_py_error_to_r2l(py, "step", error))
    }
}

/// Python-backed Gymnasium environment implementing `r2l`'s [`Env`] trait.
///
/// `GymEnv` wraps a Gymnasium environment created through `gymnasium.make` and
/// exposes its observation/action spaces through `r2l-core` space types.
///
/// This wrapper currently supports Gymnasium `Discrete`, `Box`,
/// `MultiDiscrete`, `MultiBinary`, `Tuple`, and `Dict` spaces.
///
/// Box actions are clipped to the environment's declared bounds before
/// stepping. Structured actions are read from flat tensors and recursively
/// rebuilt into the Python values expected by Gymnasium.
pub struct GymEnv {
    env: GymEnvPyhon,
    action_space: Space<VecTensor>,
    observation_space: Space<VecTensor>,
}

impl GymEnv {
    /// Creates a Gymnasium environment by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Gymnasium environment id passed to `gymnasium.make`.
    /// * `render_mode` - Optional rendering mode passed to `gymnasium.make`.
    ///
    /// # Errors
    ///
    /// Returns an error if Gymnasium cannot create or inspect the environment.
    pub fn new(name: &str, render_mode: Option<String>) -> Result<GymEnv, Error> {
        Python::with_gil(|py| {
            let gym_env_inner = GymEnvPyhon::new(py, name, render_mode)?;
            let observation_space = gym_env_inner.observation_space(py)?;
            let action_space = gym_env_inner.action_space(py)?;
            Ok(Self {
                env: gym_env_inner,
                action_space,
                observation_space,
            })
        })
    }
}

impl Env for GymEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, seed: u64) -> Result<Self::Tensor, Error> {
        Python::with_gil(|py| self.env.reset(py, seed, &self.observation_space))
    }

    fn step(&mut self, action: Self::Tensor) -> Result<Snapshot<Self::Tensor>, Error> {
        Python::with_gil(|py| {
            self.env
                .step(py, action, &self.action_space, &self.observation_space)
        })
    }

    fn env_description(&self) -> EnvDescription<VecTensor> {
        EnvDescription {
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
        }
    }
}

/// Builder for named Gymnasium environments.
///
/// This is the standard way to plug Gymnasium environments into higher-level
/// high-level builders such as `r2l::PPOBuilder` and `r2l::A2CBuilder`.
pub struct GymEnvBuilder(String);

impl GymEnvBuilder {
    /// Creates a builder for the given Gymnasium environment id.
    ///
    /// # Arguments
    ///
    /// * `name` - Gymnasium environment id used for each environment instance.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl From<String> for GymEnvBuilder {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for GymEnvBuilder {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl EnvBuilder for GymEnvBuilder {
    type Env = GymEnv;

    fn build_env(&self) -> Result<Self::Env, Error> {
        GymEnv::new(&self.0, None)
    }
}
