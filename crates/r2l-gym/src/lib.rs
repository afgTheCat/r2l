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
//! Observations are converted into flat [`TensorData`] values. Discrete
//! observations are one-hot encoded, while structured `Tuple` and `Dict`
//! observations are flattened recursively.

use anyhow::Result;
use pyo3::{
    PyObject, PyResult, Python,
    types::{PyAnyMethods, PyDict},
};
use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    tensor::TensorData,
};

mod parse;

use parse::{parse_action, parse_gym_space, parse_obs};

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
    env: PyObject,
    action_space: Space<TensorData>,
    observation_space: Space<TensorData>,
}

impl GymEnv {
    /// Creates a Gymnasium environment by name.
    ///
    /// `render_mode` is forwarded to `gymnasium.make` when provided.
    pub fn new(name: &str, render_mode: Option<String>) -> Result<GymEnv> {
        let env = Python::with_gil(|py| {
            let gym = py.import("gymnasium")?;
            let kwargs = PyDict::new(py);
            if let Some(render_mode) = render_mode {
                kwargs.set_item("render_mode", render_mode)?;
            }
            let make = gym.getattr("make")?;
            let env = make.call((name,), Some(&kwargs))?;
            let gym_spaces = py.import("gymnasium.spaces")?;
            let action_space = env.getattr("action_space")?;
            let action_space = parse_gym_space(&action_space, &gym_spaces)?;
            let observation_space = env.getattr("observation_space")?;
            let observation_space = parse_gym_space(&observation_space, &gym_spaces)?;
            PyResult::Ok(GymEnv {
                env: env.into(),
                action_space,
                observation_space,
            })
        });
        env.map_err(anyhow::Error::from)
    }

    /// Returns the flattened observation size expected by `r2l`.
    pub fn observation_size(&self) -> usize {
        self.observation_space.size()
    }

    /// Returns the action size expected by `r2l`.
    pub fn action_size(&self) -> usize {
        self.action_space.size()
    }

    /// Returns the observation space description discovered from Gymnasium.
    pub fn observation_space(&self) -> Space<TensorData> {
        self.observation_space.clone()
    }

    /// Returns the action space description discovered from Gymnasium.
    pub fn action_space(&self) -> Space<TensorData> {
        self.action_space.clone()
    }

    /// Returns `(action_size, observation_size)`.
    pub fn io_sizes(&self) -> (usize, usize) {
        (self.action_size(), self.observation_size())
    }
}

impl Env for GymEnv {
    type Tensor = TensorData;

    fn reset(&mut self, seed: u64) -> Result<TensorData> {
        let state = Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let state = self.env.call_method(py, "reset", (), Some(&kwargs))?;
            let step = state.bind(py);
            parse_obs(&step.get_item(0)?, &self.observation_space)
        })?;
        Ok(state)
    }

    fn step(&mut self, action: TensorData) -> Result<Snapshot<TensorData>> {
        let snapshot = Python::with_gil(|py| {
            let action = parse_action(py, &action.into_vec(), &self.action_space)?;
            let step = self.env.call_method(py, "step", (action,), None)?;
            let step = step.bind(py);
            let next_state = parse_obs(&step.get_item(0)?, &self.observation_space)?;
            let reward: f32 = step.get_item(1)?.extract()?;
            let terminated: bool = step.get_item(2)?.extract()?;
            let truncated: bool = step.get_item(3)?.extract()?;
            let snapshot = Snapshot::new(next_state, reward, terminated, truncated);
            PyResult::Ok(snapshot)
        })?;
        Ok(snapshot)
    }

    fn env_description(&self) -> EnvDescription<TensorData> {
        EnvDescription {
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
        }
    }
}

/// Builder for named Gymnasium environments.
///
/// This is the standard way to plug Gymnasium environments into higher-level
/// `r2l` builders such as `r2l_api::PPOAlgorithmBuilder` and
/// `r2l_api::A2CAlgorithmBuilder`.
pub struct GymEnvBuilder(String);

impl GymEnvBuilder {
    /// Creates a builder for the given Gymnasium environment id.
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

    fn build_env(&self) -> Result<Self::Env> {
        GymEnv::new(&self.0, None)
    }
}
