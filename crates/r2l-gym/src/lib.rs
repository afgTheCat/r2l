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
//! At the moment, the adapter supports:
//! - discrete action spaces
//! - box-shaped continuous action spaces
//! - observation spaces that expose a Gymnasium `shape`
//!
//! Other Gymnasium space variants are not yet handled by this crate.

use anyhow::Result;
use pyo3::{
    PyObject, PyResult, Python,
    types::{PyAnyMethods, PyDict},
};
use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    tensor::TensorData,
};

/// Python-backed Gymnasium environment implementing `r2l`'s [`Env`] trait.
///
/// `GymEnv` wraps a Gymnasium environment created through `gymnasium.make` and
/// exposes its observation/action spaces through `r2l-core` space types.
///
/// This wrapper currently supports:
/// - Gymnasium `Discrete` action spaces
/// - Gymnasium `Box` action spaces
///
/// Continuous actions are clipped to the environment's declared bounds before
/// stepping. Discrete actions are expected in the action tensor format produced
/// by the rest of the `r2l` stack.
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
            let action_space = env.getattr("action_space")?;
            let gym_spaces = py.import("gymnasium.spaces")?;
            let action_space = if action_space.is_instance(&gym_spaces.getattr("Discrete")?)? {
                let val = action_space.getattr("n")?.extract()?;
                Space::Discrete(val)
            } else if action_space.is_instance(&gym_spaces.getattr("Box")?)? {
                let shape: Vec<usize> = action_space.getattr("shape")?.extract()?;
                let low: Vec<f32> = action_space.getattr("low")?.extract()?;
                let low = TensorData::new(low, shape.clone());
                let high: Vec<f32> = action_space.getattr("high")?.extract()?;
                let high = TensorData::new(high, shape.clone());
                Space::Continuous {
                    min: Some(low),
                    max: Some(high),
                    shape,
                }
            } else {
                todo!("Other actions spaces are not yet supported");
            };
            let observation_space = env.getattr("observation_space")?;
            let observation_space: Vec<usize> = observation_space.getattr("shape")?.extract()?;
            let observation_space = Space::continuous_from_dims(observation_space);
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
            let state = step.get_item(0)?.extract()?;
            PyResult::Ok(TensorData::from_vec(state))
        })?;
        Ok(state)
    }

    fn step(&mut self, action: TensorData) -> Result<Snapshot<TensorData>> {
        let snapshot = Python::with_gil(|py| {
            let step = match &self.action_space {
                Space::Continuous {
                    min: Some(min),
                    max: Some(max),
                    ..
                } => {
                    let clipped_action = action.clamp(min, max);
                    let action_vec: Vec<f32> = clipped_action.into_vec();
                    self.env.call_method(py, "step", (action_vec,), None)?
                }
                _ => {
                    let action: Vec<f32> = action.into_vec();
                    // TODO: remove unwrap
                    let action = action.iter().position(|i| *i > 0.).unwrap();
                    self.env.call_method(py, "step", (action,), None)?
                }
            };
            let step = step.bind(py);
            let next_state: Vec<f32> = step.get_item(0)?.extract()?;
            let next_state = TensorData::from_vec(next_state);
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
