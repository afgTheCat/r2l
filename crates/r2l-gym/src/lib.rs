use anyhow::Result;
use pyo3::{
    PyObject, PyResult, Python,
    types::{PyAnyMethods, PyDict},
};
use r2l_core::{
    env::{Env, EnvironmentDescription, SnapShot, Space},
    env_builder::EnvBuilderTrait,
    tensor::R2lBuffer,
};

pub struct GymEnv {
    env: PyObject,
    action_space: Space<R2lBuffer>,
    observation_space: Space<R2lBuffer>,
}

impl GymEnv {
    pub fn new(name: &str, render_mode: Option<String>) -> GymEnv {
        Python::with_gil(|py| {
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
                let low: Vec<f32> = action_space.getattr("low")?.extract()?;
                let action_size = low.len();
                let low = R2lBuffer::new(low, vec![action_size]);
                let high: Vec<f32> = action_space.getattr("high")?.extract()?;
                let high = R2lBuffer::new(high, vec![action_size]);
                Space::Continous {
                    min: Some(low),
                    max: Some(high),
                    size: action_size,
                }
            } else {
                todo!("Other actions spaces are not yet supported");
            };
            let observation_space = env.getattr("observation_space")?;
            let observation_space: Vec<usize> = observation_space.getattr("shape")?.extract()?;
            let observation_space = Space::continous_from_dims(observation_space);
            PyResult::Ok(GymEnv {
                env: env.into(),
                action_space,
                observation_space,
            })
        })
        .unwrap()
    }

    pub fn observation_size(&self) -> usize {
        self.observation_space.size()
    }

    pub fn action_size(&self) -> usize {
        self.action_space.size()
    }

    pub fn observation_space(&self) -> Space<R2lBuffer> {
        self.observation_space.clone()
    }

    pub fn action_space(&self) -> Space<R2lBuffer> {
        self.action_space.clone()
    }

    pub fn io_sizes(&self) -> (usize, usize) {
        (self.action_size(), self.observation_size())
    }
}

impl Env for GymEnv {
    type Tensor = R2lBuffer;

    fn reset(&mut self, seed: u64) -> Result<R2lBuffer> {
        let state = Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let state = self.env.call_method(py, "reset", (), Some(&kwargs))?;
            let step = state.bind(py);
            let state = step.get_item(0)?.extract()?;
            PyResult::Ok(R2lBuffer::from_vec(state))
        })?;
        Ok(state)
    }

    fn step(&mut self, action: R2lBuffer) -> Result<SnapShot<R2lBuffer>> {
        let snapshot = Python::with_gil(|py| {
            let step = match &self.action_space {
                Space::Continous {
                    min: Some(min),
                    max: Some(max),
                    ..
                } => {
                    let clipped_action = action.clamp(min, max);
                    let action_vec: Vec<f32> = clipped_action.to_data();
                    self.env.call_method(py, "step", (action_vec,), None)?
                }
                _ => {
                    let action: Vec<f32> = action.to_data();
                    let action = action.iter().position(|i| *i > 0.).unwrap();
                    self.env.call_method(py, "step", (action,), None)?
                }
            };
            let step = step.bind(py);
            let next_state: Vec<f32> = step.get_item(0)?.extract()?;
            // TODO: remove unwrap
            let next_state = R2lBuffer::from_vec(next_state);
            let reward: f32 = step.get_item(1)?.extract()?;
            let terminated: bool = step.get_item(2)?.extract()?;
            let trancuated: bool = step.get_item(3)?.extract()?;
            PyResult::Ok(SnapShot {
                state: next_state,
                reward,
                terminated,
                trancuated,
            })
        })?;
        Ok(snapshot)
    }

    fn env_description(&self) -> EnvironmentDescription<R2lBuffer> {
        EnvironmentDescription {
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
        }
    }
}

pub struct GymEnvBuilder(String);

impl GymEnvBuilder {
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

impl EnvBuilderTrait for GymEnvBuilder {
    type Env = GymEnv;

    fn build_env(&self) -> Result<Self::Env> {
        Ok(GymEnv::new(&self.0, None))
    }
}
