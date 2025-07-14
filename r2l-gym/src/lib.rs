use candle_core::{Device, Error, Result, Tensor};
use pyo3::{
    PyObject, PyResult, Python,
    types::{PyAnyMethods, PyDict},
};
use r2l_core::env::{Env, EnvironmentDescription, Space};

pub struct GymEnv {
    env: PyObject,
    device: Device,
    action_space: Space,
    observation_space: Space,
}

impl GymEnv {
    pub fn new(name: &str, render_mode: Option<String>, device: &Device) -> Result<GymEnv> {
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
                let low = Tensor::from_slice(&low, low.len(), device).unwrap();
                let high: Vec<f32> = action_space.getattr("high")?.extract()?;
                let high = Tensor::from_slice(&high, high.len(), device).unwrap();
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
                device: device.clone(),
            })
        })
        .map_err(Error::wrap)
    }

    pub fn observation_size(&self) -> usize {
        self.observation_space.size()
    }

    pub fn action_size(&self) -> usize {
        self.action_space.size()
    }

    pub fn observation_space(&self) -> Space {
        self.observation_space.clone()
    }

    pub fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    pub fn io_sizes(&self) -> (usize, usize) {
        (self.action_size(), self.observation_size())
    }
}

impl Env for GymEnv {
    fn reset(&mut self, seed: u64) -> Result<Tensor> {
        let state: Vec<f32> = Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let state = self.env.call_method(py, "reset", (), Some(&kwargs))?;
            state.bind(py).get_item(0)?.extract()
        })
        .map_err(Error::wrap)?;
        Tensor::new(state, &self.device)
    }

    fn step(&mut self, action: &Tensor) -> Result<(Tensor, f32, bool, bool)> {
        Python::with_gil(|py| {
            let step = match &self.action_space {
                Space::Continous {
                    min: Some(min),
                    max: Some(max),
                    ..
                } => {
                    let clipped_action = action.clamp(min, max).unwrap();
                    let action_vec: Vec<f32> = clipped_action.to_vec1().unwrap();
                    self.env.call_method(py, "step", (action_vec,), None)?
                }
                _ => {
                    let action: Vec<f32> = action.to_vec1().unwrap();
                    let action = action.iter().position(|i| *i > 0.).unwrap();
                    self.env.call_method(py, "step", (action,), None)?
                }
            };
            let step = step.bind(py);
            let state: Vec<f32> = step.get_item(0)?.extract()?;
            // TODO: remove unwrap
            let state = Tensor::new(state, &self.device).unwrap();
            let reward: f32 = step.get_item(1)?.extract()?;
            let terminated: bool = step.get_item(2)?.extract()?;
            let truncated: bool = step.get_item(3)?.extract()?;
            PyResult::Ok((state, reward, terminated, truncated))
        })
        .map_err(Error::wrap)
    }

    fn env_description(&self) -> EnvironmentDescription {
        EnvironmentDescription {
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
        }
    }
}
