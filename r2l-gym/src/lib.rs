use candle_core::{Device, Error, Result, Tensor};
use pyo3::{
    PyObject, PyResult, Python,
    types::{PyAnyMethods, PyDict},
};
use r2l_core::{
    env::{Env, dummy_vec_env::DummyVecEnv},
    utils::rollout_buffer::RolloutBuffer,
};

#[derive(Debug, Clone)]
pub enum ActionSpace {
    Discrete(usize),
    Continous {
        min: Tensor,
        max: Tensor,
        action_size: usize,
    },
}

pub struct GymEnv {
    env: PyObject,
    device: Device,
    action_space: ActionSpace,
    observation_space: Vec<usize>,
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
                ActionSpace::Discrete(val)
            } else if action_space.is_instance(&gym_spaces.getattr("Box")?)? {
                let low: Vec<f32> = action_space.getattr("low")?.extract()?;
                let action_size = low.len();
                let low = Tensor::from_slice(&low, low.len(), device).unwrap();
                let high: Vec<f32> = action_space.getattr("high")?.extract()?;
                let high = Tensor::from_slice(&high, high.len(), device).unwrap();
                ActionSpace::Continous {
                    min: low,
                    max: high,
                    action_size,
                }
            } else {
                todo!("Other actions spaces are not yet supported");
            };
            let observation_space = env.getattr("observation_space")?;
            let observation_space = observation_space.getattr("shape")?.extract()?;
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
        self.observation_space.iter().product()
    }

    pub fn action_size(&self) -> Option<usize> {
        match &self.action_space {
            ActionSpace::Discrete(num_actions) => Some(*num_actions),
            ActionSpace::Continous { .. } => None,
        }
    }

    pub fn action_dim(&self) -> usize {
        match &self.action_space {
            ActionSpace::Discrete(_) => 1,
            ActionSpace::Continous { action_size, .. } => *action_size,
        }
    }

    pub fn io_sizes(&self) -> (usize, usize) {
        (
            self.observation_size(),
            self.action_size().unwrap_or(self.action_dim()),
        )
    }
}

impl Env for GymEnv {
    fn reset(&self, seed: u64) -> Result<Tensor> {
        let state: Vec<f32> = Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let state = self.env.call_method(py, "reset", (), Some(&kwargs))?;
            state.bind(py).get_item(0)?.extract()
        })
        .map_err(Error::wrap)?;
        Tensor::new(state, &self.device)
    }

    fn step(&self, action: &Tensor) -> Result<(Tensor, f32, bool, bool)> {
        let clipped_action = match &self.action_space {
            ActionSpace::Continous { min, max, .. } => action.clamp(min, max)?,
            ActionSpace::Discrete(_) => action.clone(),
        };
        Python::with_gil(|py| {
            let action_vec: Vec<f32> = clipped_action.to_vec1().unwrap();
            let step = self.env.call_method(py, "step", (action_vec,), None)?;
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
}

pub fn gym_dummy_vec_env(
    env_name: &str,
    device: &Device,
    n_env: usize,
) -> Result<DummyVecEnv<GymEnv>> {
    let buffers = vec![RolloutBuffer::default(); n_env];
    let env = (0..10)
        .map(|_| GymEnv::new(env_name, None, &device))
        .collect::<Result<Vec<_>>>()?;
    Ok(DummyVecEnv { buffers, env })
}
