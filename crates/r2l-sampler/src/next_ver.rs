use bimodal_array::{ArrayHandle, ElementHandle};
use r2l_core::buffers::Memory;
use r2l_core::env::Snapshot;
use r2l_core::error::{Error, Result};
use r2l_core::rng::sample_u64;
use r2l_core::{buffers::buffer::TrajectoryBuffer, env::Env, models::Actor, tensor::R2lTensor};

// there are two ways of going about this one:
// workers write to the buffer or they have something that normalizes shit

struct Worker<T: R2lTensor, E: Env<Tensor = T>> {
    actor: Option<Box<dyn Actor<Tensor = E::Tensor>>>,
    env: E,
}

impl<T: R2lTensor, E: Env<Tensor = T>> Worker<T, E> {
    fn get_policy(&mut self) -> Result<&mut Box<dyn Actor<Tensor = E::Tensor>>> {
        self.actor.as_mut().ok_or_else(|| {
            Error::invalid_state("step staged sampler worker", "no policy has been installed")
        })
    }

    fn with_state_handle(&mut self, state_handler: &mut ElementHandle<T>) -> Result<Memory<T>> {
        let policy = self.get_policy()?;
        let state = state_handler.lock().unwrap().clone();
        let action = policy.action(state.clone())?;
        let Snapshot {
            state: next_state,
            reward,
            terminated,
            truncated,
        } = self.env.step(action.clone())?;
        let done = terminated || truncated;
        *state_handler.lock().unwrap() = if done {
            self.env.reset(sample_u64())?
        } else {
            next_state.clone()
        };
        Ok(Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        })
    }
}

struct ImmediateWorkerPool {}

struct Sampler<E: Env> {
    buffers: ArrayHandle<TrajectoryBuffer<E::Tensor>>,
}
