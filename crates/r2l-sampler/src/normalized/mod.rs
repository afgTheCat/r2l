// Fun programming. This is very reminacsent to the the VecNormalize in sb3
//
// Idea: implement a direct way of normalizing the environment. Workers not have access to the same
// buffer, instead they return the observation and the reward directly. Normalizaiton happens
// afterwards.

mod clipped_noramlizer;
mod worker;

use r2l_core::{
    buffers::buffer::{TrajectoryBuffer, TrajectoryView},
    env::Env,
    models::Actor,
    on_policy::algorithm::Sampler,
    tensor::RunningMeanTensor,
};

use crate::normalized::{clipped_noramlizer::ClippedNormalizer, worker::WorkerPool};

const EPS: f32 = 1e-8;

struct Coordinator<E: Env<Tensor: RunningMeanTensor>> {
    pool: WorkerPool<E>,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    reward_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    // Here there is no need to have each thread own the buffer
    buffers: Vec<TrajectoryBuffer<E::Tensor>>,
}

impl<E: Env<Tensor: RunningMeanTensor>> Coordinator<E> {
    fn step_inner(&mut self) -> (Vec<E::Tensor>, Vec<f32>, Vec<bool>) {
        // something like this is implemented
        let (obs, rewards, dones) = self.pool.step();
        let obs = if let Some(obs_normalizer) = self.obs_normalizer.as_mut() {
            // TODO: update should be able to have a vector passed in. Will have to check what this
            // means in reality.
            // obs_normalizer.rm.update(&obs);
            // TODO: In sb3, this is normalized either way
            obs_normalizer.normalize(obs)
        } else {
            obs
        };
        (obs, rewards, dones)
    }

    fn step(&mut self) {}
}

impl<E: Env<Tensor: RunningMeanTensor>> Sampler for Coordinator<E> {
    type Tensor = E::Tensor;

    fn collect_rollouts<A: Actor<Tensor = Self::Tensor> + Clone>(&mut self, actor: A) {
        todo!()
    }

    fn trajectory_views<'a>(&'a mut self) -> impl AsRef<[TrajectoryView<'a, Self::Tensor>]> {
        self.buffers
            .iter()
            .map(|b| b.to_trajectory_view())
            .collect::<Vec<_>>()
    }

    fn shutdown(&mut self) {
        todo!()
    }
}
