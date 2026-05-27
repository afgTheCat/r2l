// fun programming

use itertools::izip;
use r2l_core::{env::Env, models::Actor, running_mean::RunningMeanStd2, tensor::RunningMeanTensor};

const EPS: f32 = 1e-8;

struct Worker<E: Env<Tensor: RunningMeanTensor>> {
    env: E,
}

impl<E: Env<Tensor: RunningMeanTensor>> Worker<E> {
    fn step(&mut self, action: E::Tensor) -> (E::Tensor, f32, bool) {
        todo!()
    }
}

enum WorkerPool<E: Env<Tensor: RunningMeanTensor>> {
    VecCoord(Vec<Worker<E>>),
}

impl<E: Env<Tensor: RunningMeanTensor>> WorkerPool<E> {
    fn step(&mut self) -> (Vec<E::Tensor>, Vec<f32>, Vec<bool>) {
        todo!()
    }
}

struct ClippedNormalizer<T: RunningMeanTensor> {
    rm: RunningMeanStd2<T>,
    clip: f32,
}

impl<T: RunningMeanTensor> ClippedNormalizer<T> {
    fn normalize(&self, obs: Vec<T>) -> Vec<T> {
        let (mean, _) = self.rm.mean.to_vec_and_shape();
        let (var, _) = self.rm.var.to_vec_and_shape();
        obs.into_iter()
            .map(|obs| {
                let (data, shape) = obs.to_vec_and_shape();
                let normalized = izip!(data, &mean, &var)
                    .map(|(val, mean, var)| {
                        ((val - *mean) / (*var + EPS).sqrt()).clamp(-self.clip, self.clip)
                    })
                    .collect();
                T::from_vec_and_shape(normalized, shape)
            })
            .collect()
    }
}

struct Coordinator<E: Env<Tensor: RunningMeanTensor>> {
    pool: WorkerPool<E>,
    obs_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    reward_normalizer: Option<ClippedNormalizer<E::Tensor>>,
    // This is only for backing shit up.
    old_obs: Vec<E::Tensor>,
    old_rewards: Vec<f32>,
    policy: Box<dyn Actor<Tensor = E::Tensor>>,
}

impl<E: Env<Tensor: RunningMeanTensor>> Coordinator<E> {
    fn step_inner(&mut self) -> (Vec<E::Tensor>, Vec<f32>, Vec<bool>) {
        // something like this is implemented
        let (obs, rewards, dones) = self.pool.step();
        // This is not normalied, why do we need this?
        self.old_obs = obs.clone();
        self.old_rewards = rewards.clone();
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
