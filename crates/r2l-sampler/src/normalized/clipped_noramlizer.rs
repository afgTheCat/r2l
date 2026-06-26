// I think we should move this to a different crate eventually
use itertools::izip;
use r2l_core::{running_mean::RunningMeanStd, tensor::R2lTensor};

const EPS: f32 = 1e-8;

pub struct ClippedNormalizer<T: R2lTensor> {
    rm: RunningMeanStd<T>,
    clip: f32,
}

impl<T: R2lTensor> ClippedNormalizer<T> {
    pub fn update(&mut self, obs: &[T]) {
        self.rm.update(obs);
    }

    pub fn normalize(&self, obs: &[T]) -> Vec<T> {
        let mean = self.rm.mean.to_vec();
        let var = self.rm.var.to_vec();
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

    // updates the rms + returns the noremalized observation
    pub fn update_and_normalize(&self, obs: &[T]) -> Vec<T> {
        self.update(obs);
        self.normalize(obs)
    }
}
