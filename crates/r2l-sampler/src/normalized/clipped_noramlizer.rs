// I think we should move this to a different crate eventually
use itertools::izip;
use r2l_core::{running_mean::RunningMeanStd2, tensor::RunningMeanTensor};

const EPS: f32 = 1e-8;

pub struct ClippedNormalizer<T: RunningMeanTensor> {
    rm: RunningMeanStd2<T>,
    clip: f32,
}

impl<T: RunningMeanTensor> ClippedNormalizer<T> {
    pub fn normalize(&self, obs: Vec<T>) -> Vec<T> {
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
