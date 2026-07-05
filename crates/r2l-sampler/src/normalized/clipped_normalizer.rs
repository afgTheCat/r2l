// I think we should move this to a different crate eventually
use itertools::izip;
use r2l_core::{running_mean::RunningMeanStd, tensor::R2lTensor};

const EPS: f32 = 1e-8;

#[derive(Clone)]
pub struct ClippedNormalizer<T: R2lTensor> {
    rm: RunningMeanStd<T>,
    clip: f32,
}

impl<T: R2lTensor> ClippedNormalizer<T> {
    pub fn new(clip: f32, shape: Vec<usize>) -> Self {
        let rm = RunningMeanStd::new(shape);
        Self { clip, rm }
    }

    pub fn update(&mut self, obs: &[T]) {
        self.rm.update(obs);
    }

    pub fn normalize_in_place(&self, obs: &mut [T]) {
        let mean = self.rm.mean.to_vec();
        let var = self.rm.var.to_vec();
        for obs in obs {
            let (data, shape) = obs.to_vec_and_shape();
            let normalized = izip!(data, &mean, &var)
                .map(|(val, mean, var)| {
                    ((val - mean) / (var + EPS).sqrt()).clamp(-self.clip, self.clip)
                })
                .collect();
            *obs = T::from_vec_and_shape(normalized, shape);
        }
    }

    pub fn update_and_normalize_in_place(&mut self, obs: &mut [T]) {
        self.update(obs);
        self.normalize_in_place(obs);
    }
}
