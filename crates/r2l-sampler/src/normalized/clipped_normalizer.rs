// I think we should move this to a different crate eventually
use itertools::izip;
use r2l_core::{
    running_mean::{RunningMeanStd, RunningMeanStdVec},
    tensor::R2lTensor,
};

const EPS: f32 = 1e-8;

pub struct ClippedNormalizer<T: R2lTensor> {
    rm: RunningMeanStd<T>,
    clip: Option<f32>,
}

impl<T: R2lTensor> ClippedNormalizer<T> {
    pub fn new(clip: Option<f32>, shape: Vec<usize>) -> Self {
        let rm = RunningMeanStd::new(shape);
        Self { clip, rm }
    }

    fn update(&mut self, obs: &[T]) {
        self.rm.update(obs);
    }

    fn normalize_in_place(&self, obs: &mut [T]) {
        let mean = self.rm.mean.to_vec();
        let var = self.rm.var.to_vec();
        for obs in obs {
            let (data, shape) = obs.to_vec_and_shape();
            let normalized = izip!(data, &mean, &var)
                .map(|(val, mean, var)| {
                    let normalized = (val - mean) / (var + EPS).sqrt();
                    if let Some(clip) = self.clip {
                        normalized.clamp(-clip, clip)
                    } else {
                        normalized
                    }
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

pub struct ClippedNormalizerVec {
    rm: RunningMeanStdVec,
}

impl ClippedNormalizerVec {
    pub fn new(shape: Vec<usize>) -> Self {
        let rm = RunningMeanStdVec::new(shape);
        Self { rm }
    }

    pub fn update_scalars(&mut self, values: &[f32]) {
        self.rm.update_from_scalars(values);
    }

    pub fn scalar_mean_var(&self) -> (f32, f32) {
        self.rm.scalar_mean_var()
    }
}
