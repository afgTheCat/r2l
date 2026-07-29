use r2l_core::{
    env::normalizer::{ClippedNormalizer, NormalizerMode},
    running_mean::RunningMeanStd,
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NormalizerBuilder {
    normalizer_mode: NormalizerMode,
    obs_shape: Vec<usize>,
    mean: Vec<f32>,
    var: Vec<f32>,
    count: f32,
    clip: f32,
}

impl NormalizerBuilder {
    pub fn from_normalizer<T: R2lTensor>(normalizer: ClippedNormalizer<T>) -> Self {
        let ClippedNormalizer {
            normalizer_mode,
            inner,
        } = normalizer;
        let inner = inner.lock().unwrap();
        let RunningMeanStd { mean, var, count } = &inner.rm;
        let (mean, obs_shape) = mean.to_vec_and_shape();
        let var = var.to_vec();
        NormalizerBuilder {
            normalizer_mode,
            mean,
            var,
            obs_shape,
            count: *count,
            clip: inner.clip,
        }
    }

    pub fn into_normalizer<T: R2lTensor>(self) -> ClippedNormalizer<T> {
        let mean = T::from_vec_and_shape(self.mean, self.obs_shape.clone());
        let var = T::from_vec_and_shape(self.var, self.obs_shape);
        let rm = RunningMeanStd::build(mean, var, self.count);
        ClippedNormalizer::new(self.normalizer_mode, rm, self.clip)
    }
}
