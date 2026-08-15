use r2l_core::{
    env::normalizer::{ClippedNormalizer, ClippedNormalizerSnapshot},
    error::TensorError,
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) struct NormalizerBuilder(ClippedNormalizerSnapshot);

impl NormalizerBuilder {
    pub(crate) fn from_normalizer<T: R2lTensor>(
        normalizer: &ClippedNormalizer<T>,
    ) -> Result<Self, TensorError> {
        Ok(Self(normalizer.snapshot()?))
    }

    pub(crate) fn into_normalizer<T: R2lTensor>(self) -> Result<ClippedNormalizer<T>, TensorError> {
        self.0.into_normalizer()
    }
}
