use r2l_core::{
    env::normalizer::{ClippedNormalizer, ClippedNormalizerSnapshot},
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) struct NormalizerBuilder(ClippedNormalizerSnapshot);

impl NormalizerBuilder {
    pub(crate) fn from_normalizer<T: R2lTensor>(normalizer: &ClippedNormalizer<T>) -> Self {
        Self(normalizer.snapshot())
    }

    pub(crate) fn into_normalizer<T: R2lTensor>(self) -> ClippedNormalizer<T> {
        self.0.into_normalizer()
    }
}
