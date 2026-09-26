use r2l_core::{
    env::normalizer::{Normalizer, NormalizerSnapshot},
    error::TensorError,
    tensor::R2lTensor,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) struct NormalizerBuilder(NormalizerSnapshot);

impl NormalizerBuilder {
    pub(crate) fn from_normalizer<T: R2lTensor>(
        normalizer: &Normalizer<T>,
    ) -> Result<Self, TensorError> {
        Ok(Self(normalizer.snapshot()?))
    }

    pub(crate) fn into_normalizer<T: R2lTensor>(self) -> Result<Normalizer<T>, TensorError> {
        self.0.into_normalizer()
    }
}
