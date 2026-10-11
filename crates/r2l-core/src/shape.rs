//! Dimensions shared by tensors, spaces, and network contracts.

use std::ops::Deref;

use serde::{Deserialize, Serialize};

/// Ordered dimensions. An empty dimension list denotes a scalar; a zero dimension
/// denotes an empty tensor. Every left-to-right partial product fits in `usize`.
/// Other validity requirements are checked by each operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Creates a shape from dimensions in axis order.
    ///
    /// # Panics
    ///
    /// Panics if any left-to-right partial product overflows `usize`, even if a
    /// later dimension is zero.
    #[must_use]
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self::try_new(dims.into()).expect("invalid shape")
    }

    fn try_new(dims: Vec<usize>) -> Result<Self, &'static str> {
        dims.iter()
            .try_fold(1_usize, |size, &dim| size.checked_mul(dim))
            .ok_or("shape element count exceeds usize")?;
        Ok(Self(dims))
    }

    /// Returns dimensions in axis order.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Returns the number of axes, including axes with zero length.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns the element count.
    /// Scalars have one element; shapes containing a zero dimension have none.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }
}

impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let dims = Vec::<usize>::deserialize(deserializer)?;
        Self::try_new(dims).map_err(serde::de::Error::custom)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl<const D: usize> From<[usize; D]> for Shape {
    fn from(dims: [usize; D]) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims)
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        shape.clone()
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        self.dims()
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.dims()
    }
}
