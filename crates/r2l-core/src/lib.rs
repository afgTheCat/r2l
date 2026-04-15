pub mod buffers;
pub mod env;
pub mod error;
pub mod models;
pub mod on_policy;
pub mod rng;
pub mod tensor;
pub mod utils;

use anyhow::Result;

/// A learning algorithm. Currently only `OnPolicyAlgorithm` implements this trait, but in the
/// future an off policy alternative is also going to implement it.
/// TODO: would a replay/verify method be useful here?
pub trait Algorithm {
    fn train(&mut self) -> Result<()>;
}
