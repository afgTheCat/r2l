pub mod agents;
pub mod distributions;
pub mod env;
pub mod error;
// TODO: readd this
pub mod on_policy_algorithm;
pub mod policies;
pub mod rng;
pub mod sampler;
pub mod sampler2;
pub mod sampler3;
pub mod tensor;
pub mod utils;

use anyhow::Result;

/// A learning algorithm. Currently only `OnPolicyAlgorithm` implements this trait, but in the
/// future an off policy alternative is also going to implement it.
/// TODO: would a replay/verify method be useful here?
pub trait Algorithm {
    fn train(&mut self) -> Result<()>;
}
