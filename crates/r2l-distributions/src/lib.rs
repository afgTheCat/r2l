//! Experimental policies over backend-independent tensors and network contracts.

pub mod distributions;
pub mod networks;

pub use distributions::categorical::{Categorical, Policy2};
pub use networks::Network;
