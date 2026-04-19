use burn::backend::{Autodiff, NdArray};

// builders + hooks + higher level helpers
pub mod agents;
pub mod builders;
pub mod hooks;
pub mod utils;

pub type BurnBackend = Autodiff<NdArray>;
