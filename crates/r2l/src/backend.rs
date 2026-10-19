//! Backend configuration and runtime devices for training and inference.

use burn::{
    backend::{Autodiff, Flex},
    tensor::Device as BurnDevice,
};
use candle_core::Device;
use r2l_core::error::Error;
use serde::{Deserialize, Serialize};

/// Serializable Candle device kind and ordinal, without runtime device state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CandleDevice {
    /// Uses the CPU.
    Cpu,
    /// Uses a CUDA device.
    Cuda {
        /// Zero-based CUDA device ordinal.
        ordinal: usize,
    },
    /// Uses a Metal device.
    Metal {
        /// Zero-based Metal device ordinal.
        ordinal: usize,
    },
}

impl CandleDevice {
    pub(crate) fn build(self) -> Result<Device, Error> {
        Ok(match self {
            Self::Cpu => Device::Cpu,
            Self::Cuda { ordinal } => Device::new_cuda(ordinal)?,
            Self::Metal { ordinal } => Device::new_metal(ordinal)?,
        })
    }
}

/// Default CPU backend used by Burn training builders, with automatic differentiation.
pub type BurnBackend = Autodiff<Flex>;

/// Configuration marker for the default Burn Flex backend.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BurnBackendConfig;

/// Serializable backend selection shared by training and inference recipes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BackendConfig {
    Candle(CandleDevice),
    Burn(BurnBackendConfig),
}

impl From<CandleDevice> for BackendConfig {
    fn from(value: CandleDevice) -> Self {
        Self::Candle(value)
    }
}

impl From<BurnBackendConfig> for BackendConfig {
    fn from(value: BurnBackendConfig) -> Self {
        Self::Burn(value)
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::Candle(CandleDevice::Cpu)
    }
}

impl BackendConfig {
    /// Creates the runtime backend from its configuration.
    pub(crate) fn build(self) -> Result<Backend, Error> {
        Ok(match self {
            Self::Candle(config) => Backend::Candle(config.build()?),
            Self::Burn(_) => Backend::Burn,
        })
    }
}

/// Runtime backend created after configuration has been loaded.
pub(crate) enum Backend {
    Candle(Device),
    Burn,
}

impl Backend {
    /// Seeds r2l and the selected learner backend before constructing networks or samplers.
    ///
    /// # Arguments
    ///
    /// * `seed` - Initial seed for action sampling, environment workers, and model parameters.
    pub(crate) fn seed(&self, seed: u64) -> Result<(), Error> {
        r2l_core::rng::set_seed(seed);
        match self {
            Self::Candle(device) => {
                if !matches!(device, Device::Cpu) {
                    device.set_seed(seed)?;
                }
            }
            Self::Burn => BurnDevice::flex().seed(seed),
        }
        Ok(())
    }
}
