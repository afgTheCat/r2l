//! Backend configuration and runtime devices for training and inference.

use burn::{
    backend::{Autodiff, Flex},
    tensor::Device as BurnDevice,
};
use candle_core::{Device, DeviceLocation};
use r2l_core::error::Error;
use serde::{Deserialize, Serialize};

/// Serializable Candle device kind and ordinal, without runtime device state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum CandleDeviceConfig {
    Cpu,
    Cuda { ordinal: usize },
    Metal { ordinal: usize },
}

impl CandleDeviceConfig {
    pub(crate) fn from_device(device: &Device) -> Self {
        match device.location() {
            DeviceLocation::Cpu => Self::Cpu,
            DeviceLocation::Cuda { gpu_id } => Self::Cuda { ordinal: gpu_id },
            DeviceLocation::Metal { gpu_id } => Self::Metal { ordinal: gpu_id },
        }
    }

    fn build(self) -> Result<Device, Error> {
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
pub(crate) struct BurnBackendConfig;

/// Serializable backend selection shared by training and inference recipes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum BackendConfig {
    Candle(CandleDeviceConfig),
    Burn(BurnBackendConfig),
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::Candle(CandleDeviceConfig::Cpu)
    }
}

impl BackendConfig {
    /// Creates the runtime backend, reusing a supplied Candle device when present.
    pub(crate) fn build(self, candle_device: Option<Device>) -> Result<Backend, Error> {
        Ok(match self {
            Self::Candle(config) => Backend::Candle(match candle_device {
                Some(device) => device,
                None => config.build()?,
            }),
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
