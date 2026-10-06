//! Backend types for training and inference.

use burn::backend::{Autodiff, Flex};
use candle_core::{Device, DeviceLocation};
use r2l_core::error::Error;
use serde::{Deserialize, Serialize, de::Error as _};

/// Serializable Candle device kind and ordinal, without runtime device state.
#[derive(Serialize, Deserialize)]
enum CandleDeviceConfig {
    Cpu,
    Cuda { ordinal: usize },
    Metal { ordinal: usize },
}

/// Candle device shared by training construction and inference artifact loading.
#[derive(Debug, Clone)]
pub(crate) struct CandleBackend {
    /// Device on which networks and their parameters are allocated.
    pub(crate) device: Device,
}

impl CandleBackend {
    /// Seeds the device RNG on accelerator devices.
    ///
    /// CPU devices are left unchanged; r2l seeds its own RNG separately.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed passed to the accelerator's random number generator.
    ///
    /// # Errors
    ///
    /// Returns an error if the device cannot set its RNG seed.
    pub(crate) fn seed(&self, seed: u64) -> Result<(), Error> {
        if !matches!(&self.device, Device::Cpu) {
            self.device.set_seed(seed).map_err(Error::wrap)?;
        }
        Ok(())
    }
}

impl Serialize for CandleBackend {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let device = match self.device.location() {
            DeviceLocation::Cpu => CandleDeviceConfig::Cpu,
            DeviceLocation::Cuda { gpu_id } => CandleDeviceConfig::Cuda { ordinal: gpu_id },
            DeviceLocation::Metal { gpu_id } => CandleDeviceConfig::Metal { ordinal: gpu_id },
        };
        device.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CandleBackend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let device = match CandleDeviceConfig::deserialize(deserializer)? {
            CandleDeviceConfig::Cpu => Device::Cpu,
            CandleDeviceConfig::Cuda { ordinal } => {
                Device::new_cuda(ordinal).map_err(D::Error::custom)?
            }
            CandleDeviceConfig::Metal { ordinal } => {
                Device::new_metal(ordinal).map_err(D::Error::custom)?
            }
        };
        Ok(Self { device })
    }
}

/// Default CPU backend used by Burn training builders, with automatic differentiation.
pub type BurnBackend = Autodiff<Flex>;

/// Configuration marker for the default Burn Flex backend.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub(crate) struct BurnBackendConfig;

/// Backend configuration shared by training builders and saved inference recipes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Backend {
    /// Candle backend configuration.
    Candle(CandleBackend),
    /// Default Burn backend configuration.
    Burn(BurnBackendConfig),
}
