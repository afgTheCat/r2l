use super::Distribution;
use crate::thread_safe_sequential::{ThreadSafeSequential, build_sequential};
use bincode::{
    Decode, Encode,
    error::{DecodeError, EncodeError},
};
use candle_core::{Device, Result, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{Module, VarBuilder};
use safetensors::serialize;
use std::f32;

// TODO: we may want to resample the noise better than it is now
pub struct DiagGaussianDistribution {
    pub noise: Tensor,
    pub mu_net: ThreadSafeSequential,
    pub log_std: Tensor,
}

impl Encode for DiagGaussianDistribution {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        let writer_config = bincode::config::standard();
        self.mu_net.encode(encoder)?;
        let data = [("noise", &self.noise), ("log_std", &self.log_std)];
        bincode::encode_into_writer(
            serialize(data, &None).map_err(|err| EncodeError::OtherString(err.to_string()))?,
            &mut encoder.writer(),
            writer_config,
        )
    }
}

impl Decode<()> for DiagGaussianDistribution {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        let mu_net: ThreadSafeSequential = ThreadSafeSequential::decode(decoder)?;
        let encoded_data: Vec<u8> = Vec::decode(decoder)?;
        let buffered_safetensors = BufferedSafetensors::new(encoded_data)
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let noise = buffered_safetensors
            .load("noise", &Device::Cpu)
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let log_std = buffered_safetensors
            .load("log_std", &Device::Cpu)
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        Ok(Self {
            noise,
            mu_net,
            log_std,
        })
    }
}

impl DiagGaussianDistribution {
    pub fn new(noise: Tensor, mu_net: ThreadSafeSequential, log_std: Tensor) -> Self {
        Self {
            noise,
            mu_net,
            log_std,
        }
    }

    pub fn build(
        input_dim: usize,
        layers: &[usize],
        vb: &VarBuilder,
        log_std: Tensor,
        prefix: &str,
    ) -> Result<Self> {
        let mu_net = build_sequential(input_dim, layers, vb, prefix)?;
        let noise = Tensor::randn(0f32, 1., log_std.shape(), log_std.device()).unwrap();
        Ok(Self {
            log_std,
            mu_net,
            noise,
        })
    }
}

impl Distribution for DiagGaussianDistribution {
    fn get_action(&self, observation: &Tensor) -> Result<(Tensor, Tensor)> {
        let mu = self
            .mu_net
            .forward(&observation.unsqueeze(0)?)?
            .squeeze(0)?;
        let std = self.log_std.exp()?.unsqueeze(0)?;
        let noise = Tensor::randn(0f32, 1., self.log_std.shape(), self.log_std.device())?;
        let action = (mu + std.mul(&noise.unsqueeze(0)?)?)?.squeeze(0)?.detach();
        let logp = self.log_probs(observation, &action.unsqueeze(0)?)?.detach();
        Ok((action, logp))
    }

    fn log_probs(&self, states: &Tensor, actions: &Tensor) -> Result<Tensor> {
        let mu = self.mu_net.forward(states)?;
        let std = self.log_std.exp()?.broadcast_as(mu.shape())?;
        let var = std.sqr()?;
        let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
        let log_sqrt_2pi = Tensor::full(log_sqrt_2pi, mu.shape(), mu.device())?;
        let log_probs = ((((actions - &mu)?.sqr()? / (2. * var)?)?.neg()?
            - &self.log_std.broadcast_as(mu.shape())?)?
            - log_sqrt_2pi)?;
        log_probs.sum(1)
    }

    fn entropy(&self) -> Result<Tensor> {
        let log_2pi_plus_1_div_2 = Tensor::full(
            0.5 * ((2. * f32::consts::PI).ln() + 1.),
            self.log_std.shape(),
            self.log_std.device(),
        )?;
        log_2pi_plus_1_div_2.add(&self.log_std)?.sum_all()
    }

    fn std(&self) -> Result<f32> {
        self.log_std.exp()?.mean_all()?.to_scalar::<f32>()
    }

    fn resample_noise(&mut self) -> Result<()> {
        self.noise = Tensor::randn(0f32, 1., self.noise.shape(), self.noise.device())?;
        Ok(())
    }
}
