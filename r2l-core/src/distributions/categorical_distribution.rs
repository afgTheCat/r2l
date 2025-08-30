use super::Distribution;
use crate::thread_safe_sequential::{ThreadSafeSequential, build_sequential};
use bincode::{Decode, Encode};
use candle_core::{Device, Error, Result, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::log_softmax;
use candle_nn::{Module, ops::softmax};
use rand::distr::Distribution as RandDistributiion;
use rand::distr::weighted::WeightedIndex;

#[derive(Clone, Debug)]
pub struct CategoricalDistribution {
    action_size: usize,
    logits: ThreadSafeSequential,
    device: Device,
}

impl Decode<()> for CategoricalDistribution {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        // TODO: We need to implement this
        let action_size = usize::decode(decoder)?;
        let logits: ThreadSafeSequential = ThreadSafeSequential::decode(decoder)?;
        let device_type = u32::decode(decoder)?;
        let device = match device_type {
            0 => Device::Cpu,
            _ => todo!(),
        };
        Ok(Self {
            logits,
            action_size,
            device,
        })
    }
}

impl Encode for CategoricalDistribution {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        self.action_size.encode(encoder)?;
        self.logits.encode(encoder)?;
        match &self.device {
            Device::Cpu => 0u32.encode(encoder),
            Device::Cuda(_) => 1u32.encode(encoder),
            _ => todo!(),
        }
    }
}

impl CategoricalDistribution {
    pub fn new(action_size: usize, logits: ThreadSafeSequential, device: Device) -> Self {
        Self {
            action_size,
            logits,
            device,
        }
    }

    pub fn build(
        input_dim: usize,
        action_size: usize,
        layers: &[usize],
        vb: &VarBuilder,
        device: Device,
        prefix: &str,
    ) -> Result<Self> {
        let logits = build_sequential(input_dim, layers, vb, prefix)?;
        Ok(Self {
            action_size,
            logits,
            device,
        })
    }
}

impl Distribution for CategoricalDistribution {
    type Tensor = Tensor;

    fn get_action(&self, observation: Tensor) -> Result<Tensor> {
        assert!(
            observation.rank() == 1,
            "Observation should be a flattened tensor"
        );
        let observation = observation.unsqueeze(0)?;
        let logits = self.logits.forward(&observation)?;
        let action_probs: Vec<f32> = softmax(&logits, 1)?.squeeze(0)?.to_vec1()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = distribution.sample(&mut rand::rng());
        // TODO: there is a one_hot function in candle. Should we use it?
        let mut action_mask: Vec<f32> = vec![0.0; self.action_size];
        action_mask[action] = 1.;
        let action = Tensor::from_vec(action_mask, self.action_size, &self.device)?.detach();
        Ok(action)
    }

    fn log_probs(&self, states: Tensor, actions: Tensor) -> Result<Tensor> {
        let logits = self.logits.forward(&states)?;
        let log_probs = log_softmax(&logits, 1)?;
        actions.mul(&log_probs)?.sum(1)
    }

    fn std(&self) -> Result<f32> {
        todo!()
    }

    fn entropy(&self) -> Result<Tensor> {
        todo!()
    }
}
