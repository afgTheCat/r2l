// TODO: It's debatable if we gonna need this

use crate::utils::rollout_buffer::RolloutBuffer;
use bincode::{
    BorrowDecode, Decode, Encode,
    error::{DecodeError, EncodeError},
};
use candle_core::{Device, Result, Tensor};

impl Encode for RolloutBuffer<Tensor> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        let writer_config = bincode::config::standard();
        let states = self
            .states
            .iter()
            .map(|t| t.to_vec1::<f32>())
            .collect::<Result<Vec<_>>>()
            .map_err(|err| EncodeError::OtherString(err.to_string()))?;
        bincode::encode_into_writer(&states, &mut encoder.writer(), writer_config)?;
        let actions = self
            .actions
            .iter()
            .map(|t| t.to_vec1::<f32>())
            .collect::<Result<Vec<_>>>()
            .map_err(|err| EncodeError::OtherString(err.to_string()))?;
        bincode::encode_into_writer(&actions, &mut encoder.writer(), writer_config)?;
        bincode::encode_into_writer(&self.rewards, &mut encoder.writer(), writer_config)?;
        bincode::encode_into_writer(&self.dones, &mut encoder.writer(), writer_config)?;
        match &self.last_state {
            None => 0u32.encode(encoder)?,
            Some(last_state) => {
                1u32.encode(encoder)?;
                let last_state = last_state
                    .to_vec1::<f32>()
                    .map_err(|err| EncodeError::OtherString(err.to_string()))?;
                bincode::encode_into_writer(&last_state, &mut encoder.writer(), writer_config)?;
            }
        };
        Ok(())
    }
}

// TODO: we need to add the device into the context here to be able to use Tensors on the GPU
impl<C> Decode<C> for RolloutBuffer<Tensor> {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        let states_raw: Vec<Vec<f32>> = Vec::decode(decoder)?;
        let states = states_raw
            .into_iter()
            .map(|v| Tensor::from_slice(&v, v.len(), &Device::Cpu))
            .collect::<Result<Vec<_>>>()
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let actions_raw: Vec<Vec<f32>> = Vec::decode(decoder)?;
        let actions = actions_raw
            .into_iter()
            .map(|v| Tensor::from_slice(&v, v.len(), &Device::Cpu))
            .collect::<Result<Vec<_>>>()
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let rewards: Vec<f32> = Vec::decode(decoder)?;
        let dones: Vec<bool> = Vec::decode(decoder)?;
        let last_state_type = u32::decode(decoder)?;
        let last_state: Option<Tensor> = match last_state_type {
            0 => None,
            1 => {
                let last_state: Vec<f32> = Vec::decode(decoder)?;
                let last_state = Tensor::from_slice(&last_state, last_state.len(), &Device::Cpu)
                    .map_err(|err| DecodeError::OtherString(err.to_string()))?;
                Some(last_state)
            }
            _ => unreachable!(),
        };
        Ok(Self {
            states,
            actions,
            rewards,
            dones,
            last_state,
        })
    }
}

impl<'de, C> BorrowDecode<'de, C> for RolloutBuffer<Tensor> {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = C>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        RolloutBuffer::decode(decoder)
    }
}
