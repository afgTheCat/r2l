use bincode::{
    Decode, Encode,
    de::{self},
    error::{DecodeError, EncodeError},
};
use candle_core::{Device, Error, Result, Tensor, safetensors::BufferedSafetensors};
use candle_nn::{Activation, Linear, Module, VarBuilder, linear};
use either::Either;
use safetensors::serialize;

#[derive(Debug, Clone)]
pub struct LinearLayer {
    layer: Linear,
    weight_name: String,
    bias_name: String,
}

impl LinearLayer {
    pub fn new(in_dim: usize, out_dim: usize, vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let weight_name = format!("{prefix}_weight");
        let bias_name = format!("{prefix}_bias");
        let layer = linear(in_dim, out_dim, vb.pp(prefix))?;
        Ok(Self {
            layer,
            weight_name,
            bias_name,
        })
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let data = [
            (&self.weight_name, self.layer.weight()),
            (&self.bias_name, self.layer.bias().unwrap()), // TODO: maybe error handle here?
        ];
        serialize(data, &None).map_err(Error::wrap)
    }
}

impl Encode for LinearLayer {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        let writer_config = bincode::config::standard();
        bincode::encode_into_writer(
            self.serialize()
                .map_err(|err| EncodeError::OtherString(err.to_string()))?,
            &mut encoder.writer(),
            writer_config,
        )?;
        self.weight_name.encode(encoder)?;
        self.bias_name.encode(encoder)
    }
}

impl Decode<()> for LinearLayer {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, bincode::error::DecodeError> {
        let encoded_layer: Vec<u8> = Vec::decode(decoder)?;
        let weight_name: String = String::decode(decoder)?;
        let bias_name = String::decode(decoder)?;
        let buffered_safetensors = BufferedSafetensors::new(encoded_layer)
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let weight_tensor = buffered_safetensors
            .load(&weight_name, &Device::Cpu)
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let bias_tensor = buffered_safetensors
            .load(&bias_name, &Device::Cpu)
            .map_err(|err| DecodeError::OtherString(err.to_string()))?;
        let layer = Linear::new(weight_tensor, Some(bias_tensor));
        Ok(Self {
            layer,
            weight_name,
            bias_name,
        })
    }
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.layer.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct ActivationLayer(pub Activation);

impl Encode for ActivationLayer {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        let writer_config = bincode::config::standard();
        bincode::encode_into_writer(
            bincode::serde::encode_to_vec(self.0, writer_config)?,
            &mut encoder.writer(),
            writer_config,
        )
    }
}

impl Decode<()> for ActivationLayer {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, DecodeError> {
        let reader_config = bincode::config::standard();
        let acvitavtion_vec: Vec<u8> = Vec::decode(decoder)?;
        let (activation, _) = bincode::serde::decode_from_slice(&acvitavtion_vec, reader_config)?;
        // TODO: for some reason the one below does not work
        // let activation: Activation =
        //     bincode::serde::decode_from_reader(&mut decoder.reader(), reader_config).unwrap();
        Ok(ActivationLayer(activation))
    }
}

impl Module for ActivationLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct ThreadSafeLayer(pub Either<LinearLayer, ActivationLayer>);

impl Encode for ThreadSafeLayer {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), bincode::error::EncodeError> {
        match &self.0 {
            Either::Left(l) => {
                0u32.encode(encoder)?;
                l.encode(encoder)
            }
            Either::Right(r) => {
                1u32.encode(encoder)?;
                r.encode(encoder)
            }
        }
    }
}

impl Decode<()> for ThreadSafeLayer {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, DecodeError> {
        let enum_type = u32::decode(decoder)?;
        match enum_type {
            0 => {
                let layer = LinearLayer::decode(decoder)?;
                Ok(Self(Either::Left(layer)))
            }
            1 => {
                let activation: ActivationLayer = ActivationLayer::decode(decoder)?;
                Ok(Self(Either::Right(activation)))
            }
            _ => unreachable!(),
        }
    }
}

impl ThreadSafeLayer {
    pub fn linear(linear: LinearLayer) -> Self {
        Self(Either::Left(linear))
    }

    pub fn activation(activation: ActivationLayer) -> Self {
        Self(Either::Right(activation))
    }
}

impl Module for ThreadSafeLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.0 {
            Either::Left(linear) => linear.forward(xs),
            Either::Right(activation) => activation.forward(xs),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct ThreadSafeSequential {
    layers: Vec<ThreadSafeLayer>,
}

impl Module for ThreadSafeSequential {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        Ok(xs)
    }
}

pub fn build_sequential(
    input_dim: usize,
    layers: &[usize],
    vb: &VarBuilder,
    prefix: &str,
) -> Result<ThreadSafeSequential> {
    let mut last_dim = input_dim;
    let mut nn = ThreadSafeSequential::default();
    let num_layers = layers.len();
    for (layer_idx, layer_size) in layers.iter().enumerate() {
        let layer_pp = format!("{prefix}{layer_idx}");
        if layer_idx == num_layers - 1 {
            let layer = LinearLayer::new(last_dim, *layer_size, vb, &layer_pp)?;
            nn = nn.add(ThreadSafeLayer::linear(layer))
        } else {
            let lin_layer = LinearLayer::new(last_dim, *layer_size, vb, &layer_pp)?;
            nn = nn
                .add(ThreadSafeLayer::linear(lin_layer))
                .add(ThreadSafeLayer::activation(ActivationLayer(
                    Activation::Relu,
                )));
        }
        last_dim = *layer_size;
    }
    Ok(nn)
}

impl ThreadSafeSequential {
    pub fn add(mut self, layer: ThreadSafeLayer) -> Self {
        self.layers.push(layer);
        self
    }
}

impl Encode for ThreadSafeSequential {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> std::result::Result<(), EncodeError> {
        self.layers.encode(encoder)
    }
}

impl Decode<()> for ThreadSafeSequential {
    fn decode<D: de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> std::result::Result<Self, DecodeError> {
        let layers = Vec::decode(decoder)?;
        Ok(Self { layers })
    }
}

// TODO: we need to make sure that all serializations work always as intended. This needs to be
// turned into a complete package at one point
#[cfg(test)]
mod test {
    use super::{LinearLayer, ThreadSafeLayer, ThreadSafeSequential};
    use crate::thread_safe_sequential::{ActivationLayer, build_sequential};
    use candle_core::{DType, Device, Error, Result};
    use candle_nn::{Activation, VarBuilder, VarMap};

    #[test]
    fn serialize_tss() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let tss = build_sequential(10, &vec![10, 10], &vb, "value")?;
        let config = bincode::config::standard();
        let bin_thing = bincode::encode_to_vec(tss, config).map_err(Error::wrap)?;
        let (decoded, _): (ThreadSafeSequential, usize) =
            bincode::decode_from_slice(&bin_thing, config).map_err(Error::wrap)?;
        println!("{decoded:#?}");
        Ok(())
    }

    #[test]
    fn serialize_tsl() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let layer = LinearLayer::new(10, 10, &vb, "value")?;
        let l1 = ThreadSafeLayer::linear(layer);
        let config = bincode::config::standard();
        let bin_thing = bincode::encode_to_vec(l1, config).map_err(Error::wrap)?;
        let (decoded, _): (ThreadSafeLayer, usize) =
            bincode::decode_from_slice(&bin_thing, config).map_err(Error::wrap)?;
        println!("{decoded:#?}");

        let activation = ActivationLayer(Activation::Relu);
        let l2 = ThreadSafeLayer::activation(activation);
        let bin_thing = bincode::encode_to_vec(l2, config).map_err(Error::wrap)?;
        let (decoded, _): (ThreadSafeLayer, usize) =
            bincode::decode_from_slice(&bin_thing, config).map_err(Error::wrap)?;
        println!("{decoded:#?}");
        Ok(())
    }
}
