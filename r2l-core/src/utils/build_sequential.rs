use candle_core::Result;
use candle_nn::{Activation, VarBuilder};

use crate::thread_safe_sequential::{
    ActivationLayer, LinearLayer, ThreadSafeLayer, ThreadSafeSequential,
};

// TODO: we do not need input dim and last_dim returned
pub fn build_sequential(
    input_dim: usize,
    layers: &[usize],
    vb: &VarBuilder,
    prefix: &str,
) -> Result<(ThreadSafeSequential, usize)> {
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
    Ok((nn, last_dim))
}
