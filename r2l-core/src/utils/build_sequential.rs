use candle_core::Result;
use candle_nn::{Activation, Sequential, VarBuilder, linear, seq};

// TODO: we do not need input dim and last_dim returned
pub fn build_sequential(
    input_dim: usize,
    layers: &[usize],
    vb: &VarBuilder,
    prefix: &str,
) -> Result<(Sequential, usize)> {
    let mut last_dim = input_dim;
    let mut nn = seq();
    let num_layers = layers.len();
    for (layer_idx, layer_size) in layers.iter().enumerate() {
        let layer_pp = format!("{prefix}{layer_idx}");
        if layer_idx == num_layers - 1 {
            nn = nn.add(linear(last_dim, *layer_size, vb.pp(layer_pp))?)
        } else {
            nn = nn
                .add(linear(last_dim, *layer_size, vb.pp(layer_pp))?)
                .add(Activation::Relu);
        }
        last_dim = *layer_size;
    }
    Ok((nn, last_dim))
}
