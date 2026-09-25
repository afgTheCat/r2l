#![cfg(not(feature = "simd"))]

use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    tensor::{Tensor, TensorData},
};

type Backend = NdArray<f32>;

fn reciprocal_bits() -> Vec<u32> {
    let device = NdArrayDevice::default();
    let input = Tensor::<Backend, 1>::from_data(TensorData::new(vec![3.0; 257], [257]), &device);

    input
        .recip()
        .into_data()
        .as_slice::<f32>()
        .expect("reciprocal output should contain f32 values")
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
fn ndarray_simd_reciprocal_is_bitwise_deterministic() {
    let expected = reciprocal_bits();

    for run in 1..32 {
        let actual = reciprocal_bits();
        if actual != expected {
            let index = actual
                .iter()
                .zip(&expected)
                .position(|(actual, expected)| actual != expected)
                .expect("different vectors should contain a differing element");
            panic!(
                "SIMD reciprocal diverged on run {run} at element {index}: {:#010x} != {:#010x}",
                actual[index], expected[index]
            );
        }
    }
}
