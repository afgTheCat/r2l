use candle_core::{Device, Result, Shape, Tensor, Var};
use rand::RngExt;

pub(crate) fn overwrite_uniform(tensor: &Tensor, bound: f32) -> Result<()> {
    if !matches!(tensor.device(), Device::Cpu) || !tensor.is_variable() {
        return Ok(());
    }
    let values = r2l_core::rng::with_rng(|rng| {
        (0..tensor.elem_count())
            .map(|_| rng.random_range(-bound..bound))
            .collect::<Vec<f32>>()
    });
    let values = Tensor::from_vec(values, tensor.shape().clone(), tensor.device())?;
    Var::from_tensor(tensor)?.set(&values)
}

pub(crate) fn standard_normal(shape: &Shape, device: &Device) -> Result<Tensor> {
    if !matches!(device, Device::Cpu) {
        return Tensor::randn(0f32, 1f32, shape, device);
    }
    let values = r2l_core::rng::with_rng(|rng| {
        let mut values = Vec::with_capacity(shape.elem_count());
        while values.len() < shape.elem_count() {
            let radius = (-2.0 * (1.0 - rng.random::<f32>()).ln()).sqrt();
            let angle = 2.0 * std::f32::consts::PI * rng.random::<f32>();
            values.push(radius * angle.cos());
            if values.len() < shape.elem_count() {
                values.push(radius * angle.sin());
            }
        }
        values
    });
    Tensor::from_vec(values, shape.clone(), device)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Var};

    use super::{overwrite_uniform, standard_normal};

    fn sample() -> (Vec<f32>, Vec<f32>) {
        r2l_core::rng::set_seed(42);
        let variable = Var::zeros(4, DType::F32, &Device::Cpu).unwrap();
        overwrite_uniform(variable.as_tensor(), 1.0).unwrap();
        let variable = variable.as_tensor().to_vec1().unwrap();
        let noise = standard_normal(&4.into(), &Device::Cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        (variable, noise)
    }

    #[test]
    fn cpu_randomness_uses_r2l_seed() {
        assert_eq!(sample(), sample());
    }
}
