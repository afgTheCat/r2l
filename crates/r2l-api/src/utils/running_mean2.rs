#[allow(unused_imports)]
pub use r2l_core::running_mean::RunningMeanStd3;

#[cfg(test)]
mod test {
    use anyhow::Result;
    use candle_core::{Device, Tensor};
    use r2l_core::tensor::R2lTensor;
    use rand::{RngExt, rng};

    use crate::utils::running_mean2::RunningMeanStd3;

    fn biased_var(t: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
        let mean = t.mean_keepdim(dim)?;
        let squares = t.broadcast_sub(&mean)?.sqr()?;
        (squares.sum_keepdim(dim)? / t.dim(dim)? as f64)?.squeeze(dim)
    }

    #[test]
    fn new_test_biased_var() -> Result<()> {
        let device = Device::Cpu;
        let data = [
            0.488_830_6_f32,
            0.48259816,
            0.793_288_1,
            0.191_038_6,
            0.11694599,
            0.538_540_4,
        ];
        let tensors = data
            .chunks_exact(3)
            .map(|chunk| Tensor::from_slice(chunk, 3, &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let var = Tensor::var_tensors(&tensors);
        let reference_var =
            Tensor::from_slice(&[0.02217002f32, 0.03342538, 0.01622409], 3, &device)?;
        let var_diff = (&var - &reference_var)?.abs()?.max(0)?;
        assert!(var_diff.to_scalar::<f32>()? < 1e-5, "var diff");
        Ok(())
    }

    #[test]
    fn new_test_running_mean_std_running() -> Result<()> {
        let device = Device::Cpu;
        let mut rng = rng();
        let shape = (10, 3);
        let mut rms = RunningMeanStd3::<Tensor>::new(vec![shape.1]);
        let mut all_data = vec![];

        for _ in 0..100 {
            let data: Vec<f32> = (0..30).map(|_| rng.random_range(-1.0..1.0)).collect();
            let tensors = data
                .chunks_exact(shape.1)
                .map(|chunk| Tensor::from_slice(chunk, shape.1, &device))
                .collect::<candle_core::Result<Vec<_>>>()?;
            rms.update(&tensors);
            all_data.extend(data);
        }

        let all_tensor = Tensor::from_slice(&all_data, (all_data.len() / 3, 3), &device)?;
        let reference_mean = all_tensor.mean(0)?;
        let reference_var = biased_var(&all_tensor, 0)?;

        let mean_diff = (&rms.mean - &reference_mean)?.abs()?.max(0)?;
        let var_diff = (&rms.var - &reference_var)?.abs()?.max(0)?;

        let eps = 1e-5;

        assert!(mean_diff.to_scalar::<f32>()? < eps, "mean mismatch");
        assert!(var_diff.to_scalar::<f32>()? < eps, "variance mismatch");

        Ok(())
    }
}
