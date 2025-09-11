use candle_core::Device;
use candle_core::Result;
use candle_core::Shape;
use candle_core::Tensor;
use candle_core::shape::Dim;

pub struct RunningMeanStd {
    pub mean: Tensor,
    pub var: Tensor,
    pub count: f32,
    pub device: Device,
}

fn biased_var<D: Dim>(t: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(t.shape(), "var")?;
    let mean = t.mean_keepdim(dim)?;
    let squares = t.broadcast_sub(&mean)?.sqr()?;
    (squares.sum_keepdim(dim)? / t.dim(dim)? as f64)?.squeeze(dim)
}

impl RunningMeanStd {
    pub fn new<S: Into<Shape> + Copy>(shape: S, device: Device) -> Self {
        let mean = Tensor::zeros(shape, candle_core::DType::F32, &device).unwrap();
        let var = Tensor::zeros(shape, candle_core::DType::F32, &device).unwrap();
        Self {
            mean,
            var,
            count: 0.,
            device,
        }
    }

    pub fn update(&mut self, arr: &Tensor) -> Result<()> {
        let batch_mean = arr.mean(0)?;
        let batch_var = biased_var(&arr, 0)?;
        let batch_count = arr.shape().dim(0)? as f32;
        self.update_from_moments(batch_mean, batch_var, batch_count)?;
        Ok(())
    }

    // implements Welford's algorithm
    fn update_from_moments(
        &mut self,
        batch_mean: Tensor,
        batch_var: Tensor,
        batch_count: f32,
    ) -> Result<()> {
        let delta = batch_mean.sub(&self.mean)?;
        let tot_count = self.count + batch_count;
        self.mean = self.mean.add(
            &(&delta.broadcast_mul(&Tensor::full(batch_count / tot_count, (), &self.device)?)?),
        )?;
        let m_a = self
            .var
            .broadcast_mul(&Tensor::full(self.count, (), self.var.device())?)?;
        let m_b = batch_var.broadcast_mul(&Tensor::full(batch_count, (), self.var.device())?)?;
        let m_2 = m_a
            .add(&m_b)?
            .add(&delta.sqr()?.broadcast_mul(&Tensor::full(
                self.count * batch_count / (self.count + batch_count),
                (),
                &self.device,
            )?)?)?;
        self.var = m_2.broadcast_mul(&Tensor::full(
            1. / (self.count + batch_count),
            (),
            &self.device,
        )?)?;
        self.count = batch_count + self.count;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use candle_core::{Device, Result, Tensor};
    use rand::{Rng, thread_rng};

    use crate::utils::running_mean::{RunningMeanStd, biased_var};

    #[test]
    fn test_biased_var() -> Result<()> {
        let test_t = Tensor::from_slice(
            &[
                0.48883059f32,
                0.48259816,
                0.79328812,
                0.19103859,
                0.11694599,
                0.53854045,
            ],
            (2, 3),
            &Device::Cpu,
        )?;
        let var = biased_var(&test_t, 0)?;
        let reference_var =
            Tensor::from_slice(&[0.02217002f32, 0.03342538, 0.01622409], 3, &Device::Cpu)?;
        let var_diff = (&var - &reference_var)?.abs()?.max(0)?;
        assert!(var_diff.to_scalar::<f32>()? < 1e-5, "var diff");
        Ok(())
    }

    #[test]
    fn test_running_mean_std_running() -> Result<()> {
        let device = Device::Cpu;
        let mut rng = thread_rng();
        let shape = (10, 3);
        let mut rms = RunningMeanStd::new(shape.1, device.clone()); // replace `shape` with e.g. 3 or 10
        let mut all_data = vec![];

        for _ in 0..100 {
            let data: Vec<f32> = (0..30).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let tensor = Tensor::from_slice(&data, shape, &device)?; // shape (10, 3)
            rms.update(&tensor)?;
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
