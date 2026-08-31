use crate::{error::TensorError, tensor::R2lTensor};

type Result<T> = std::result::Result<T, TensorError>;

/// Online per-element mean and variance for tensor samples.
#[derive(Clone)]
pub struct RunningMeanStd<T: R2lTensor> {
    /// Current per-element mean.
    pub mean: T,
    /// Current per-element population variance.
    pub var: T,
    pub count: f32,
}

// mega simplified view
impl<T: R2lTensor> RunningMeanStd<T> {
    /// Creates zero-count statistics for tensors with `shape`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor backend cannot create tensors with `shape`.
    pub fn new(shape: Vec<usize>) -> Result<Self> {
        let mean = T::zeros(shape.clone())?;
        let var = T::zeros(shape)?;
        Ok(Self {
            mean,
            var,
            count: 0.,
        })
    }

    pub fn build(mean: T, var: T, count: f32) -> Self {
        Self { mean, var, count }
    }

    fn update_from_moments(
        &mut self,
        batch_mean: &T,
        batch_var: &T,
        batch_count: f32,
    ) -> Result<()> {
        if batch_count == 0.0 {
            return Ok(());
        }
        let tot_count = self.count + batch_count;
        let delta = batch_mean.sub(&self.mean)?;
        self.mean = self.mean.add(&delta.mul_scalar(batch_count / tot_count)?)?;
        let m_a = self.var.mul_scalar(self.count)?;
        let m_b = batch_var.mul_scalar(batch_count)?;
        let m_2 = m_a.add(&m_b)?.add(
            &delta
                .sqr()?
                .mul_scalar(self.count * batch_count / tot_count)?,
        )?;
        self.var = m_2.mul_scalar(1.0 / tot_count)?;
        self.count = tot_count;
        Ok(())
    }

    /// Updates the statistics from a batch of tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty or a tensor operation fails.
    pub fn update(&mut self, t: &[T]) -> Result<()> {
        let mean = T::mean_tensors(t)?;
        let var = T::var_tensors(t)?;
        self.update_from_moments(&mean, &var, t.len() as f32)
    }

    /// Converts flat samples to tensors and updates the statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if a sample has an incompatible shape or the update fails.
    pub fn update_from_vec(&mut self, t: &[Vec<f32>]) -> Result<()> {
        let t = t
            .iter()
            .map(|t| T::from_slice_and_shape(t, self.mean.to_shape()))
            .collect::<Result<Vec<_>>>()?;
        self.update(&t)
    }
}

/// Running mean and variance for scalar `f32` samples.
#[derive(Clone, Debug)]
pub struct RunningMeanStdF32 {
    /// Current scalar mean.
    pub mean: f32,
    /// Current scalar population variance.
    pub var: f32,
    count: f32,
}

impl Default for RunningMeanStdF32 {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningMeanStdF32 {
    /// Creates scalar running statistics with a small initial sample count.
    #[must_use]
    pub fn new() -> Self {
        Self::with_epsilon(1e-4)
    }

    /// Creates scalar running statistics with the provided initial sample count.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is negative.
    #[must_use]
    pub fn with_epsilon(epsilon: f32) -> Self {
        assert!(epsilon >= 0.0);
        Self {
            mean: 0.0,
            var: 1.0,
            count: epsilon,
        }
    }

    fn update_from_moments(&mut self, batch_mean: f32, batch_var: f32, batch_count: f32) {
        if batch_count == 0.0 {
            return;
        }
        let total_count = self.count + batch_count;
        let delta = batch_mean - self.mean;
        self.mean += delta * batch_count / total_count;
        let m_a = self.var * self.count;
        let m_b = batch_var * batch_count;
        let m_2 = m_a + m_b + delta.powi(2) * self.count * batch_count / total_count;
        self.var = m_2 / total_count;
        self.count = total_count;
    }

    /// Updates the running statistics from a batch of scalar samples.
    pub fn update(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let batch_count = samples.len() as f32;
        let batch_mean = samples.iter().sum::<f32>() / batch_count;
        let batch_var = samples
            .iter()
            .map(|sample| (*sample - batch_mean).powi(2))
            .sum::<f32>()
            / batch_count;
        self.update_from_moments(batch_mean, batch_var, batch_count);
    }
}
