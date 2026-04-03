use crate::tensor::{R2lBuffer, R2lTensor, R2lTensorOp};
use candle_core::{Device, Tensor as CandleTensor};

impl From<R2lBuffer> for CandleTensor {
    fn from(val: R2lBuffer) -> Self {
        let R2lBuffer { data, shape } = val;
        CandleTensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }
}

impl From<CandleTensor> for R2lBuffer {
    fn from(value: CandleTensor) -> Self {
        let shape = value.shape().clone().into_dims();
        let data: Vec<f32> = value.to_vec1().unwrap();
        Self { data, shape }
    }
}

impl R2lTensor for CandleTensor {
    fn to_vec(&self) -> Vec<f32> {
        self.to_vec1().unwrap()
    }
}

impl R2lBuffer {
    pub fn to_candle_tensor(&self) -> CandleTensor {
        self.clone().into()
    }

    // TODO: implement this without relying on candle
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        let t = self.to_candle_tensor();
        let min_t = min.to_candle_tensor();
        let max_t = max.to_candle_tensor();
        (t.clamp(&min_t, &max_t).unwrap()).into()
    }

    pub fn from_candle_tensor(tensor: &CandleTensor) -> Self {
        tensor.clone().into()
    }
}

impl R2lTensorOp for CandleTensor {
    fn calculate_logp_diff(logp: &Self, logp_old: &Self) -> anyhow::Result<Self> {
        Ok((logp - logp_old)?)
    }

    fn calculate_ratio(logp_diff: &Self) -> anyhow::Result<Self> {
        Ok(logp_diff.exp()?)
    }

    fn calculate_policy_loss(
        ratio: &Self,
        advantages: &Self,
        clip_range: f32,
    ) -> anyhow::Result<Self> {
        let clip_adv = (ratio.clamp(1. - clip_range, 1. + clip_range)? * advantages.clone())?;
        Ok(
            candle_core::Tensor::minimum(&(ratio * advantages)?, &clip_adv)?
                .neg()?
                .mean_all()?,
        )
    }

    fn calculate_value_loss(returns: &Self, values_pred: &Self) -> anyhow::Result<Self> {
        Ok(returns.sub(values_pred)?.sqr()?.mean_all()?)
    }
}
