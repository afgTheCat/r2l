use candle_core::Tensor;
use r2l_core::tensors::{Logp, LogpDiff, ValuesPred};

pub enum HookResult {
    Continue,
    Break,
}

pub struct PPOBatchData {
    pub logp: Logp,
    pub values_pred: ValuesPred,
    pub logp_diff: LogpDiff,
    pub ratio: Tensor,
}
