// TODO: re evaluate wheter we need dedicated new types or not
use candle_core::Tensor;
use derive_more::{Deref, DerefMut, Display};

#[derive(Deref, DerefMut, Debug, Display)]
pub struct Advantages(pub Tensor);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct Returns(pub Tensor);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct Logp(pub Tensor);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct ValuesPred(pub Tensor);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct PolicyLoss(pub Tensor);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct ValueLoss(pub Tensor);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct LogpDiff(pub Tensor);
