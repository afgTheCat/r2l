// TODO: re evaluate wheter we need dedicated new types or not
use candle_core::Tensor;
use derive_more::{Deref, DerefMut};

#[derive(Deref, DerefMut)]
pub struct Advantages(pub Tensor);

#[derive(Deref, DerefMut)]
pub struct Returns(pub Tensor);

#[derive(Deref, DerefMut)]
pub struct Logp(pub Tensor);

#[derive(Deref, DerefMut)]
pub struct ValuesPred(pub Tensor);

#[derive(Deref, DerefMut)]
pub struct PolicyLoss(pub Tensor);

#[derive(Deref, DerefMut)]
pub struct ValueLoss(pub Tensor);

#[derive(Deref, DerefMut)]
pub struct LogpDiff(pub Tensor);

#[derive(Deref, DerefMut)]
pub struct NotUsed(pub Tensor);
