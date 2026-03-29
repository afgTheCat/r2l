// TODO: re evaluate wheter we need dedicated new types or not
use burn::{Tensor, prelude::Backend};
use derive_more::{Deref, DerefMut, Display};

#[derive(Deref, DerefMut, Debug, Display)]
pub struct Advantages<B: Backend>(pub Tensor<B, 1>);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct Returns<B: Backend>(pub Tensor<B, 1>);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct Logp<B: Backend>(pub Tensor<B, 1>);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct ValuesPred<B: Backend>(pub Tensor<B, 1>);

#[derive(Deref, DerefMut, Debug, Display)]
pub struct LogpDiff<B: Backend>(pub Tensor<B, 1>);
