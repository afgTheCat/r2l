use crate::hooks::a2c::DefaultA2CHook;
use std::marker::PhantomData;

#[derive(Debug, Clone, Default)]
pub struct DefaultA2CHookBuilder;

impl DefaultA2CHookBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build<T>(self) -> DefaultA2CHook<T> {
        DefaultA2CHook { _lm: PhantomData }
    }
}
