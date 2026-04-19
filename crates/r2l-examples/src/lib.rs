use std::any::Any;

pub type EventBox = Box<dyn Any + Send + Sync>;
