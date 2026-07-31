//! Small shared types used by the runnable workspace examples.

use std::any::Any;

/// Thread-safe, type-erased event payload used by UI examples.
pub type EventBox = Box<dyn Any + Send + Sync>;
