//! R2L - a RL framework written in Rust
//!
//! ## Design goals
//!
//! R2L aims to implement some common RL algorithms and by exposing the trainig loop, let's you
//! customize how you want to log and control learning.
//!
//! ## Terminology
//!
//! Since I did not find an authorative list of definitions, these are the ones I came up with
//! (subject to change at least until v.0.1.0):
//!
//! - Algorithm (trait `Algorithm`): Encompasses all elements of the training infrastructure. It
//! owns the environment, knows how to collect rollouts and train itself. Example: `OnPolicyAlgorithm`
//! - EnvPool: Has access to one or more environments. It is responsible for coordinating trainig
//! accross multiple environments. Whether this abstraction will be folded into Env is an open
//! question.
//! - Env: Basically the same as a gym env
//!
//! Since
//!
//! ## Other crates
//!
//! R2L has a number of crates. This one contains the neccessary infrastructure for the training
//! loop by introducing common traits and structures.

pub mod distributions;
pub mod policies;
pub mod thread_safe_sequential;
pub mod utils;
