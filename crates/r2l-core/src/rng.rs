use std::cell::RefCell;

use rand::{SeedableRng, rngs::StdRng};

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}

/// Replaces the current thread's deterministic random stream with `seed`.
pub fn set_seed(seed: u64) {
    RNG.with_borrow_mut(|rng| *rng = StdRng::seed_from_u64(seed));
}

/// Samples a `u64` from the current thread's random stream.
#[must_use]
pub fn sample_u64() -> u64 {
    RNG.with_borrow_mut(rand::RngExt::random::<u64>)
}

/// Runs a closure with the policy/action-sampling random stream.
pub fn with_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    RNG.with_borrow_mut(f)
}
