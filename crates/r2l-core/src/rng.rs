use std::cell::RefCell;

use rand::{RngExt, SeedableRng, rngs::StdRng};

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}

pub fn set_seed(seed: u64) {
    RNG.with_borrow_mut(|rng| *rng = StdRng::seed_from_u64(seed));
}

pub fn next_seed() -> u64 {
    RNG.with_borrow_mut(|rng| rng.random::<u64>())
}

/// Runs a closure with the policy/action-sampling random stream.
pub fn with_policy_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    RNG.with_borrow_mut(f)
}
