use std::{
    cell::RefCell,
    sync::atomic::{AtomicU64, Ordering},
};

use rand::{RngExt, SeedableRng, rngs::StdRng};

static RNG_SEED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(env_rng(0));
}

pub fn set_seed(seed: u64) {
    RNG_SEED.store(seed, Ordering::Relaxed);
    RNG.with_borrow_mut(|rng| *rng = env_rng(0));
}

/// Generates a deterministic seed for an environment reset.
pub fn env_seed() -> u64 {
    RNG.with_borrow_mut(|rng| rng.random::<u64>())
}

/// Runs a closure with the policy/action-sampling random stream.
pub fn with_policy_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    RNG.with_borrow_mut(f)
}

pub fn env_rng(stream_id: usize) -> StdRng {
    let seed = RNG_SEED.load(Ordering::Relaxed);
    StdRng::seed_from_u64(seed.wrapping_add(stream_id as u64))
}
