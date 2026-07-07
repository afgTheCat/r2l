use std::{
    cell::RefCell,
    sync::atomic::{AtomicU64, Ordering},
};

use rand::{RngExt, SeedableRng, rngs::StdRng};

static RNG_SEED: AtomicU64 = AtomicU64::new(0);

const ENV_STREAM: u64 = 0x9e37_79b9_7f4a_7c15;
const POLICY_STREAM: u64 = 0xbf58_476d_1ce4_e5b9;

thread_local! {
    static ENV_RNG: RefCell<StdRng> = RefCell::new(env_rng(0));
    static POLICY_RNG: RefCell<StdRng> = RefCell::new(policy_rng());
}

pub fn set_seed(seed: u64) {
    RNG_SEED.store(seed, Ordering::Relaxed);
    ENV_RNG.with_borrow_mut(|rng| *rng = env_rng(0));
    POLICY_RNG.with_borrow_mut(|rng| *rng = policy_rng());
}

/// Generates a deterministic seed for an environment reset.
pub fn env_seed() -> u64 {
    ENV_RNG.with_borrow_mut(|rng| rng.random::<u64>())
}

/// Creates a deterministic environment-reset RNG for one sampler worker.
pub fn env_worker_rng(worker_idx: usize) -> StdRng {
    env_rng(worker_idx + 1)
}

/// Runs a closure with the policy/action-sampling random stream.
pub fn with_policy_rng<T>(f: impl FnOnce(&mut StdRng) -> T) -> T {
    POLICY_RNG.with_borrow_mut(f)
}

fn env_rng(stream_id: usize) -> StdRng {
    StdRng::seed_from_u64(stream_seed(ENV_STREAM, stream_id as u64))
}

fn policy_rng() -> StdRng {
    StdRng::seed_from_u64(stream_seed(POLICY_STREAM, 0))
}

fn stream_seed(stream: u64, stream_id: u64) -> u64 {
    let seed = RNG_SEED.load(Ordering::Relaxed);
    splitmix64(seed ^ stream ^ stream_id.wrapping_mul(0xd2b7_4407_b1ce_6e93))
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
