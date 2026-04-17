use std::cell::RefCell;

use rand::{SeedableRng, rngs::StdRng};

thread_local! {
    pub static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}
