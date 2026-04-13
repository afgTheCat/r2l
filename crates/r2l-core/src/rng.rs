// We basically want to expose a function that sets the seed. If no seet is set, then we will set
// one on the first invocation.

use std::cell::RefCell;

use rand::{SeedableRng, rngs::StdRng};

thread_local! {
    pub static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}
