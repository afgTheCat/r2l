// We basically want to expose a function that sets the seed. If no seet is set, then we will set
// one on the first invocation.

use rand::{SeedableRng, rngs::StdRng};
use std::cell::RefCell;

thread_local! {
    pub static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}
