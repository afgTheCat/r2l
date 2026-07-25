//! Small shared types used by the runnable workspace examples.

use std::any::Any;

/// Thread-safe, type-erased event payload used by UI examples.
pub type EventBox = Box<dyn Any + Send + Sync>;

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use burn::backend::NdArray;
    use burn_store::{ModuleStore, SafetensorsStore};
    use r2l_burn::distributions::diagonal::DiagGaussianDistribution;

    #[test]
    fn module_loading() {
        let best_model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ppo.safetensor");
        let mut store = SafetensorsStore::from_file(best_model_path);
        let all_tensors = store.get_all_snapshots().unwrap();
        println!("{all_tensors:#?}");

        let _model = DiagGaussianDistribution::<NdArray>::from_store(&mut store);
    }
}
