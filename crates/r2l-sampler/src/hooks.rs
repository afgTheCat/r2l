use r2l_core::{models::Actor, prelude::TrajectoryContainer, tensor::R2lTensor};

// TODO: highly experimental sampler trait
// should be ready for next release
#[allow(dead_code)]
trait PreprocessorY<T: R2lTensor, B: TrajectoryContainer<Tensor = T>> {
    // The question is, can we make this dyn compatible? Otherwise we just use a ref
    fn preprocess_states(&mut self, policy: &dyn Actor<Tensor = T>, buffers: &mut [B]);
}
