use r2l_api::builders::sampler::SamplerType;

fn build_sampler() {
    let sampler = SamplerType {
        capacity: 2048,
        hook_options: Default::default(),
        env_pool_type: Default::default(),
    };
    // sampler.build_with_builder_type(builder_type, device)
}
