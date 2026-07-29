use r2l_api::PPOAlgorithmBuilder;

#[test]
fn inference_thing() {
    let b = PPOAlgorithmBuilder::gym("", 10);
    let algo = b.build().unwrap();
}
