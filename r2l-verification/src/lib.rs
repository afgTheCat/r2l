// Tools for verification
mod experiment_manager;
mod parse_config;
mod python_verifier;

// TODO: move this to a dedicated tests folder
#[cfg(test)]
mod test {
    use crate::{
        experiment_manager::test_construct_configs, parse_config::parse_config_files,
        python_verifier::python_verification_pass,
    };
    // Vericifation happens in 5 steps
    // Step 1: Parse the configuration files for the pre trained model => we parse the config
    // files, save the associated metadata to be used further on
    // Step 2: Try constructing each model according to the configuration
    // Step 3: Run the sb3 model => collect the rewards received per episode
    // Step 4: Run the r2l model => collect the rewards received per episode
    // Step 5: Compare the rewards according to some metric

    // TODO: This will probably need to be moved to it's own place
    #[test]
    fn verification() {
        // Step 1: Parse the configuration files for the pre trained model
        let configs = parse_config_files().unwrap();
        // Step 2: Try constructing each model according to the configuration
        // test_construct_configs(configs);
        let py_results = python_verification_pass(configs);

        // let a2c_config = configs.into_iter().find(|c| c.model == "a2c").unwrap();
        // let mut exp_manager = ExperimentManager {};
        // exp_manager.process_config1(a2c_config);
    }
}
