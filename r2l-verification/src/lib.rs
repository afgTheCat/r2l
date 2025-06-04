// Tools for verification
mod experiment_manager;
mod parse_config;

// TODO: move this to a dedicated tests folder
#[cfg(test)]
mod test {
    use crate::parse_config::parse_config_files;
    // Vericifation happens in 5 steps
    // Step 1: Parse the configuration files for the pre trained model
    // Step 2: Try constructing each model according to the configuration
    // Step 3: Run the sb3 model => collect the rewards received per episode
    // Step 4: Run the r2l model => collect the rewards received per episode
    // Step 5: Compare the rewards according to some metric

    // TODO: This will probably need to be moved to it's own place
    #[test]
    fn verification() {
        // STEP 1: parse the config files
        let config_files = parse_config_files().unwrap();
        let a2c_acrobat_config = config_files.into_iter().find(|c| c.model == "ppo").unwrap();
    }
}
