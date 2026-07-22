// The main reponsibiliy of this code is to:
// - run extensive test suits if we need be
//  -- with seeds etc (repro test)
//  -- with seeds not set
// - generate statistics, generate figures etc

mod zoo_parser;

use std::path::PathBuf;

use r2l_core::rng::set_seed;

use crate::zoo_parser::ZooConfig;

const SEED: u64 = 0;
const CONFIG_PATH: &str = "/home/gabor/projects/r2l/assets/ppo.yaml";
const LOG_DIR: &str = "/home/gabor/projects/r2l/logs";
const SMALL_ENVIRONMENTS: [&str; 5] = [
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
];

fn main() {
    // Sets the seed. Burn does respect this, making the tests reproducible
    set_seed(SEED);
    let config_path = PathBuf::from(CONFIG_PATH);
    let mut zoo_config = ZooConfig::parse_rl_zoo_config(config_path);
    println!("Not testing {}", zoo_config.unsupported_envs.len());
    for env_name in SMALL_ENVIRONMENTS {
        let Some(env_config) = zoo_config.supported_envs.remove(env_name) else {
            eprintln!("Skipping missing configuration for {env_name}");
            continue;
        };
        println!("{env_name}");
        let log_file = format!("{LOG_DIR}/{env_name}.csv");
        match env_config.build_burn_ppo_algorithm(env_name, log_file.into()) {
            Ok(mut algo) => {
                if let Err(err) = algo.train() {
                    eprintln!("Training {env_name} failed: {err:#}");
                }
            }
            Err(err) => eprintln!("Building {env_name} failed: {err:#}"),
        }
    }
}
