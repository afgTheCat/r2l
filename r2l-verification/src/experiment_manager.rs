use crate::parse_config::ModelConfigs;
use r2l_core::Algorithm;

struct ExperimentManager {}

impl ExperimentManager {
    // parse generic config, has to pass in the a buffer to collect rollouts to for later analysis
    fn process_config(&mut self, model_config: ModelConfigs) -> Vec<Box<dyn Algorithm>> {
        todo!()
    }
}
