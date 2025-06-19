use nom::{
    IResult, Parser,
    branch::alt,
    bytes::{tag, take_until},
    character::complete::newline,
    combinator::{complete, opt},
    multi::many0,
};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub env_name: String,
    pub config_file: String,
    pub args: HashMap<String, String>,
    pub config: HashMap<String, String>,
}

#[derive(Debug)]
pub struct ModelConfigs {
    pub model: String,
    pub envs: Vec<EnvConfig>,
}

// TODO: add more models once ready
const MODELS_TO_VERIFY: [&str; 2] = ["ppo", "a2c"];

fn skip_header(input: &str) -> IResult<&str, ()> {
    let (input, _) = take_until("\n").parse(input)?;
    let (input, _) = newline(input)?;
    Ok((input, ()))
}

fn parse_key_value(input: &str) -> IResult<&str, (String, String)> {
    let (input, p) = alt((tag("- - - "), tag("  - - "))).parse(input)?;
    let (input, key) = take_until("\n").parse(input)?;
    let (input, _) = newline(input)?;
    let (input, _) = tag("    - ").parse(input)?;
    let (input, value) = take_until("\n").parse(input)?;
    let (input, _) = opt(newline).parse(input)?;
    Ok((input, (key.trim().to_string(), value.trim().to_string())))
}

fn parse_file(input: &str) -> IResult<&str, HashMap<String, String>> {
    let (input, _) = skip_header(input)?;
    let (input, pairs) = many0(complete(parse_key_value)).parse(input)?;
    Ok((input, pairs.into_iter().collect()))
}

pub fn parse_config_files() -> std::io::Result<Vec<ModelConfigs>> {
    let models_path = format!("{}/rl-trained-agents", env!("CARGO_MANIFEST_DIR"));
    let path = Path::new(&models_path);
    let agent_paths = MODELS_TO_VERIFY.map(|agent_path| path.join(Path::new(agent_path)));
    let mut model_configs = vec![];
    for (agent_path, agent_name) in agent_paths.into_iter().zip(MODELS_TO_VERIFY) {
        let mut model_config = ModelConfigs {
            model: agent_name.to_owned(),
            envs: vec![],
        };
        // TODO: this is not super readable
        let env_paths: Vec<_> = fs::read_dir(agent_path)?
            .filter_map(|s| s.ok().and_then(|s| s.path().is_dir().then(|| s.path())))
            .collect();
        for env_path in env_paths {
            // TODO: unwrap hell
            // let env_name = env_path.file_name().unwrap().to_str().unwrap().to_owned();
            let config_dir = fs::read_dir(env_path)?
                .find_map(|s| s.ok().and_then(|s| s.path().is_dir().then(|| s.path())))
                .unwrap(); // NOTE: this directory always exists, so it's safe to unwrap
            let args_file = config_dir.join("args.yml");
            let args_file_content = std::fs::read_to_string(args_file)?;
            let (_, args) = parse_file(&args_file_content)
                .map_err(|err| std::io::Error::other(err.to_owned()))?;
            let env_name = args.get("env").unwrap().to_owned();
            let config_file = config_dir.join("config.yml");
            let config_file_content = std::fs::read_to_string(config_file.clone())?;
            let (_, config) = parse_file(&config_file_content)
                .map_err(|err| std::io::Error::other(err.to_owned()))?;
            let config_file = fs::canonicalize(&config_file)?;
            let config_file = config_file.to_string_lossy().into_owned();
            model_config.envs.push(EnvConfig {
                env_name,
                args,
                config_file,
                config,
            });
        }
        model_configs.push(model_config);
    }
    Ok(model_configs)
}
