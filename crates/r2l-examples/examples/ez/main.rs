use r2l::PPOBuilder;

fn main() -> anyhow::Result<()> {
    let mut algorithm = PPOBuilder::gym("Pendulum-v1", 4)?.build()?;
    algorithm.train()?;
    Ok(())
}
