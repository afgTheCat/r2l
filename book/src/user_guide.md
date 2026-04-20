> [!WARNING]  
> **Pre-Alpha:** This library is under active development. Current APIs are almost surely going to change in the future.


# Environments

In order to use `r2l` an environments needs to be constructed. The environment trait is defined as.

```rust
{{#include ../../crates/r2l-core/src/env.rs:env}}
```


# PPO

# A2C

Below are some examples how to use `r2l-api`, which provides builders and APIs for the most commonly used usecases.
R2l is highly extensible, and if you are curious about the how to extend it/use it, start at [architecture](./architecture.md). 

# Examples

Below you can find full examples.
## PPO

<details open>
<summary>Sampler trait</summary>

```rust
{{#include ../../crates/r2l-examples/examples/ppo/main.rs:ppo}}
```

</details>


## A2C

<details open>
<summary>Sampler trait</summary>

```rust
{{#include ../../crates/r2l-examples/examples/a2c/main.rs:a2c}}
```

</details>
