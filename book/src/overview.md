> [!WARNING]  
> **Pre-alpha:** This library is under active development. APIs may change
> between releases, and some planned features are not implemented yet.
> Basic familiarity with reinforcement learning concepts and rust knowledge is
> assumed.

## Why **r2l**

The goal of **r2l** is to be a customizable, ergonomic and easily embeddable
library. To be more exact:

- **Customizable**: users have control over _how_ agents are trained. **r2l**
  defines how components interact and exposes lifecycle hooks for custom
  behavior.
- **Ergonomic**: most users are not necessarily concerned with implementation
  details. High-level builders provide common configurations.
- **Embeddable**: **r2l** describes its backend requirements with traits.
  Candle and Burn implementations are currently available.

The scope of **r2l** is what Stable Baselines3 covers (by version 0.1.0) and
Tianshou (by version 1.0.0). On top of core algorithms, a hyperparameter tuning
library is to be included in the future.

## About this book

This book will help you get up to speed with _using_ and _hacking_ **r2l**. In
particular:

- [User Guide](./user_guide.md): Introduces how environments are to be
  implemented and how to work with the higher level APIs. Most users should
  start here. Some basic examples are also shown.
- [On policy algorithms](./on_policy_algorithms.md): A detailed architectural
  overview on what components on policy algorithms consists of, how the pieces
  fit together, and how to create your own custom hook system.
- [Off-policy algorithms](./off_policy_algorithms.md): Current support status.
