> [!WARNING]  
> **Pre-Alpha:** This library is under active development. Current APIs are
> almost surely going to change in the future and documentation might not be up
> to date. On top of **r2l**, this book is also heavily under construction.
> Basic familiarity with reinforcement learning concepts and rust knowledge is
> assumed.

## Why **r2l**

The goal of **r2l** is to be a customizable, ergonomic and easily embeddable
library. To be more exact:

- **Customizable**: the user has great control influencing _how_ agents are
  trained. While **r2l** defines how different components interact with each
  other, and the core logic of they implement, it also exposes the internals
  through a hook system for the user.
- **Ergonomic**: most users are not necessarily concerned with implementation
  details. In order to alleviate the burden of implementing a complete
  algorithm, building on the core components, **r2l** aims to implement commonly
  setups.
- **Embeddable**: my goal with **r2l** is to be able to use it within a diverse
  set of applications/environments. Instead of choosing a single deep learning
  framework, **r2l** uses traits to describe it's needs. In practice, we
  currently support **candle** and **burn**. If you have a deep learning
  framework, and would like to have **r2l** support it, open a PR/make an issue.

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
- [Off policy algorithms](./off_policy_algorithms.md): To be added in v0.0.4.
