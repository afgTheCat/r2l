#![feature(prelude_import)]
//! R2L - a RL framework written in Rust
//!
//! ## Design goals
//!
//! R2L aims to implement some common RL algorithms and by exposing the trainig loop, let's you
//! customize how you want to log and control learning.
//!
//! ## Terminology
//!
//! Since I did not find an authorative list of definitions, these are the ones I came up with
//! (subject to change at least until v.0.1.0):
//!
//! - Algorithm (trait `Algorithm`): Encompasses all elements of the training infrastructure. It
//! owns the environment, knows how to collect rollouts and train itself. Example: `OnPolicyAlgorithm`
//! - EnvPool: Has access to one or more environments. It is responsible for coordinating trainig
//! accross multiple environments. Whether this abstraction will be folded into Env is an open
//! question.
//! - Env: Basically the same as a gym env
//!
//! Since
//!
//! ## Other crates
//!
//! R2L has a number of crates. This one contains the neccessary infrastructure for the training
//! loop by introducing common traits and structures.
#[macro_use]
extern crate std;
#[prelude_import]
use std::prelude::rust_2024::*;
pub mod distributions {
    pub mod diagonal_distribution {
        use core::f32;
        use burn::{prelude::Backend, tensor::{Distribution as TDistribution, Tensor}};
        use crate::{
            distributions::Distribution, thread_safe_sequential::ThreadSafeSequential,
            utils::tensor_sqr,
        };
        pub struct DiagGaussianDistribution<B: Backend> {
            noise: Tensor<B, 2>,
            log_std: Tensor<B, 2>,
            mu_net: ThreadSafeSequential<B>,
            device: B::Device,
        }
        #[automatically_derived]
        impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug
        for DiagGaussianDistribution<B>
        where
            B::Device: ::core::fmt::Debug,
        {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                ::core::fmt::Formatter::debug_struct_field4_finish(
                    f,
                    "DiagGaussianDistribution",
                    "noise",
                    &self.noise,
                    "log_std",
                    &self.log_std,
                    "mu_net",
                    &self.mu_net,
                    "device",
                    &&self.device,
                )
            }
        }
        impl<B: Backend> Distribution<B> for DiagGaussianDistribution<B> {
            fn get_action(
                &self,
                observation: Tensor<B, 2>,
            ) -> (Tensor<B, 2>, Tensor<B, 1>) {
                let mu = self.mu_net.forward(observation.clone());
                let std = self.log_std.clone().exp();
                let noise = Tensor::<
                    B,
                    2,
                >::random(
                    self.log_std.shape(),
                    TDistribution::Normal(0., 1.),
                    &self.device,
                );
                let actions = mu + std * noise;
                let logp = self.log_probs(observation, actions.clone());
                (actions, logp)
            }
            fn log_probs(
                &self,
                states: Tensor<B, 2>,
                actions: Tensor<B, 2>,
            ) -> Tensor<B, 1> {
                let mu = self.mu_net.forward(states);
                let mut std = self.log_std.clone().exp();
                let var = tensor_sqr(std);
                let log_sqrt_2pi = f32::ln(f32::sqrt(2f32 * f32::consts::PI));
                let log_probs: Tensor<B, 2> = tensor_sqr(actions - mu) / (2. * var);
                let log_probs: Tensor<B, 2> = log_probs.neg();
                let log_probs: Tensor<B, 2> = log_probs - self.log_std.clone()
                    - log_sqrt_2pi;
                log_probs.sum()
            }
            fn entropy(&self) -> Tensor<B, 1> {
                let log_2pi_plus_1_div_2 = 0.5 * ((2. * f32::consts::PI).ln() + 1.);
                (self.log_std.clone() + log_2pi_plus_1_div_2).sum()
            }
            fn std(&self) -> f32 {
                ::core::panicking::panic("not yet implemented")
            }
            fn resample_noise(&mut self) {
                ::core::panicking::panic("not yet implemented")
            }
        }
    }
    use burn::{prelude::Backend, tensor::Tensor};
    use enum_dispatch::enum_dispatch;
    use std::{f32, fmt::Debug};
    use crate::distributions::diagonal_distribution::DiagGaussianDistribution;
    pub trait Distribution<B: Backend>: Sync + Debug + 'static {
        fn get_action(&self, observation: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>);
        fn log_probs(&self, states: Tensor<B, 2>, actions: Tensor<B, 2>) -> Tensor<B, 1>;
        fn std(&self) -> f32;
        fn entropy(&self) -> Tensor<B, 1>;
        fn resample_noise(&mut self) {}
    }
    pub enum DistribnutionKind<B: Backend> {
        Diagonal(DiagGaussianDistribution<B>),
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for DistribnutionKind<B> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                DistribnutionKind::Diagonal(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Diagonal",
                        &__self_0,
                    )
                }
            }
        }
    }
    impl<B: Backend> Distribution<B> for DistribnutionKind<B> {
        fn get_action(&self, observation: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
            ::core::panicking::panic("not yet implemented")
        }
        fn log_probs(
            &self,
            states: Tensor<B, 2>,
            actions: Tensor<B, 2>,
        ) -> Tensor<B, 1> {
            ::core::panicking::panic("not yet implemented")
        }
        fn std(&self) -> f32 {
            ::core::panicking::panic("not yet implemented")
        }
        fn entropy(&self) -> Tensor<B, 1> {
            ::core::panicking::panic("not yet implemented")
        }
        fn resample_noise(&mut self) {
            ::core::panicking::panic("not yet implemented")
        }
    }
}
pub mod policies {
    pub mod paralell_actor_critic {
        use burn::{
            optim::{AdamW, SimpleOptimizer, adaptor::OptimizerAdaptor},
            prelude::Backend, tensor::backend::AutodiffBackend,
        };
        use crate::{
            distributions::DistribnutionKind, policies::Policy,
            thread_safe_sequential::ThreadSafeSequential,
        };
        struct ParallModel {}
        pub struct ParalellActorCritic<B: AutodiffBackend> {
            distribution: DistribnutionKind<B>,
            value_net: ThreadSafeSequential<B>,
        }
        impl<B: AutodiffBackend> Policy<B> for ParalellActorCritic<B> {
            type Dist = DistribnutionKind<B>;
            fn distribution(&self) -> &Self::Dist {
                &self.distribution
            }
            fn update(
                &mut self,
                policy_loss: burn::prelude::Tensor<B, 2>,
                value_loss: burn::prelude::Tensor<B, 2>,
            ) {
                let loss = policy_loss + value_loss;
            }
        }
    }
    use burn::{prelude::Backend, tensor::Tensor};
    use crate::distributions::Distribution;
    pub trait Policy<B: Backend> {
        type Dist: Distribution<B>;
        fn distribution(&self) -> &Self::Dist;
        fn update(&mut self, policy_loss: Tensor<B, 2>, value_loss: Tensor<B, 2>);
    }
}
pub mod thread_safe_sequential {
    use burn::module::{AutodiffModule, ModuleMapper, ModuleVisitor};
    use burn::{
        module::Module, nn::Linear, prelude::Backend,
        tensor::{Tensor, activation::relu, backend::AutodiffBackend, module::linear},
    };
    pub struct ThreadSafeLinear<B: Backend> {
        pub weight: Tensor<B, 2>,
        pub bias: Option<Tensor<B, 1>>,
    }
    impl<B: Backend> burn::module::Module<B> for ThreadSafeLinear<B> {
        type Record = ThreadSafeLinearRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                weight: burn::module::Module::<
                    B,
                >::load_record(self.weight, record.weight),
                bias: burn::module::Module::<B>::load_record(self.bias, record.bias),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                weight: burn::module::Module::<B>::into_record(self.weight),
                bias: burn::module::Module::<B>::into_record(self.bias),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.weight);
            num_params += burn::module::Module::<B>::num_params(&self.bias);
            num_params
        }
        fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
            burn::module::Module::visit(&self.weight, visitor);
            burn::module::Module::visit(&self.bias, visitor);
        }
        fn map<Mapper: burn::module::ModuleMapper<B>>(
            self,
            mapper: &mut Mapper,
        ) -> Self {
            let weight = burn::module::Module::<B>::map(self.weight, mapper);
            let bias = burn::module::Module::<B>::map(self.bias, mapper);
            Self { weight, bias }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.weight, devices);
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.bias, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let weight = burn::module::Module::<B>::to_device(self.weight, device);
            let bias = burn::module::Module::<B>::to_device(self.bias, device);
            Self { weight, bias }
        }
        fn fork(self, device: &B::Device) -> Self {
            let weight = burn::module::Module::<B>::fork(self.weight, device);
            let bias = burn::module::Module::<B>::fork(self.bias, device);
            Self { weight, bias }
        }
    }
    impl<B: Backend> burn::module::AutodiffModule<B> for ThreadSafeLinear<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: Backend,
    {
        type InnerModule = ThreadSafeLinear<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let weight = burn::module::AutodiffModule::<B>::valid(&self.weight);
            let bias = burn::module::AutodiffModule::<B>::valid(&self.bias);
            Self::InnerModule { weight, bias }
        }
    }
    impl<B: Backend> core::fmt::Display for ThreadSafeLinear<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let formatted = burn::module::ModuleDisplay::format(
                self,
                Default::default(),
            );
            f.write_fmt(format_args!("{0}", formatted))
        }
    }
    impl<B: Backend> burn::module::ModuleDisplayDefault for ThreadSafeLinear<B> {
        fn content(
            &self,
            mut content: burn::module::Content,
        ) -> Option<burn::module::Content> {
            content
                .set_top_level_type(&"ThreadSafeLinear")
                .add("weight", &self.weight)
                .add("bias", &self.bias)
                .optional()
        }
        fn num_params(&self) -> usize {
            burn::module::Module::num_params(self)
        }
    }
    impl<B: Backend> Clone for ThreadSafeLinear<B> {
        fn clone(&self) -> Self {
            let weight = self.weight.clone();
            let bias = self.bias.clone();
            Self { weight, bias }
        }
    }
    /// The record type for the module.
    pub struct ThreadSafeLinearRecord<B: Backend> {
        /// The module record associative type.
        pub weight: <Tensor<B, 2> as burn::module::Module<B>>::Record,
        /// The module record associative type.
        pub bias: <Option<Tensor<B, 1>> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(crate = "burn::serde")]
    #[serde(
        bound = "< < Tensor < B, 2 > as burn :: module :: Module < B > > :: Record as burn ::\nrecord :: Record < B >> :: Item < S > : burn :: serde :: Serialize + burn ::\nserde :: de :: DeserializeOwned, < < Option < Tensor < B, 1 > > as burn ::\nmodule :: Module < B > > :: Record as burn :: record :: Record < B >> :: Item\n< S > : burn :: serde :: Serialize + burn :: serde :: de :: DeserializeOwned,"
    )]
    pub struct ThreadSafeLinearRecordItem<
        B: Backend,
        S: burn::record::PrecisionSettings,
    > {
        /// Field to be serialized.
        pub weight: <<Tensor<
            B,
            2,
        > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
        /// Field to be serialized.
        pub bias: <<Option<
            Tensor<B, 1>,
        > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        use burn::serde as _serde;
        #[automatically_derived]
        impl<B: Backend, S: burn::record::PrecisionSettings> burn::serde::Serialize
        for ThreadSafeLinearRecordItem<B, S>
        where
            <<Tensor<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            <<Option<
                Tensor<B, 1>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> burn::serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: burn::serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "ThreadSafeLinearRecordItem",
                    false as usize + 1 + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "weight",
                    &self.weight,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "bias",
                    &self.bias,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        use burn::serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: Backend,
            S: burn::record::PrecisionSettings,
        > burn::serde::Deserialize<'de> for ThreadSafeLinearRecordItem<B, S>
        where
            <<Tensor<
                B,
                2,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            <<Option<
                Tensor<B, 1>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> burn::serde::__private::Result<Self, __D::Error>
            where
                __D: burn::serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "weight" => _serde::__private::Ok(__Field::__field0),
                            "bias" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"weight" => _serde::__private::Ok(__Field::__field0),
                            b"bias" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<'de, B: Backend, S: burn::record::PrecisionSettings>
                where
                    <<Tensor<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    <<Option<
                        Tensor<B, 1>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<
                        ThreadSafeLinearRecordItem<B, S>,
                    >,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<
                    'de,
                    B: Backend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Tensor<
                        B,
                        2,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    <<Option<
                        Tensor<B, 1>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                {
                    type Value = ThreadSafeLinearRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct ThreadSafeLinearRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Tensor<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record<B>>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct ThreadSafeLinearRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        let __field1 = match _serde::de::SeqAccess::next_element::<
                            <<Option<
                                Tensor<B, 1>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record<B>>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        1usize,
                                        &"struct ThreadSafeLinearRecordItem with 2 elements",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(ThreadSafeLinearRecordItem {
                            weight: __field0,
                            bias: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Tensor<
                                B,
                                2,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record<B>>::Item<S>,
                        > = _serde::__private::None;
                        let mut __field1: _serde::__private::Option<
                            <<Option<
                                Tensor<B, 1>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record<B>>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key) = _serde::de::MapAccess::next_key::<
                            __Field,
                        >(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("weight"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Tensor<
                                                B,
                                                2,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record<B>>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("bias"),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Option<
                                                Tensor<B, 1>,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record<B>>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("weight")?
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("bias")?
                            }
                        };
                        _serde::__private::Ok(ThreadSafeLinearRecordItem {
                            weight: __field0,
                            bias: __field1,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["weight", "bias"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "ThreadSafeLinearRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<
                            ThreadSafeLinearRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: Backend> burn::record::Record<B> for ThreadSafeLinearRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = ThreadSafeLinearRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            ThreadSafeLinearRecordItem {
                weight: burn::record::Record::<B>::into_item::<S>(self.weight),
                bias: burn::record::Record::<B>::into_item::<S>(self.bias),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(
            item: Self::Item<S>,
            device: &B::Device,
        ) -> Self {
            Self {
                weight: burn::record::Record::<B>::from_item::<S>(item.weight, device),
                bias: burn::record::Record::<B>::from_item::<S>(item.bias, device),
            }
        }
    }
    impl<B: Backend> burn::module::ModuleDisplay for ThreadSafeLinear<B> {}
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for ThreadSafeLinear<B> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field2_finish(
                f,
                "ThreadSafeLinear",
                "weight",
                &self.weight,
                "bias",
                &&self.bias,
            )
        }
    }
    impl<B: Backend> ThreadSafeLinear<B> {
        pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
            linear(input, self.weight.clone(), self.bias.as_ref().map(|b| b.clone()))
        }
    }
    pub struct ReluAct;
    #[automatically_derived]
    impl ::core::fmt::Debug for ReluAct {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(f, "ReluAct")
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for ReluAct {
        #[inline]
        fn clone(&self) -> ReluAct {
            ReluAct
        }
    }
    impl<B: burn::tensor::backend::Backend> burn::module::Module<B> for ReluAct {
        type Record = burn::module::ConstantRecord;
        fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {}
        fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
            self
        }
        fn load_record(self, _record: Self::Record) -> Self {
            self
        }
        fn into_record(self) -> Self::Record {
            burn::module::ConstantRecord::new()
        }
        fn to_device(self, _: &B::Device) -> Self {
            self
        }
        fn fork(self, _: &B::Device) -> Self {
            self
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            devices
        }
    }
    impl<B: burn::tensor::backend::AutodiffBackend> burn::module::AutodiffModule<B>
    for ReluAct {
        type InnerModule = ReluAct;
        fn valid(&self) -> Self::InnerModule {
            self.clone()
        }
    }
    impl core::fmt::Display for ReluAct {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let formatted = burn::module::ModuleDisplay::format(
                self,
                Default::default(),
            );
            f.write_fmt(format_args!("{0}", formatted))
        }
    }
    impl burn::module::ModuleDisplayDefault for ReluAct {
        fn content(
            &self,
            mut content: burn::module::Content,
        ) -> Option<burn::module::Content> {
            content.set_top_level_type(&"ReluAct").optional()
        }
    }
    impl burn::module::ModuleDisplay for ReluAct {}
    enum ThreadSafeLayer<B: Backend> {
        Activation(ReluAct),
        Layer(ThreadSafeLinear<B>),
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug for ThreadSafeLayer<B> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                ThreadSafeLayer::Activation(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Activation",
                        &__self_0,
                    )
                }
                ThreadSafeLayer::Layer(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Layer",
                        &__self_0,
                    )
                }
            }
        }
    }
    impl<B: Backend> ThreadSafeLayer<B> {
        fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
            match &self {
                Self::Layer(linear) => linear.forward(t),
                Self::Activation(ReluAct) => relu(t),
            }
        }
    }
    pub struct ThreadSafeSequential<B: Backend> {
        layers: Vec<ThreadSafeLayer<B>>,
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + Backend> ::core::fmt::Debug
    for ThreadSafeSequential<B> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "ThreadSafeSequential",
                "layers",
                &&self.layers,
            )
        }
    }
    impl<B: Backend> ThreadSafeSequential<B> {
        pub fn forward(&self, mut t: Tensor<B, 2>) -> Tensor<B, 2> {
            for layer in self.layers.iter() {
                t = layer.forward(t);
            }
            t
        }
    }
    enum SequentialLayer<B: AutodiffBackend> {
        Activation(ReluAct),
        Layer(Linear<B>),
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + AutodiffBackend> ::core::fmt::Debug
    for SequentialLayer<B> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                SequentialLayer::Activation(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Activation",
                        &__self_0,
                    )
                }
                SequentialLayer::Layer(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Layer",
                        &__self_0,
                    )
                }
            }
        }
    }
    impl<B: AutodiffBackend> burn::module::Module<B> for SequentialLayer<B> {
        type Record = SequentialLayerRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            match self {
                Self::Activation(module) => {
                    let Self::Record::Activation(r) = record else {
                        {
                            ::core::panicking::panic_fmt(
                                format_args!("Can\'t parse record from a different variant"),
                            );
                        };
                    };
                    Self::Activation(burn::module::Module::<B>::load_record(module, r))
                }
                Self::Layer(module) => {
                    let Self::Record::Layer(r) = record else {
                        {
                            ::core::panicking::panic_fmt(
                                format_args!("Can\'t parse record from a different variant"),
                            );
                        };
                    };
                    Self::Layer(burn::module::Module::<B>::load_record(module, r))
                }
            }
        }
        fn into_record(self) -> Self::Record {
            match self {
                Self::Activation(module) => {
                    Self::Record::Activation(
                        burn::module::Module::<B>::into_record(module),
                    )
                }
                Self::Layer(module) => {
                    Self::Record::Layer(burn::module::Module::<B>::into_record(module))
                }
            }
        }
        fn num_params(&self) -> usize {
            match self {
                Self::Activation(module) => burn::module::Module::<B>::num_params(module),
                Self::Layer(module) => burn::module::Module::<B>::num_params(module),
            }
        }
        fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
            match self {
                Self::Activation(module) => burn::module::Module::visit(module, visitor),
                Self::Layer(module) => burn::module::Module::visit(module, visitor),
            }
        }
        fn map<Mapper: burn::module::ModuleMapper<B>>(
            self,
            mapper: &mut Mapper,
        ) -> Self {
            match self {
                Self::Activation(module) => {
                    Self::Activation(burn::module::Module::<B>::map(module, mapper))
                }
                Self::Layer(module) => {
                    Self::Layer(burn::module::Module::<B>::map(module, mapper))
                }
            }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            match self {
                Self::Activation(module) => {
                    burn::module::Module::<B>::collect_devices(module, devices)
                }
                Self::Layer(module) => {
                    burn::module::Module::<B>::collect_devices(module, devices)
                }
            }
        }
        fn to_device(self, device: &B::Device) -> Self {
            match self {
                Self::Activation(module) => {
                    Self::Activation(
                        burn::module::Module::<B>::to_device(module, device),
                    )
                }
                Self::Layer(module) => {
                    Self::Layer(burn::module::Module::<B>::to_device(module, device))
                }
            }
        }
        fn fork(self, device: &B::Device) -> Self {
            match self {
                Self::Activation(module) => {
                    Self::Activation(burn::module::Module::<B>::fork(module, device))
                }
                Self::Layer(module) => {
                    Self::Layer(burn::module::Module::<B>::fork(module, device))
                }
            }
        }
    }
    impl<B: AutodiffBackend> burn::module::AutodiffModule<B> for SequentialLayer<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: AutodiffBackend,
    {
        type InnerModule = SequentialLayer<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            match self {
                Self::Activation(module) => {
                    Self::InnerModule::Activation(
                        burn::module::AutodiffModule::<B>::valid(module),
                    )
                }
                Self::Layer(module) => {
                    Self::InnerModule::Layer(
                        burn::module::AutodiffModule::<B>::valid(module),
                    )
                }
            }
        }
    }
    impl<B: AutodiffBackend> core::fmt::Display for SequentialLayer<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let formatted = burn::module::ModuleDisplay::format(
                self,
                Default::default(),
            );
            f.write_fmt(format_args!("{0}", formatted))
        }
    }
    impl<B: AutodiffBackend> burn::module::ModuleDisplayDefault for SequentialLayer<B> {
        fn content(
            &self,
            mut content: burn::module::Content,
        ) -> Option<burn::module::Content> {
            match self {
                Self::Activation(_0) => {
                    content.set_top_level_type(&"Activation").add("_0", _0).optional()
                }
                Self::Layer(_0) => {
                    content.set_top_level_type(&"Layer").add("_0", _0).optional()
                }
            }
        }
        fn num_params(&self) -> usize {
            burn::module::Module::num_params(self)
        }
    }
    impl<B: AutodiffBackend> Clone for SequentialLayer<B> {
        fn clone(&self) -> Self {
            match self {
                Self::Activation(module) => Self::Activation(module.clone()),
                Self::Layer(module) => Self::Layer(module.clone()),
            }
        }
    }
    /// The record type for the module.
    enum SequentialLayerRecord<B: AutodiffBackend> {
        /// The module record associative type.
        Activation(<ReluAct as burn::module::Module<B>>::Record),
        /// The module record associative type.
        Layer(<Linear<B> as burn::module::Module<B>>::Record),
    }
    /// The record item type for the module.
    #[serde(crate = "burn::serde")]
    #[serde(
        bound = "< < ReluAct as burn :: module :: Module < B > > :: Record as burn :: record ::\nRecord < B >> :: Item < S > : burn :: serde :: Serialize + burn :: serde :: de\n:: DeserializeOwned, < < Linear < B > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record < B >> :: Item < S > : burn :: serde ::\nSerialize + burn :: serde :: de :: DeserializeOwned,"
    )]
    enum SequentialLayerRecordItem<
        B: AutodiffBackend,
        S: burn::record::PrecisionSettings,
    > {
        /// Variant to be serialized.
        Activation(
            <<ReluAct as burn::module::Module<
                B,
            >>::Record as burn::record::Record<B>>::Item<S>,
        ),
        /// Variant to be serialized.
        Layer(
            <<Linear<
                B,
            > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
        ),
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        use burn::serde as _serde;
        #[automatically_derived]
        impl<
            B: AutodiffBackend,
            S: burn::record::PrecisionSettings,
        > burn::serde::Serialize for SequentialLayerRecordItem<B, S>
        where
            <<ReluAct as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> burn::serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: burn::serde::Serializer,
            {
                match *self {
                    SequentialLayerRecordItem::Activation(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "SequentialLayerRecordItem",
                            0u32,
                            "Activation",
                            __field0,
                        )
                    }
                    SequentialLayerRecordItem::Layer(ref __field0) => {
                        _serde::Serializer::serialize_newtype_variant(
                            __serializer,
                            "SequentialLayerRecordItem",
                            1u32,
                            "Layer",
                            __field0,
                        )
                    }
                }
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        use burn::serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: AutodiffBackend,
            S: burn::record::PrecisionSettings,
        > burn::serde::Deserialize<'de> for SequentialLayerRecordItem<B, S>
        where
            <<ReluAct as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
            <<Linear<
                B,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> burn::serde::__private::Result<Self, __D::Error>
            where
                __D: burn::serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __field1,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "variant identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => {
                                _serde::__private::Err(
                                    _serde::de::Error::invalid_value(
                                        _serde::de::Unexpected::Unsigned(__value),
                                        &"variant index 0 <= i < 2",
                                    ),
                                )
                            }
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "Activation" => _serde::__private::Ok(__Field::__field0),
                            "Layer" => _serde::__private::Ok(__Field::__field1),
                            _ => {
                                _serde::__private::Err(
                                    _serde::de::Error::unknown_variant(__value, VARIANTS),
                                )
                            }
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"Activation" => _serde::__private::Ok(__Field::__field0),
                            b"Layer" => _serde::__private::Ok(__Field::__field1),
                            _ => {
                                let __value = &_serde::__private::from_utf8_lossy(__value);
                                _serde::__private::Err(
                                    _serde::de::Error::unknown_variant(__value, VARIANTS),
                                )
                            }
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<
                    'de,
                    B: AutodiffBackend,
                    S: burn::record::PrecisionSettings,
                >
                where
                    <<ReluAct as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<
                        SequentialLayerRecordItem<B, S>,
                    >,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<
                    'de,
                    B: AutodiffBackend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<ReluAct as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                    <<Linear<
                        B,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                {
                    type Value = SequentialLayerRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "enum SequentialLayerRecordItem",
                        )
                    }
                    fn visit_enum<__A>(
                        self,
                        __data: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::EnumAccess<'de>,
                    {
                        match _serde::de::EnumAccess::variant(__data)? {
                            (__Field::__field0, __variant) => {
                                _serde::__private::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        <<ReluAct as burn::module::Module<
                                            B,
                                        >>::Record as burn::record::Record<B>>::Item<S>,
                                    >(__variant),
                                    SequentialLayerRecordItem::Activation,
                                )
                            }
                            (__Field::__field1, __variant) => {
                                _serde::__private::Result::map(
                                    _serde::de::VariantAccess::newtype_variant::<
                                        <<Linear<
                                            B,
                                        > as burn::module::Module<
                                            B,
                                        >>::Record as burn::record::Record<B>>::Item<S>,
                                    >(__variant),
                                    SequentialLayerRecordItem::Layer,
                                )
                            }
                        }
                    }
                }
                #[doc(hidden)]
                const VARIANTS: &'static [&'static str] = &["Activation", "Layer"];
                _serde::Deserializer::deserialize_enum(
                    __deserializer,
                    "SequentialLayerRecordItem",
                    VARIANTS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<
                            SequentialLayerRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: AutodiffBackend> burn::record::Record<B> for SequentialLayerRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = SequentialLayerRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            match self {
                Self::Activation(record) => {
                    Self::Item::Activation(
                        burn::record::Record::<B>::into_item::<S>(record),
                    )
                }
                Self::Layer(record) => {
                    Self::Item::Layer(burn::record::Record::<B>::into_item::<S>(record))
                }
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(
            item: Self::Item<S>,
            device: &B::Device,
        ) -> Self {
            match item {
                Self::Item::Activation(item) => {
                    Self::Activation(
                        burn::record::Record::<B>::from_item::<S>(item, device),
                    )
                }
                Self::Item::Layer(item) => {
                    Self::Layer(burn::record::Record::<B>::from_item::<S>(item, device))
                }
            }
        }
    }
    impl<B: AutodiffBackend> burn::module::ModuleDisplay for SequentialLayer<B> {}
    impl<B: AutodiffBackend> SequentialLayer<B> {
        fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
            match &self {
                Self::Layer(linear) => linear.forward(t),
                Self::Activation(ReluAct) => relu(t),
            }
        }
    }
    pub struct Sequential<B: AutodiffBackend> {
        layers: Vec<SequentialLayer<B>>,
    }
    #[automatically_derived]
    impl<B: ::core::fmt::Debug + AutodiffBackend> ::core::fmt::Debug for Sequential<B> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field1_finish(
                f,
                "Sequential",
                "layers",
                &&self.layers,
            )
        }
    }
    impl<B: AutodiffBackend> burn::module::Module<B> for Sequential<B> {
        type Record = SequentialRecord<B>;
        fn load_record(self, record: Self::Record) -> Self {
            Self {
                layers: burn::module::Module::<
                    B,
                >::load_record(self.layers, record.layers),
            }
        }
        fn into_record(self) -> Self::Record {
            Self::Record {
                layers: burn::module::Module::<B>::into_record(self.layers),
            }
        }
        fn num_params(&self) -> usize {
            let mut num_params = 0;
            num_params += burn::module::Module::<B>::num_params(&self.layers);
            num_params
        }
        fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
            burn::module::Module::visit(&self.layers, visitor);
        }
        fn map<Mapper: burn::module::ModuleMapper<B>>(
            self,
            mapper: &mut Mapper,
        ) -> Self {
            let layers = burn::module::Module::<B>::map(self.layers, mapper);
            Self { layers }
        }
        fn collect_devices(
            &self,
            devices: burn::module::Devices<B>,
        ) -> burn::module::Devices<B> {
            let devices = burn::module::Module::<
                B,
            >::collect_devices(&self.layers, devices);
            devices
        }
        fn to_device(self, device: &B::Device) -> Self {
            let layers = burn::module::Module::<B>::to_device(self.layers, device);
            Self { layers }
        }
        fn fork(self, device: &B::Device) -> Self {
            let layers = burn::module::Module::<B>::fork(self.layers, device);
            Self { layers }
        }
    }
    impl<B: AutodiffBackend> burn::module::AutodiffModule<B> for Sequential<B>
    where
        B: burn::tensor::backend::AutodiffBackend,
        <B as burn::tensor::backend::AutodiffBackend>::InnerBackend: AutodiffBackend,
    {
        type InnerModule = Sequential<B::InnerBackend>;
        fn valid(&self) -> Self::InnerModule {
            let layers = burn::module::AutodiffModule::<B>::valid(&self.layers);
            Self::InnerModule { layers }
        }
    }
    impl<B: AutodiffBackend> core::fmt::Display for Sequential<B> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let formatted = burn::module::ModuleDisplay::format(
                self,
                Default::default(),
            );
            f.write_fmt(format_args!("{0}", formatted))
        }
    }
    impl<B: AutodiffBackend> burn::module::ModuleDisplayDefault for Sequential<B> {
        fn content(
            &self,
            mut content: burn::module::Content,
        ) -> Option<burn::module::Content> {
            content
                .set_top_level_type(&"Sequential")
                .add("layers", &self.layers)
                .optional()
        }
        fn num_params(&self) -> usize {
            burn::module::Module::num_params(self)
        }
    }
    impl<B: AutodiffBackend> Clone for Sequential<B> {
        fn clone(&self) -> Self {
            let layers = self.layers.clone();
            Self { layers }
        }
    }
    /// The record type for the module.
    pub struct SequentialRecord<B: AutodiffBackend> {
        /// The module record associative type.
        pub layers: <Vec<SequentialLayer<B>> as burn::module::Module<B>>::Record,
    }
    /// The record item type for the module.
    #[serde(crate = "burn::serde")]
    #[serde(
        bound = "< < Vec < SequentialLayer < B > > as burn :: module :: Module < B > > ::\nRecord as burn :: record :: Record < B >> :: Item < S > : burn :: serde ::\nSerialize + burn :: serde :: de :: DeserializeOwned,"
    )]
    pub struct SequentialRecordItem<
        B: AutodiffBackend,
        S: burn::record::PrecisionSettings,
    > {
        /// Field to be serialized.
        pub layers: <<Vec<
            SequentialLayer<B>,
        > as burn::module::Module<B>>::Record as burn::record::Record<B>>::Item<S>,
    }
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        use burn::serde as _serde;
        #[automatically_derived]
        impl<
            B: AutodiffBackend,
            S: burn::record::PrecisionSettings,
        > burn::serde::Serialize for SequentialRecordItem<B, S>
        where
            <<Vec<
                SequentialLayer<B>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> burn::serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: burn::serde::Serializer,
            {
                let mut __serde_state = _serde::Serializer::serialize_struct(
                    __serializer,
                    "SequentialRecordItem",
                    false as usize + 1,
                )?;
                _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "layers",
                    &self.layers,
                )?;
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(
        non_upper_case_globals,
        unused_attributes,
        unused_qualifications,
        clippy::absolute_paths,
    )]
    const _: () = {
        use burn::serde as _serde;
        #[automatically_derived]
        impl<
            'de,
            B: AutodiffBackend,
            S: burn::record::PrecisionSettings,
        > burn::serde::Deserialize<'de> for SequentialRecordItem<B, S>
        where
            <<Vec<
                SequentialLayer<B>,
            > as burn::module::Module<
                B,
            >>::Record as burn::record::Record<
                B,
            >>::Item<S>: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
        {
            fn deserialize<__D>(
                __deserializer: __D,
            ) -> burn::serde::__private::Result<Self, __D::Error>
            where
                __D: burn::serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                #[doc(hidden)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                #[doc(hidden)]
                struct __FieldVisitor;
                #[automatically_derived]
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "field identifier",
                        )
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "layers" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"layers" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                #[automatically_derived]
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(
                            __deserializer,
                            __FieldVisitor,
                        )
                    }
                }
                #[doc(hidden)]
                struct __Visitor<
                    'de,
                    B: AutodiffBackend,
                    S: burn::record::PrecisionSettings,
                >
                where
                    <<Vec<
                        SequentialLayer<B>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                {
                    marker: _serde::__private::PhantomData<SequentialRecordItem<B, S>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                #[automatically_derived]
                impl<
                    'de,
                    B: AutodiffBackend,
                    S: burn::record::PrecisionSettings,
                > _serde::de::Visitor<'de> for __Visitor<'de, B, S>
                where
                    <<Vec<
                        SequentialLayer<B>,
                    > as burn::module::Module<
                        B,
                    >>::Record as burn::record::Record<
                        B,
                    >>::Item<
                        S,
                    >: burn::serde::Serialize + burn::serde::de::DeserializeOwned,
                {
                    type Value = SequentialRecordItem<B, S>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct SequentialRecordItem",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match _serde::de::SeqAccess::next_element::<
                            <<Vec<
                                SequentialLayer<B>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record<B>>::Item<S>,
                        >(&mut __seq)? {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(
                                    _serde::de::Error::invalid_length(
                                        0usize,
                                        &"struct SequentialRecordItem with 1 element",
                                    ),
                                );
                            }
                        };
                        _serde::__private::Ok(SequentialRecordItem {
                            layers: __field0,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<
                            <<Vec<
                                SequentialLayer<B>,
                            > as burn::module::Module<
                                B,
                            >>::Record as burn::record::Record<B>>::Item<S>,
                        > = _serde::__private::None;
                        while let _serde::__private::Some(__key) = _serde::de::MapAccess::next_key::<
                            __Field,
                        >(&mut __map)? {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field("layers"),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        _serde::de::MapAccess::next_value::<
                                            <<Vec<
                                                SequentialLayer<B>,
                                            > as burn::module::Module<
                                                B,
                                            >>::Record as burn::record::Record<B>>::Item<S>,
                                        >(&mut __map)?,
                                    );
                                }
                                _ => {
                                    let _ = _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)?;
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                _serde::__private::de::missing_field("layers")?
                            }
                        };
                        _serde::__private::Ok(SequentialRecordItem {
                            layers: __field0,
                        })
                    }
                }
                #[doc(hidden)]
                const FIELDS: &'static [&'static str] = &["layers"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "SequentialRecordItem",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<
                            SequentialRecordItem<B, S>,
                        >,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl<B: AutodiffBackend> burn::record::Record<B> for SequentialRecord<B> {
        type Item<S: burn::record::PrecisionSettings> = SequentialRecordItem<B, S>;
        fn into_item<S: burn::record::PrecisionSettings>(self) -> Self::Item<S> {
            SequentialRecordItem {
                layers: burn::record::Record::<B>::into_item::<S>(self.layers),
            }
        }
        fn from_item<S: burn::record::PrecisionSettings>(
            item: Self::Item<S>,
            device: &B::Device,
        ) -> Self {
            Self {
                layers: burn::record::Record::<B>::from_item::<S>(item.layers, device),
            }
        }
    }
    impl<B: AutodiffBackend> burn::module::ModuleDisplay for Sequential<B> {}
    impl<B: AutodiffBackend> Sequential<B> {
        pub fn forward(&self, mut t: Tensor<B, 2>) -> Tensor<B, 2> {
            for layer in self.layers.iter() {
                t = layer.forward(t);
            }
            t
        }
    }
}
pub mod utils {
    use burn::{prelude::Backend, tensor::Tensor};
    pub fn tensor_sqr<const N: usize, B: Backend>(t: Tensor<B, N>) -> Tensor<B, N> {
        t.clone() * t
    }
}
