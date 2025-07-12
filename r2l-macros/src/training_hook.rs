use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Error, FnArg, Ident, ItemTrait, ReturnType, TraitItem, Type, TypeReference,
    parse::Parse, parse_quote, spanned::Spanned,
};

// TODO: we might not even need this at the end, one hook seems enough for now, this is mostly
// copy+pase for now
pub struct TrainingHook {
    hook_trait: ItemTrait,
    arg_types: Vec<TypeReference>,
}

impl Parse for TrainingHook {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let hook_trait: ItemTrait = input.parse()?;
        if hook_trait.generics.params.len() != 0 {
            return syn::Result::Err(Error::new(
                hook_trait.span(),
                "Training hook should be have no generics",
            ));
        }
        let call_hook = if hook_trait.items.len() == 1
            && let TraitItem::Fn(call_hook) = &hook_trait.items[0]
            && call_hook.sig.ident == "call_hook"
        {
            call_hook
        } else {
            return syn::Result::Err(Error::new(
                hook_trait.span(),
                "Hooktrait needs exactly one method called `call_hook`",
            ));
        };
        let mut arg_types = vec![];
        for arg in call_hook.sig.inputs.iter() {
            // There are no restrictions on the receiver
            // TODO: should we require this to be mut?
            if let FnArg::Receiver(_) = arg {
                continue;
            }
            // Hooks arguments are only allowed to be references.
            // TODO: this seems reasonable right now, but we might want to change this requirement in
            // the future to allow allow copy types, maybe?
            if let FnArg::Typed(typed_fn) = arg
                && let Type::Reference(type_reference) = typed_fn.ty.as_ref()
                && let Type::Path(type_path) = type_reference.elem.as_ref()
            {
                if arg_types.iter().any(|r: &TypeReference| {
                    let Type::Path(p) = r.elem.as_ref() else {
                        unreachable!()
                    };
                    p == type_path
                }) {
                    return syn::Result::Err(Error::new(
                        type_path.span(),
                        "All arguments of `call_hook` must be unique",
                    ));
                }
                arg_types.push(type_reference.clone());
            } else {
                return syn::Result::Err(Error::new(
                    arg.span(),
                    "All arguments of `call_hook` be references",
                ));
            }
        }
        if call_hook.default.is_some() {
            return syn::Result::Err(Error::new(
                call_hook.default.span(),
                "Default implementation for `call_hook` is not allowed",
            ));
        }
        let expected_return_type: Type = parse_quote!(candle_core::Result<bool>);
        match &call_hook.sig.output {
            ReturnType::Type(_, return_type) if return_type.as_ref() == &expected_return_type => {}
            _ => {
                return syn::Result::Err(Error::new(
                    call_hook.sig.output.span(),
                    "Return type of `call_hook` should be Result<bool>",
                ));
            }
        }
        Ok(TrainingHook {
            hook_trait,
            arg_types,
        })
    }
}

fn build_implementation(
    trait_name: &Ident,                 // the name of the trait
    into_trait_name: &Ident,            // the name of the into trait
    call_hook_args: &TokenStream,       // the arguments names and with
    implementor_arg_types: TokenStream, // the argument types for the implementor
    implementor_arg_names: TokenStream, // the argument names for the implementor
    marker: TokenStream,                // the marker to avoid ambigous implemnations
) -> TokenStream {
    quote! {
        impl<F> #into_trait_name<#marker> for F
        where
            F: FnMut(#implementor_arg_types) -> candle_core::Result<bool> + Send + 'static,
        {
            fn into_boxed(self) -> Box<dyn #trait_name> {
                struct Hook<F>(F);
                impl<F> #trait_name for Hook<F>
                where
                    F: FnMut(#implementor_arg_types) -> candle_core::Result<bool>,
                {
                    fn call_hook(&mut self, #call_hook_args) -> candle_core::Result<bool> {
                        (self.0)(#implementor_arg_names)
                    }
                }
                Box::new(Hook(self))
            }
        }
    }
}

// TODO:  This generates a lot of code! There might be an easier way to implement I want via
// implementing a custom trait on a collection of types and then check if all the types implement
// that type on the implementation
fn implement_all_fn_trait_permutations(
    trait_name: &Ident,
    into_trait_name: &Ident,
    call_hook_args: &TokenStream,
    all_args: &[(Ident, TypeReference)],
    tokens: &mut TokenStream,
) {
    for perm_len in 0..=all_args.len() {
        let permutations = (0..all_args.len())
            .permutations(perm_len)
            .collect::<Vec<_>>();
        for perm in permutations {
            let mut implementor_arg_names = vec![];
            let mut implementor_arg_types = vec![];
            for i in perm {
                implementor_arg_names.push(all_args[i].0.clone());
                implementor_arg_types.push(all_args[i].1.clone());
            }
            let marker: TokenStream =
                quote! { fn(#(#implementor_arg_types,)*) -> candle_core::Result<bool> };
            let implementor_arg_names: TokenStream = quote! { #(#implementor_arg_names,)* };
            let implementor_arg_types: TokenStream = quote! { #(#implementor_arg_types,)* };
            let implementation = build_implementation(
                trait_name,
                into_trait_name,
                call_hook_args,
                implementor_arg_types,
                implementor_arg_names,
                marker,
            );
            tokens.extend(implementation);
        }
    }
}

impl ToTokens for TrainingHook {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.hook_trait.to_tokens(tokens);
        let trait_name = &self.hook_trait.ident;
        let into_trait_name = Ident::new(&format!("Into{trait_name}"), trait_name.span());
        tokens.extend(quote! {
            pub trait #into_trait_name<Marker> {
                fn into_boxed(self) -> Box<dyn #trait_name>;
            }
        });
        let mut call_hook_args: Vec<FnArg> = vec![];
        let mut all_args: Vec<(Ident, TypeReference)> = vec![];
        for (i, arg) in self.arg_types.iter().enumerate() {
            let arg_name = Ident::new(&format!("_{i}"), arg.span());
            all_args.push((arg_name.clone(), arg.clone()));
            let fn_arg = parse_quote!(#arg_name: #arg);
            call_hook_args.push(fn_arg);
        }
        let call_hook_args = quote! { #(#call_hook_args,)* };
        implement_all_fn_trait_permutations(
            trait_name,
            &into_trait_name,
            &call_hook_args,
            &all_args,
            tokens,
        );
    }
}

#[cfg(test)]
mod test {
    use super::TrainingHook;
    use quote::ToTokens;
    use syn::parse_quote;

    #[test]
    fn training_hook_test() {
        let hooktrait: TrainingHook = parse_quote! {
            pub trait BatchHook {
                fn call_hook(
                    &mut self,
                    thing: &usize,
                ) -> candle_core::Result<bool>;
            }
        };
        let mut token_stream = proc_macro2::TokenStream::new();
        hooktrait.to_tokens(&mut token_stream);
        println!("{token_stream}");
    }
}
