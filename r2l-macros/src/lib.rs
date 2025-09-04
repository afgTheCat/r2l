mod policy_hook;
mod training_hook;

use crate::training_hook::TrainingHook;
use policy_hook::PolicyTrait;
use quote::quote;
use syn::parse_macro_input;

#[proc_macro_attribute]
pub fn policy_hook(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let hook = parse_macro_input! { item as PolicyTrait };
    quote! { #hook }.into()
}

#[proc_macro_attribute]
pub fn training_hook(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let hook = parse_macro_input! { item as TrainingHook };
    quote! { #hook }.into()
}
