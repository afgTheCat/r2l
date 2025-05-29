use hook::HookTrait;
use quote::quote;
use syn::parse_macro_input;

mod hook;

#[proc_macro_attribute]
pub fn rlhook(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let hook = parse_macro_input! { item as HookTrait };
    quote! { #hook }.into()
}
