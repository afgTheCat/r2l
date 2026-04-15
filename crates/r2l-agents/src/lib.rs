pub mod on_policy_algorithms;

pub enum HookResult {
    Continue,
    Break,
}

#[macro_export]
macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            $crate::HookResult::Continue => {}
            $crate::HookResult::Break => return Ok(()),
        }
    };
}
