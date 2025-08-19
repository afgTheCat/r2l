pub mod hooks;
pub mod ppo3;

// This changes the contorlflow, returning on hook break
macro_rules! process_hook_result {
    ($hook_res:expr) => {
        match $hook_res? {
            HookResult::Continue => {}
            HookResult::Break => return Ok(()),
        }
    };
}
