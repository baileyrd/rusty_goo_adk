//! Capability C0420: `exit_loop`, ported from
//! `google.adk.tools.exit_loop_tool`.

use std::collections::BTreeMap;

use rusty_serde::value::Value;

use crate::tool_context::ToolContext;

/// C0420: exits the loop. Call this function only when instructed to do
/// so. Sets `escalate=True`+`skip_summarization=True` to break a
/// loop-type agent, matching the source exactly.
pub fn exit_loop(_args: &BTreeMap<String, Value>, tool_context: &mut ToolContext) -> Value {
    let actions = tool_context.actions_mut();
    actions.escalate = true;
    actions.skip_summarization = true;
    Value::Null
}

#[cfg(test)]
mod tests {
    use super::*;
    use adk_agents::context::Context;
    use adk_agents::invocation_context::InvocationContextBuilder;
    use adk_agents::session::Session;

    fn ctx() -> Context {
        Context::new(
            InvocationContextBuilder::new("inv-1", Session::new("app", "user", "s1")).build(),
        )
    }

    #[test]
    fn sets_escalate_and_skip_summarization() {
        let mut context = ctx();
        let result = exit_loop(&BTreeMap::new(), &mut context);
        assert_eq!(result, Value::Null);
        assert!(context.actions().escalate);
        assert!(context.actions().skip_summarization);
    }
}
