//! C0629: `evaluation.simulation.static_user_simulator`, ported from
//! `google.adk.evaluation.simulation.static_user_simulator`.

use adk_events::Event;

use crate::eval_case::StaticConversation;
use crate::evaluator::Evaluator;
use crate::user_simulator::{BoxFuture, NextUserMessage, Status, UserSimulator};

/// C0629: `static_user_simulator.StaticUserSimulator` — a `UserSimulator`
/// that returns a static list of user messages.
pub struct StaticUserSimulator {
    static_conversation: StaticConversation,
    invocation_idx: usize,
}

impl StaticUserSimulator {
    pub fn new(static_conversation: StaticConversation) -> Self {
        Self {
            static_conversation,
            invocation_idx: 0,
        }
    }
}

impl UserSimulator for StaticUserSimulator {
    /// Returns the next message in the static list, or
    /// `Status::StopSignalDetected` once the list is exhausted.
    fn get_next_user_message<'a>(
        &'a mut self,
        _events: &'a [Event],
    ) -> BoxFuture<'a, Result<NextUserMessage, String>> {
        Box::pin(async move {
            let Some(invocation) = self.static_conversation.get(self.invocation_idx) else {
                return Ok(NextUserMessage {
                    status: Status::StopSignalDetected,
                    user_message: None,
                });
            };
            let next_user_content = invocation.user_content.clone();
            self.invocation_idx += 1;
            Ok(NextUserMessage {
                status: Status::Success,
                user_message: Some(next_user_content),
            })
        })
    }

    /// The `StaticUserSimulator` does not require an evaluator.
    fn get_simulation_evaluator(&self) -> Option<Box<dyn Evaluator>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_case::Invocation;
    use adk_genai::content::Content;

    fn invocation(text: &str) -> Invocation {
        Invocation {
            invocation_id: "inv".to_string(),
            user_content: Content::user_text(text),
            final_response: None,
            intermediate_data: None,
            creation_timestamp: 0.0,
            rubrics: None,
            app_details: None,
        }
    }

    #[rusty_tokio::test]
    async fn returns_each_invocation_in_order_then_stops() {
        let mut simulator =
            StaticUserSimulator::new(vec![invocation("first"), invocation("second")]);

        let first = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(first.status, Status::Success);
        assert_eq!(first.user_message, Some(Content::user_text("first")));

        let second = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(second.status, Status::Success);
        assert_eq!(second.user_message, Some(Content::user_text("second")));

        let third = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(third.status, Status::StopSignalDetected);
        assert_eq!(third.user_message, None);
    }

    #[rusty_tokio::test]
    async fn stays_stopped_after_exhausting_the_conversation() {
        let mut simulator = StaticUserSimulator::new(vec![invocation("only")]);
        simulator.get_next_user_message(&[]).await.unwrap();
        let after_exhausted = simulator.get_next_user_message(&[]).await.unwrap();
        assert_eq!(after_exhausted.status, Status::StopSignalDetected);
    }

    #[rusty_tokio::test]
    async fn stops_immediately_for_an_empty_conversation() {
        let mut simulator = StaticUserSimulator::new(vec![]);
        assert_eq!(
            simulator.get_next_user_message(&[]).await.unwrap().status,
            Status::StopSignalDetected
        );
    }

    #[test]
    fn has_no_simulation_evaluator() {
        let simulator = StaticUserSimulator::new(vec![]);
        assert!(simulator.get_simulation_evaluator().is_none());
    }
}
