//! Capability C0395: `SkillRegistry`, ported from
//! `google.adk.skills.skill_registry`.
//!
//! **Adaptation**: `search_tool_description` (a concrete method every
//! subclass inherits a `None`-returning default of) becomes a trait
//! method with the same default, matching this port's usual "class
//! attribute/method → trait method" translation (`get_skill`/
//! `search_skills` stay required, matching the source's `@abstractmethod`
//! pair).

use crate::base_tool::BoxFuture;
use crate::skills_models::{Frontmatter, Skill};

/// C0395: interface for a skill registry — dynamic, on-demand skill
/// lookup and search, as opposed to [`crate::skill_toolset::SkillToolset`]'s
/// statically-provided `skills` list.
pub trait SkillRegistry: Send + Sync {
    /// Fetches a skill from the registry by name. `Err` if no skill with
    /// that name exists.
    fn get_skill<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<Skill, String>>;

    /// Searches for skills in the registry, returning `Frontmatter` for
    /// discovery (not the full skill content).
    fn search_skills<'a>(&'a self, query: &'a str) -> BoxFuture<'a, Vec<Frontmatter>>;

    /// The description for the `search_skills` tool. Registries can
    /// override this to give the model specialized instructions on how
    /// to use their specific search capabilities.
    fn search_tool_description(&self) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubRegistry;

    impl SkillRegistry for StubRegistry {
        fn get_skill<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<Skill, String>> {
            Box::pin(async move {
                if name == "known" {
                    Ok(Skill::default())
                } else {
                    Err(format!("no such skill: {name}"))
                }
            })
        }

        fn search_skills<'a>(&'a self, _query: &'a str) -> BoxFuture<'a, Vec<Frontmatter>> {
            Box::pin(async { vec![Frontmatter::default()] })
        }
    }

    #[rusty_tokio::test]
    async fn get_skill_resolves_a_known_skill() {
        let registry = StubRegistry;
        assert!(registry.get_skill("known").await.is_ok());
    }

    #[rusty_tokio::test]
    async fn get_skill_errors_for_an_unknown_skill() {
        let registry = StubRegistry;
        assert!(registry.get_skill("missing").await.is_err());
    }

    #[rusty_tokio::test]
    async fn search_skills_returns_frontmatter() {
        let registry = StubRegistry;
        assert_eq!(registry.search_skills("query").await.len(), 1);
    }

    #[test]
    fn search_tool_description_defaults_to_none() {
        assert_eq!(StubRegistry.search_tool_description(), None);
    }
}
