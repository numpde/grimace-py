//! Text selection over every native writer state compatible with a prefix.

use std::error::Error;
use std::fmt;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::{PreparedWriter, Walker, WalkerFailure};

/// All native writer states that emit the same selected token sequence.
#[derive(Clone, Debug)]
pub struct TextPrefix {
    frontier: PrefixFrontier,
}

#[derive(Clone, Debug)]
enum PrefixFrontier {
    Materialized(Arc<Vec<Walker>>),
    Deferred {
        parent: Arc<Vec<Walker>>,
        text: String,
        count: usize,
        accepted: bool,
        cached: Arc<OnceLock<Arc<Vec<Walker>>>>,
    },
}

/// One available token text and the complete frontier after selecting it.
#[derive(Clone, Debug)]
pub struct TextPrefixChoice {
    pub text: String,
    pub successor: TextPrefix,
}

impl TextPrefix {
    pub fn initial(prepared: &PreparedWriter) -> Result<Self, WalkerFailure> {
        Ok(Self {
            frontier: PrefixFrontier::Materialized(Arc::new(vec![prepared.initial()?])),
        })
    }

    /// Whether at least one represented state accepts the selected text.
    pub fn is_accepted(&self) -> bool {
        match &self.frontier {
            PrefixFrontier::Materialized(states) => states.iter().any(Walker::is_accepted),
            PrefixFrontier::Deferred { accepted, .. } => *accepted,
        }
    }

    /// Number of retained native states, including distinct states with equal output histories.
    pub fn frontier_size(&self) -> usize {
        match &self.frontier {
            PrefixFrontier::Materialized(states) => states.len(),
            PrefixFrontier::Deferred { count, .. } => *count,
        }
    }

    /// Discover every token text while deferring unselected native successors.
    pub fn choices(&self) -> Result<Vec<TextPrefixChoice>, WalkerFailure> {
        let parent = self.materialized_states()?;
        Ok(choice_summaries(&parent)?
            .into_iter()
            .map(|(text, count, accepted)| TextPrefixChoice {
                successor: Self { frontier: PrefixFrontier::Deferred {
                    parent: Arc::clone(&parent),
                    text: text.clone(),
                    count,
                    accepted,
                    cached: Arc::new(OnceLock::new()),
                } },
                text,
            })
            .collect())
    }

    /// Select one complete token text, retaining all matching successors.
    pub fn select_text(&self, text: &str) -> Result<Self, PrefixSelectionError> {
        let parent = self.materialized_states().map_err(PrefixSelectionError::Walker)?;
        let states = selected_successors(&parent, text).map_err(PrefixSelectionError::Walker)?;
        if states.is_empty() {
            return Err(PrefixSelectionError::InvalidText(text.to_owned()));
        }
        Ok(Self { frontier: PrefixFrontier::Materialized(Arc::new(states)) })
    }

    fn materialized_states(&self) -> Result<Arc<Vec<Walker>>, WalkerFailure> {
        match &self.frontier {
            PrefixFrontier::Materialized(states) => Ok(Arc::clone(states)),
            PrefixFrontier::Deferred { parent, text, count, cached, .. } => {
                if let Some(states) = cached.get() {
                    return Ok(Arc::clone(states));
                }
                let states = selected_successors(parent, text)?;
                assert_eq!(states.len(), *count, "selected token discovery and materialization must agree");
                let states = Arc::new(states);
                let _ = cached.set(Arc::clone(&states));
                Ok(states)
            }
        }
    }
}

fn choice_summaries(states: &[Walker]) -> Result<Vec<(String, usize, bool)>, WalkerFailure> {
    let mut groups: Vec<(String, usize, bool)> = Vec::new();
    let mut positions = HashMap::<String, usize>::new();
    for state in states {
        for choice in state.choices()? {
            if let Some(&position) = positions.get(&choice.text) {
                groups[position].1 += 1;
                groups[position].2 |= choice.successor.is_accepted();
            } else {
                positions.insert(choice.text.clone(), groups.len());
                groups.push((choice.text, 1, choice.successor.is_accepted()));
            }
        }
    }
    Ok(groups)
}

fn selected_successors(states: &[Walker], text: &str) -> Result<Vec<Walker>, WalkerFailure> {
    let mut selected = Vec::new();
    for state in states {
        for choice in state.choices()? {
            if choice.text == text {
                selected.push(choice.successor);
            }
        }
    }
    Ok(selected)
}

#[derive(Debug)]
pub enum PrefixSelectionError {
    InvalidText(String),
    Walker(WalkerFailure),
}

impl fmt::Display for PrefixSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidText(text) => {
                write!(formatter, "No writer choice has token text {text:?}")
            }
            Self::Walker(error) => write!(formatter, "Cannot select writer token text: {error}"),
        }
    }
}

impl Error for PrefixSelectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidText(_) => None,
            Self::Walker(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AtomToken, BondToken, PreparedGraphBuilder};
    use std::collections::{BTreeMap, BTreeSet};

    fn eager_reference(states: &[Walker]) -> Vec<(String, Vec<Walker>)> {
        let mut groups: Vec<(String, Vec<Walker>)> = Vec::new();
        for state in states {
            for choice in state.choices().unwrap() {
                if let Some((_, successors)) = groups.iter_mut().find(|(text, _)| *text == choice.text) {
                    successors.push(choice.successor);
                } else {
                    groups.push((choice.text, vec![choice.successor]));
                }
            }
        }
        groups
    }

    fn prepared(tokens: &[&str], edges: &[(usize, usize)]) -> PreparedWriter {
        let mut graph = PreparedGraphBuilder::new();
        let atoms = tokens
            .iter()
            .map(|_| graph.add_atom().unwrap())
            .collect::<Vec<_>>();
        for &(left, right) in edges {
            graph.add_bond(atoms[left], atoms[right]).unwrap();
        }
        PreparedWriter::new(
            graph.build(),
            tokens
                .iter()
                .map(|text| AtomToken::Fixed((*text).to_owned()))
                .collect(),
            vec![BondToken::Elided; edges.len()],
            Vec::new(),
            crate::WriterRegime::Generic {
                component_order: crate::ComponentOrder::AnyOrder,
            },
        )
        .unwrap()
    }

    #[test]
    fn selecting_unrooted_cco_retains_both_carbon_roots_and_unions_next_texts() {
        let prepared = prepared(&["C", "C", "O"], &[(0, 1), (1, 2)]);
        let prefix = TextPrefix::initial(&prepared).unwrap();
        let selected = prefix.select_text("C").unwrap();
        assert_eq!(selected.frontier_size(), 2);
        assert_eq!(prefix.frontier_size(), 1);
        let initial_choices = prefix.choices().unwrap();
        assert_eq!(initial_choices.len(), 3);
        assert_eq!(
            initial_choices
                .iter()
                .map(|choice| (choice.text.as_str(), choice.successor.frontier_size()))
                .collect::<BTreeMap<_, _>>(),
            BTreeMap::from([("[", 3), ("C", 2), ("O", 1)])
        );

        let grouped = selected.choices().unwrap();
        let texts = grouped
            .iter()
            .map(|choice| choice.text.as_str())
            .collect::<BTreeSet<_>>();
        let per_state = selected
            .materialized_states().unwrap()
            .iter()
            .map(|state| {
                let mut counts = BTreeMap::new();
                for choice in state.choices().unwrap() {
                    *counts.entry(choice.text).or_insert(0usize) += 1;
                }
                counts
            })
            .collect::<Vec<_>>();
        let mut expected = BTreeMap::new();
        for counts in &per_state {
            for (text, count) in counts {
                *expected.entry(text.clone()).or_insert(0usize) += count;
            }
        }
        let actual = grouped
            .iter()
            .map(|choice| (choice.text.clone(), choice.successor.frontier_size()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(texts, expected.keys().map(String::as_str).collect());
        assert_eq!(actual, expected);
        assert_ne!(per_state[0], per_state[1]);
    }

    #[test]
    fn invalid_token_text_is_reported_without_changing_the_prefix() {
        let prefix = TextPrefix::initial(&prepared(&["C"], &[])).unwrap();
        let error = prefix.select_text("N").unwrap_err();
        assert!(matches!(error, PrefixSelectionError::InvalidText(text) if text == "N"));
        assert_eq!(prefix.frontier_size(), 1);
    }

    #[test]
    fn deferred_siblings_match_direct_selection_after_either_is_explored() {
        let prefix = TextPrefix::initial(&prepared(&["C", "C", "O"], &[(0, 1), (1, 2)])).unwrap();
        let choices = prefix.choices().unwrap();
        let eager = eager_reference(&prefix.materialized_states().unwrap());
        assert_eq!(
            choices.iter().map(|choice| (choice.text.as_str(), choice.successor.frontier_size(), choice.successor.is_accepted())).collect::<Vec<_>>(),
            eager.iter().map(|(text, states)| (text.as_str(), states.len(), states.iter().any(Walker::is_accepted))).collect::<Vec<_>>(),
        );
        let carbon = choices.iter().find(|choice| choice.text == "C").unwrap();
        let oxygen = choices.iter().find(|choice| choice.text == "O").unwrap();
        assert_eq!(carbon.successor.frontier_size(), 2);
        assert_eq!(oxygen.successor.frontier_size(), 1);

        let signature = |prefix: &TextPrefix| {
            prefix.choices().unwrap().into_iter()
                .map(|choice| (choice.text, choice.successor.frontier_size(), choice.successor.is_accepted()))
                .collect::<Vec<_>>()
        };
        assert_eq!(signature(&carbon.successor), signature(&prefix.select_text("C").unwrap()));
        assert_eq!(signature(&oxygen.successor), signature(&prefix.select_text("O").unwrap()));
        for sibling in [carbon, oxygen] {
            let eager_states = &eager.iter().find(|(text, _)| text == &sibling.text).unwrap().1;
            let eager_next = eager_reference(eager_states);
            assert_eq!(
                signature(&sibling.successor),
                eager_next.iter().map(|(text, states)| (text.clone(), states.len(), states.iter().any(Walker::is_accepted))).collect::<Vec<_>>(),
            );
        }
        assert_eq!(signature(&carbon.successor), signature(&carbon.successor));

        let terminal = TextPrefix::initial(&prepared(&["C"], &[])).unwrap();
        let deferred = terminal.choices().unwrap().into_iter().find(|choice| choice.text == "C").unwrap();
        assert!(deferred.successor.is_accepted());
        assert_eq!(deferred.successor.frontier_size(), terminal.select_text("C").unwrap().frontier_size());
    }

    #[test]
    fn acceptance_and_live_choices_coexist_in_a_seeded_mixed_frontier() {
        let accepted = prepared(&[], &[]).initial().unwrap();
        let live = prepared(&["C"], &[]).initial().unwrap();
        // The deliberately mixed frontier tests only the prefix client's EOS/token-union law;
        // these states come from different prepared graphs and are not a reachable molecule prefix.
        let prefix = TextPrefix {
            frontier: PrefixFrontier::Materialized(Arc::new(vec![accepted, live])),
        };
        assert!(prefix.is_accepted());
        let choices = prefix.choices().unwrap();
        assert_eq!(
            choices
                .iter()
                .map(|choice| (choice.text.as_str(), choice.successor.frontier_size()))
                .collect::<BTreeMap<_, _>>(),
            BTreeMap::from([("[", 1), ("C", 1)])
        );
    }
}

#[cfg(test)]
#[path = "support_growth_tests.rs"]
mod support_growth_tests;
