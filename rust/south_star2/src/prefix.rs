//! Text selection over every native writer state compatible with a prefix.

use std::error::Error;
use std::fmt;

use crate::{PreparedWriter, Walker, WalkerFailure};

/// All native writer states that emit the same selected token sequence.
#[derive(Clone, Debug)]
pub struct TextPrefix {
    states: Vec<Walker>,
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
            states: vec![prepared.initial()?],
        })
    }

    /// Whether at least one represented state accepts the selected text.
    pub fn is_accepted(&self) -> bool {
        self.states.iter().any(Walker::is_accepted)
    }

    /// Number of retained native states, including distinct states with equal output histories.
    pub fn frontier_size(&self) -> usize {
        self.states.len()
    }

    /// Group equal token texts while carrying every corresponding native successor.
    pub fn choices(&self) -> Result<Vec<TextPrefixChoice>, WalkerFailure> {
        Ok(grouped_successors(&self.states)?
            .into_iter()
            .map(|(text, states)| TextPrefixChoice {
                text,
                successor: Self { states },
            })
            .collect())
    }

    /// Select one complete token text, retaining all matching successors.
    pub fn select_text(&self, text: &str) -> Result<Self, PrefixSelectionError> {
        grouped_successors(&self.states)
            .map_err(PrefixSelectionError::Walker)?
            .into_iter()
            .find_map(|(candidate, states)| (candidate == text).then_some(Self { states }))
            .ok_or_else(|| PrefixSelectionError::InvalidText(text.to_owned()))
    }
}

fn grouped_successors(states: &[Walker]) -> Result<Vec<(String, Vec<Walker>)>, WalkerFailure> {
    let mut groups: Vec<(String, Vec<Walker>)> = Vec::new();
    for state in states {
        for choice in state.choices()? {
            if let Some((_, successors)) = groups.iter_mut().find(|(text, _)| text == &choice.text)
            {
                successors.push(choice.successor);
            } else {
                groups.push((choice.text, vec![choice.successor]));
            }
        }
    }
    Ok(groups)
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
        assert_eq!(initial_choices.len(), 2);
        assert_eq!(
            initial_choices
                .iter()
                .map(|choice| (choice.text.as_str(), choice.successor.frontier_size()))
                .collect::<BTreeMap<_, _>>(),
            BTreeMap::from([("C", 2), ("O", 1)])
        );

        let grouped = selected.choices().unwrap();
        let texts = grouped
            .iter()
            .map(|choice| choice.text.as_str())
            .collect::<BTreeSet<_>>();
        let per_state = selected
            .states
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
    fn acceptance_and_live_choices_coexist_in_a_seeded_mixed_frontier() {
        let accepted = prepared(&[], &[]).initial().unwrap();
        let live = prepared(&["C"], &[]).initial().unwrap();
        // The deliberately mixed frontier tests only the prefix client's EOS/token-union law;
        // these states come from different prepared graphs and are not a reachable molecule prefix.
        let prefix = TextPrefix {
            states: vec![accepted, live],
        };
        assert!(prefix.is_accepted());
        let choices = prefix.choices().unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].text, "C");
        assert_eq!(choices[0].successor.frontier_size(), 1);
    }
}
