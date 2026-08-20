use std::collections::{BTreeMap, BTreeSet};

use pyo3::exceptions::PyKeyError;
use pyo3::PyResult;

#[derive(Clone)]
pub(crate) struct DecoderChoice<S> {
    pub(crate) text: String,
    pub(crate) next_frontier: Vec<S>,
}

pub(crate) fn extend_transitions<S>(
    transitions: &mut BTreeMap<String, BTreeSet<S>>,
    successors: BTreeMap<String, Vec<S>>,
) where
    S: Ord,
{
    for (token, states) in successors {
        transitions.entry(token).or_default().extend(states);
    }
}

pub(crate) fn finalize_transitions<S>(
    transitions: BTreeMap<String, BTreeSet<S>>,
) -> BTreeMap<String, Vec<S>>
where
    S: Ord,
{
    transitions
        .into_iter()
        .map(|(token, states)| (token, states.into_iter().collect()))
        .collect()
}

pub(crate) fn frontier_prefix<S>(frontier: &[S], prefix_of: impl Fn(&S) -> &str) -> String {
    frontier
        .first()
        .map(|state| prefix_of(state).to_owned())
        .unwrap_or_default()
}

pub(crate) fn dedup_frontier<S>(states: Vec<S>) -> Vec<S>
where
    S: Ord,
{
    states
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn grouped_choice_texts<S>(choices: &[DecoderChoice<S>]) -> Vec<String> {
    choices
        .iter()
        .map(|choice| choice.text.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn choice_texts<S>(choices: &[DecoderChoice<S>]) -> Vec<String> {
    choices.iter().map(|choice| choice.text.clone()).collect()
}

pub(crate) fn take_choice_or_err<S>(
    choices: &mut Vec<DecoderChoice<S>>,
    chosen_idx: usize,
) -> PyResult<Vec<S>> {
    if chosen_idx >= choices.len() {
        return Err(PyKeyError::new_err(format!(
            "Choice index {chosen_idx} is not available; choice_count={}",
            choices.len()
        )));
    }
    Ok(choices.remove(chosen_idx).next_frontier)
}

pub(crate) fn take_transition_or_err<S>(
    transitions: &mut BTreeMap<String, Vec<S>>,
    chosen_token: &str,
) -> PyResult<Vec<S>> {
    transitions.remove(chosen_token).ok_or_else(|| {
        let available = transitions.keys().cloned().collect::<Vec<_>>();
        PyKeyError::new_err(format!(
            "Token {chosen_token:?} is not available; choices={available:?}"
        ))
    })
}

pub(crate) fn take_grouped_choices_or_err<S>(
    choices: Vec<DecoderChoice<S>>,
    chosen_token: &str,
) -> PyResult<Vec<S>>
where
    S: Ord,
{
    let mut available = BTreeSet::new();
    let mut next_frontier = Vec::new();
    for choice in choices {
        available.insert(choice.text.clone());
        if choice.text == chosen_token {
            next_frontier.extend(choice.next_frontier);
        }
    }
    if next_frontier.is_empty() {
        return Err(PyKeyError::new_err(format!(
            "Token {chosen_token:?} is not available; choices={:?}",
            available.into_iter().collect::<Vec<_>>()
        )));
    }
    Ok(dedup_frontier(next_frontier))
}
