use std::collections::{BTreeMap, BTreeSet};

use pyo3::exceptions::PyKeyError;
use pyo3::PyResult;

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
