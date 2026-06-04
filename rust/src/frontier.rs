use std::collections::{BTreeMap, BTreeSet};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::{PyErr, PyResult};

#[derive(Clone)]
pub(crate) struct DecoderChoice<S> {
    pub(crate) text: String,
    pub(crate) successors: Vec<S>,
}

impl<S> DecoderChoice<S> {
    pub(crate) fn single(text: String, successor: S) -> Self {
        Self {
            text,
            successors: vec![successor],
        }
    }
}

pub(crate) struct GroupedTransition<S> {
    pub(crate) text: String,
    pub(crate) branch_count: usize,
    pub(crate) successors: Vec<S>,
}

pub(crate) fn group_decoder_choices<S>(
    choices: Vec<DecoderChoice<S>>,
    mut dedup_successors: impl FnMut(Vec<S>) -> Vec<S>,
) -> Vec<GroupedTransition<S>> {
    let mut buckets = BTreeMap::<String, (usize, Vec<S>)>::new();
    for choice in choices {
        let (branch_count, successors) = buckets.entry(choice.text).or_default();
        *branch_count += 1;
        successors.extend(choice.successors);
    }
    buckets
        .into_iter()
        .map(|(text, (branch_count, successors))| GroupedTransition {
            text,
            branch_count,
            successors: dedup_successors(successors),
        })
        .collect()
}

pub(crate) fn decoder_choices_from_token_successors<S>(
    successors_by_token: BTreeMap<String, Vec<S>>,
) -> Vec<DecoderChoice<S>> {
    let mut choices = Vec::new();
    extend_decoder_choices_from_token_successors(&mut choices, successors_by_token);
    choices
}

pub(crate) fn extend_decoder_choices_from_token_successors<S>(
    choices: &mut Vec<DecoderChoice<S>>,
    successors_by_token: BTreeMap<String, Vec<S>>,
) {
    for (token, successors) in successors_by_token {
        for successor in successors {
            choices.push(DecoderChoice::single(token.clone(), successor));
        }
    }
}

pub(crate) fn token_support_from_choices<S>(choices: &[DecoderChoice<S>]) -> Vec<String> {
    choices
        .iter()
        .map(|choice| choice.text.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn frontier_prefix<S>(frontier: &[S], prefix_of: impl Fn(&S) -> &str) -> String {
    let Some(first) = frontier.first() else {
        return String::new();
    };
    let prefix = prefix_of(first);
    debug_assert!(
        frontier.iter().all(|state| prefix_of(state) == prefix),
        "decoder frontiers must be prefix-homogeneous"
    );
    prefix.to_owned()
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

pub(crate) fn branch_choice_texts<S>(choices: &[DecoderChoice<S>]) -> Vec<String> {
    choices.iter().map(|choice| choice.text.clone()).collect()
}

pub(crate) fn take_choice_index_or_err<T>(mut choices: Vec<T>, chosen_idx: usize) -> PyResult<T> {
    if chosen_idx >= choices.len() {
        return Err(PyKeyError::new_err(format!(
            "Choice index {chosen_idx} is not available; choice_count={}",
            choices.len()
        )));
    }
    Ok(choices.remove(chosen_idx))
}

pub(crate) fn take_branch_choice_successors_or_err<S>(
    choices: Vec<DecoderChoice<S>>,
    chosen_idx: usize,
) -> PyResult<Vec<S>> {
    Ok(take_choice_index_or_err(choices, chosen_idx)?.successors)
}

pub(crate) fn take_token_successors_or_err<S>(
    mut transitions: BTreeMap<String, Vec<S>>,
    chosen_token: &str,
) -> PyResult<Vec<S>> {
    transitions.remove(chosen_token).ok_or_else(|| {
        let available = transitions.keys().cloned().collect::<Vec<_>>();
        unavailable_token_error(chosen_token, available)
    })
}

fn unavailable_token_error(chosen_token: &str, available: Vec<String>) -> PyErr {
    PyKeyError::new_err(format!(
        "Token {chosen_token:?} is not available; choices={available:?}"
    ))
}

pub(crate) fn take_grouped_transition_successors_or_err<S>(
    transitions: Vec<GroupedTransition<S>>,
    chosen_token: &str,
) -> PyResult<Vec<S>> {
    let mut available = Vec::with_capacity(transitions.len());
    for transition in transitions {
        debug_assert!(transition.branch_count > 0);
        if transition.text == chosen_token {
            if transition.successors.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Expected at least one successor state for token {chosen_token:?}, got 0"
                )));
            }
            return Ok(transition.successors);
        }
        available.push(transition.text);
    }
    Err(unavailable_token_error(chosen_token, available))
}

pub(crate) fn take_token_support_successors_or_err<S>(
    choices: Vec<DecoderChoice<S>>,
    chosen_token: &str,
) -> PyResult<Vec<S>>
where
    S: Ord,
{
    take_grouped_transition_successors_or_err(
        group_decoder_choices(choices, dedup_frontier),
        chosen_token,
    )
}

pub(crate) fn take_only_successor_or_err<S>(mut successors: Vec<S>, context: &str) -> PyResult<S> {
    if successors.len() != 1 {
        return Err(PyValueError::new_err(format!(
            "Expected exactly one successor state for {context}, got {}",
            successors.len()
        )));
    }
    Ok(successors.remove(0))
}

pub(crate) fn take_first_successor_or_err<S>(successors: Vec<S>, context: &str) -> PyResult<S> {
    match successors.into_iter().next() {
        Some(successor) => Ok(successor),
        None => Err(PyValueError::new_err(format!(
            "Expected at least one successor state for {context}, got 0"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        decoder_choices_from_token_successors, frontier_prefix, group_decoder_choices,
        take_choice_index_or_err, take_first_successor_or_err,
        take_grouped_transition_successors_or_err, take_only_successor_or_err,
        take_token_support_successors_or_err, DecoderChoice,
    };

    fn choice(text: &str, successors: Vec<i32>) -> DecoderChoice<i32> {
        DecoderChoice {
            text: text.to_owned(),
            successors,
        }
    }

    #[test]
    fn grouped_transitions_are_ordered_by_token_text() {
        let grouped = group_decoder_choices(
            vec![
                choice("C", vec![1]),
                choice("B", vec![2]),
                choice("A", vec![3]),
            ],
            |successors| successors,
        );

        assert_eq!(
            grouped
                .iter()
                .map(|transition| transition.text.as_str())
                .collect::<Vec<_>>(),
            vec!["A", "B", "C"]
        );
    }

    #[test]
    fn grouped_transitions_count_branches_before_successor_deduplication() {
        let grouped = group_decoder_choices(
            vec![
                choice("C", vec![1, 2]),
                choice("C", vec![2, 3]),
                choice("O", vec![4]),
            ],
            |mut successors| {
                successors.sort();
                successors.dedup();
                successors
            },
        );

        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].text, "C");
        assert_eq!(grouped[0].branch_count, 2);
        assert_eq!(grouped[0].successors, vec![1, 2, 3]);
        assert_eq!(grouped[1].text, "O");
        assert_eq!(grouped[1].branch_count, 1);
        assert_eq!(grouped[1].successors, vec![4]);
    }

    #[test]
    fn grouped_transitions_use_the_injected_successor_policy() {
        #[derive(Debug, PartialEq, Eq)]
        struct NonOrd(i32);

        let grouped = group_decoder_choices(
            vec![DecoderChoice {
                text: "C".to_owned(),
                successors: vec![NonOrd(3), NonOrd(1), NonOrd(3), NonOrd(2)],
            }],
            |successors| {
                let mut out = Vec::new();
                for successor in successors {
                    if !out
                        .iter()
                        .any(|existing: &NonOrd| existing.0 == successor.0)
                    {
                        out.push(successor);
                    }
                }
                out
            },
        );

        assert_eq!(grouped[0].branch_count, 1);
        assert_eq!(grouped[0].successors, vec![NonOrd(3), NonOrd(1), NonOrd(2)]);
    }

    #[test]
    fn grouped_transitions_accept_empty_input() {
        let grouped =
            group_decoder_choices(Vec::<DecoderChoice<i32>>::new(), |successors| successors);

        assert!(grouped.is_empty());
    }

    #[test]
    fn token_successors_flatten_to_branch_preserving_choices() {
        let choices = decoder_choices_from_token_successors(BTreeMap::from([
            ("C".to_owned(), vec![1, 2]),
            ("O".to_owned(), vec![3]),
        ]));

        assert_eq!(
            choices
                .iter()
                .map(|choice| (choice.text.as_str(), choice.successors.as_slice()))
                .collect::<Vec<_>>(),
            vec![("C", &[1][..]), ("C", &[2][..]), ("O", &[3][..])]
        );
    }

    #[test]
    fn frontier_prefix_accepts_empty_or_homogeneous_frontiers() {
        assert_eq!(frontier_prefix::<&str>(&[], |prefix| prefix), "");
        assert_eq!(frontier_prefix(&["CC", "CC"], |prefix| prefix), "CC");
    }

    #[test]
    #[should_panic(expected = "decoder frontiers must be prefix-homogeneous")]
    fn frontier_prefix_rejects_mixed_prefix_frontiers_in_debug_builds() {
        let _ = frontier_prefix(&["C", "O"], |prefix| prefix);
    }

    #[test]
    fn take_token_support_successors_merges_and_rejects_empty_successors() {
        let choices = vec![
            choice("C", vec![2, 1]),
            choice("O", vec![4]),
            choice("C", vec![2, 3]),
        ];
        assert_eq!(
            take_token_support_successors_or_err(choices, "C").unwrap(),
            vec![1, 2, 3]
        );

        assert!(take_token_support_successors_or_err(vec![choice("C", Vec::new())], "C").is_err());
    }

    #[test]
    fn take_choice_index_consumes_the_selected_item_and_reports_count() {
        assert_eq!(
            take_choice_index_or_err(vec!["A", "B", "C"], 1).unwrap(),
            "B"
        );

        assert!(take_choice_index_or_err(vec!["A"], 3).is_err());
    }

    #[test]
    fn take_grouped_transition_successors_consumes_the_selected_token_and_rejects_empty_successors()
    {
        let transitions = group_decoder_choices(
            vec![choice("C", vec![1, 2]), choice("O", vec![3])],
            |successors| successors,
        );
        assert_eq!(
            take_grouped_transition_successors_or_err(transitions, "C").unwrap(),
            vec![1, 2]
        );

        let empty = group_decoder_choices(vec![choice("C", Vec::new())], |successors| successors);
        assert!(take_grouped_transition_successors_or_err(empty, "C").is_err());

        let missing = group_decoder_choices(vec![choice("C", vec![1])], |successors| successors);
        assert!(take_grouped_transition_successors_or_err(missing, "N").is_err());
    }

    #[test]
    fn take_only_successor_requires_exactly_one_successor() {
        assert_eq!(
            take_only_successor_or_err(vec![7], "test context").unwrap(),
            7
        );
        assert!(take_only_successor_or_err(Vec::<i32>::new(), "test context").is_err());
        assert!(take_only_successor_or_err(vec![1, 2], "test context").is_err());
    }

    #[test]
    fn take_first_successor_accepts_any_nonempty_successor_list() {
        assert_eq!(
            take_first_successor_or_err(vec![7, 8], "test context").unwrap(),
            7
        );
        assert!(take_first_successor_or_err(Vec::<i32>::new(), "test context").is_err());
    }
}
