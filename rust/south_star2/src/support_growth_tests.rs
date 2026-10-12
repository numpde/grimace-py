//! Complete walks of declared finite Generic lexical witnesses, not writer-parity support.
//! All bonds are elided single bonds; this comparison does not widen bond text.
//! Unrestricted Generic atom spellings have unbounded numeric leading-zero prefixes.

use super::*;
use crate::{
    AtomFacts, AtomSourceContext, AtomToken, BondToken, ComponentOrder, ElementSymbol,
    PreparedGraphBuilder, WriterRegime,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
enum Rings {
    One,
    Four,
    All110,
}

struct Family {
    brackets: bool,
    rings: Rings,
}

impl Family {
    // Prefix recognition restricts lexical spellings only; native traversal and
    // ring-number correlation still decide which continuations are possible.
    fn permits(&self, text: &str, brackets: &[String]) -> bool {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'[' => {
                    if !self.brackets {
                        return false;
                    }
                    let suffix = &text[i..];
                    if let Some(end) = suffix.find(']') {
                        if !brackets.iter().any(|s| s == &suffix[..=end]) {
                            return false;
                        }
                        i += end + 1;
                    } else {
                        return brackets.iter().any(|s| s.starts_with(suffix));
                    }
                }
                b'0'..=b'9' | b'%' => {
                    let width = if bytes[i] == b'%' { 3 } else { 1 };
                    let end = (i + width).min(bytes.len());
                    let spelling = &text[i..end];
                    let allowed = match self.rings {
                        Rings::One => spelling == "1",
                        Rings::Four => ["0", "1", "%00", "%01"].iter().any(|s| {
                            if end - i < width {
                                s.starts_with(spelling)
                            } else {
                                *s == spelling
                            }
                        }),
                        Rings::All110 => true,
                    };
                    if !allowed {
                        return false;
                    }
                    i = end;
                }
                b'C' | b'O' | b'(' | b')' => i += 1,
                _ => return false,
            }
        }
        true
    }
}

#[derive(Default, Debug)]
struct Counts {
    token_prefixes: usize,
    rendered_prefixes: BTreeSet<String>,
    frontier_states: usize,
    max_frontier: usize,
    emitted_choices: usize,
    all_visible_groups: usize,
    admitted_choices: usize,
    admitted_visible_groups: usize,
    group_multiplicities: BTreeMap<usize, usize>,
    accepted: BTreeMap<String, usize>,
}

enum WalkOutcome {
    Complete(Counts),
    Capped { token_prefixes: usize, frontier_states: usize, pending_prefixes: usize },
}

fn walk(prepared: &PreparedWriter, family: &Family, brackets: &[String]) -> WalkOutcome {
    let mut counts = Counts::default();
    let mut pending = vec![(String::new(), TextPrefix::initial(prepared).unwrap(), 0)];
    while let Some((text, prefix, depth)) = pending.pop() {
        // A capped diagnostic is incomplete; it supplies no support-equality claim.
        if counts.token_prefixes >= 100_000
            || counts.frontier_states + prefix.states.len() > 2_000_000
            || depth > 128
        {
            return WalkOutcome::Capped {
                token_prefixes: counts.token_prefixes,
                frontier_states: counts.frontier_states,
                pending_prefixes: pending.len() + 1,
            };
        }
        counts.token_prefixes += 1;
        counts.frontier_states += prefix.states.len();
        counts.rendered_prefixes.insert(text.clone());
        counts.max_frontier = counts.max_frontier.max(prefix.states.len());
        let accepting = prefix.states.iter().filter(|s| s.is_accepted()).count();
        if accepting > 0 {
            *counts.accepted.entry(text.clone()).or_default() += accepting;
        }
        let mut raw = BTreeMap::new();
        for state in &prefix.states {
            for choice in state.choices().unwrap() {
                *raw.entry(choice.text).or_insert(0usize) += 1;
            }
        }
        let groups = grouped_successors(&prefix.states).unwrap();
        let grouped_counts = groups
            .iter()
            .map(|(s, states)| (s.clone(), states.len()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            raw, grouped_counts,
            "grouping must preserve every emitted successor"
        );
        counts.emitted_choices += raw.values().sum::<usize>();
        counts.all_visible_groups += groups.len();
        for (token, states) in groups {
            let next = format!("{text}{token}");
            if family.permits(&next, brackets) {
                counts.admitted_choices += states.len();
                counts.admitted_visible_groups += 1;
                *counts.group_multiplicities.entry(states.len()).or_default() += 1;
                pending.push((next, TextPrefix { states }, depth + 1));
            }
        }
    }
    assert!(!counts.accepted.is_empty());
    assert_eq!(counts.token_prefixes, counts.admitted_visible_groups + 1);
    assert_eq!(counts.frontier_states, counts.admitted_choices + 1);
    assert_eq!(
        counts.group_multiplicities.values().sum::<usize>(),
        counts.admitted_visible_groups
    );
    assert_eq!(
        counts
            .group_multiplicities
            .iter()
            .map(|(n, groups)| n * groups)
            .sum::<usize>(),
        counts.admitted_choices
    );
    WalkOutcome::Complete(counts)
}

fn molecule(elements: &[&str], edges: &[(usize, usize)]) -> (PreparedWriter, Vec<String>) {
    let mut graph = PreparedGraphBuilder::new();
    let atoms = elements
        .iter()
        .map(|_| graph.add_atom().unwrap())
        .collect::<Vec<_>>();
    let mut degrees = vec![0; elements.len()];
    for &(a, b) in edges {
        graph.add_bond(atoms[a], atoms[b]).unwrap();
        degrees[a] += 1;
        degrees[b] += 1;
    }
    let mut brackets = Vec::new();
    let tokens = elements
        .iter()
        .enumerate()
        .map(|(i, element)| {
            let valence = match *element {
                "C" => 4,
                "O" => 2,
                _ => unreachable!(),
            };
            let h = valence - degrees[i];
            brackets.push(match h {
                0 => format!("[{element}]"),
                1 => format!("[{element}H]"),
                _ => format!("[{element}H{h}]"),
            });
            AtomToken::Facts {
                facts: AtomFacts::new(
                    ElementSymbol::new(element).unwrap(),
                    false,
                    None,
                    0,
                    h,
                    0,
                    None,
                )
                .unwrap(),
                source_context: AtomSourceContext::default(),
                tetrahedral_reference_order: None,
                at_for_even_parity: false,
            }
        })
        .collect();
    brackets.sort();
    brackets.dedup();
    (
        PreparedWriter::new(
            graph.build(),
            tokens,
            vec![BondToken::Elided; edges.len()],
            Vec::new(),
            WriterRegime::Generic {
                component_order: ComponentOrder::AnyOrder,
            },
        )
        .unwrap(),
        brackets,
    )
}

fn remove_brackets(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch == '[' {
            out.push(chars.next().unwrap());
            assert!(chars.by_ref().any(|c| c == ']'));
        } else {
            out.push(ch);
        }
    }
    out
}

#[test]
fn bounded_support_growth() {
    for (name, elements, edges) in [
        ("CO", vec!["C", "O"], vec![(0, 1)]),
        ("CCO", vec!["C", "C", "O"], vec![(0, 1), (1, 2)]),
        (
            "CC(C)O",
            vec!["C", "C", "C", "O"],
            vec![(0, 1), (1, 2), (1, 3)],
        ),
        ("C1CO1", vec!["C", "C", "O"], vec![(0, 1), (1, 2), (2, 0)]),
    ] {
        let (prepared, brackets) = molecule(&elements, &edges);
        for rings in [Rings::One, Rings::Four, Rings::All110] {
            // Acyclic graphs have no ring choices, so repeat only the relevant comparison.
            if name != "C1CO1" && !matches!(rings, Rings::One) {
                continue;
            }
            let bare = walk(
                &prepared,
                &Family {
                    brackets: false,
                    rings,
                },
                &brackets,
            );
            let expanded = walk(
                &prepared,
                &Family {
                    brackets: true,
                    rings,
                },
                &brackets,
            );
            let (bare, expanded) = match (bare, expanded) {
                (WalkOutcome::Complete(bare), WalkOutcome::Complete(expanded)) => (bare, expanded),
                (bare, expanded) => {
                    assert!(
                        name == "C1CO1" && matches!(rings, Rings::All110),
                        "small support witnesses must complete within the fixed diagnostic budget"
                    );
                    for (atoms, outcome) in [("bare", bare), ("brackets", expanded)] {
                        if let WalkOutcome::Capped { token_prefixes, frontier_states, pending_prefixes } = outcome {
                            println!("growth {name} {atoms} {rings:?}: incomplete diagnostic at fixed budget; visited_prefixes={token_prefixes} visited_states={frontier_states} pending_prefixes={pending_prefixes}");
                        }
                    }
                    continue;
                }
            };
            assert_eq!(
                bare.accepted.keys().cloned().collect::<BTreeSet<_>>(),
                expanded
                    .accepted
                    .keys()
                    .map(|s| remove_brackets(s))
                    .collect()
            );
            if name == "CCO" {
                let carbon = TextPrefix::initial(&prepared)
                    .unwrap()
                    .select_text("C")
                    .unwrap();
                assert_eq!(carbon.states.len(), 2);
                let futures = carbon
                    .states
                    .iter()
                    .map(|state| {
                        state
                            .choices()
                            .unwrap()
                            .into_iter()
                            .filter(|choice| {
                                Family {
                                    brackets: false,
                                    rings,
                                }
                                .permits(&format!("C{}", choice.text), &brackets)
                            })
                            .map(|choice| choice.text)
                            .collect::<BTreeSet<_>>()
                    })
                    .collect::<Vec<_>>();
                assert_ne!(futures[0], futures[1]);
                println!("growth CCO prefix C: two retained roots have distinct admitted next-text sets {futures:?}");
                assert_eq!(
                    bare.accepted
                        .keys()
                        .map(String::as_str)
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["CCO", "OCC", "C(C)O", "C(O)C"])
                );
            }
            if name == "C1CO1" {
                // Opening and closing lexemes must name the same numeric ring ID.
                // This is specific to this one-cycle graph, not a product law.
                let spellings = match rings {
                    Rings::One => vec![(1, "1".to_owned())],
                    Rings::Four => vec![
                        (0, "0".to_owned()),
                        (1, "1".to_owned()),
                        (0, "%00".to_owned()),
                        (1, "%01".to_owned()),
                    ],
                    Rings::All110 => (0..=99)
                        .flat_map(|id| {
                            let mut options = vec![(id, format!("%{id:02}"))];
                            if id < 10 {
                                options.push((id, id.to_string()));
                            }
                            options
                        })
                        .collect(),
                };
                let mut expected = BTreeSet::new();
                for (opening_id, opening) in &spellings {
                    for (closing_id, closing) in &spellings {
                        if opening_id == closing_id {
                            for (root, middle, last) in
                                [("C", "C", "O"), ("C", "O", "C"), ("O", "C", "C")]
                            {
                                expected.insert(format!("{root}{opening}{middle}{last}{closing}"));
                            }
                        }
                    }
                }
                assert_eq!(
                    bare.accepted.keys().cloned().collect::<BTreeSet<_>>(),
                    expected
                );
                assert!(bare.accepted.values().all(|n| *n == 2));
                assert!(expanded.accepted.values().all(|n| *n == 2));
            }
            if name == "CO" {
                assert_eq!(
                    expanded
                        .accepted
                        .keys()
                        .map(String::as_str)
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from([
                        "CO",
                        "C[OH]",
                        "[CH3]O",
                        "[CH3][OH]",
                        "OC",
                        "O[CH3]",
                        "[OH]C",
                        "[OH][CH3]",
                    ])
                );
                assert_eq!(
                    bare.accepted
                        .keys()
                        .map(String::as_str)
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["CO", "OC"])
                );
            }
            for (atoms, c) in [("bare", bare), ("brackets", expanded)] {
                println!("growth {name} {atoms} {rings:?}: strings={} rendered_prefixes={} token_prefixes={} states={} max_frontier={} emitted={} all_groups={} admitted={} visible={} multiplicities={:?} accepting_paths={} accepting_multiplicities={:?}",
                    c.accepted.len(), c.rendered_prefixes.len(), c.token_prefixes,
                    c.frontier_states, c.max_frontier, c.emitted_choices, c.all_visible_groups,
                    c.admitted_choices, c.admitted_visible_groups, c.group_multiplicities,
                    c.accepted.values().sum::<usize>(),
                    c.accepted.values().fold(BTreeMap::new(), |mut map, n| { *map.entry(*n).or_insert(0usize) += 1; map }));
            }
        }
    }
}
