use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::{One, Zero};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

type ChoiceTerm = (String, BigUint, u32, BigUint, BigUint, BigUint);
type NodeTerm = (
    u32,
    String,
    bool,
    BigUint,
    BigUint,
    Vec<ChoiceTerm>,
    BigUint,
    BigUint,
);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ContinuationChoice {
    token_id: u32,
    immediate_multiplicity: BigUint,
    successor_node_id: u32,
    successor_scale: BigUint,
    support_count: BigUint,
    completion_count: BigUint,
}

#[derive(Clone, Debug)]
struct ContinuationNode {
    terminal_available: bool,
    terminal_multiplicity: BigUint,
    terminal_completion_count: BigUint,
    first_choice: u32,
    choice_count: u32,
    support_count: BigUint,
    completion_count: BigUint,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct SemanticSignature {
    terminal_available: bool,
    terminal_multiplicity: BigUint,
    terminal_completion_count: BigUint,
    choices: Vec<(u32, BigUint, BigUint, u32)>,
}

#[derive(Debug)]
struct ContinuationCore {
    manifest_digest: String,
    tokens: Vec<String>,
    nodes: Vec<ContinuationNode>,
    choices: Vec<ContinuationChoice>,
    root_node_id: u32,
    root_scale: BigUint,
}

#[pyclass(name = "WriterContinuationRustCore", frozen)]
pub struct PyWriterContinuationRustCore {
    core: Arc<ContinuationCore>,
}

#[pyclass(name = "WriterContinuationRustCursor", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyWriterContinuationRustCursor {
    core: Arc<ContinuationCore>,
    node_id: u32,
    completion_scale: BigUint,
}

#[pyclass(name = "WriterContinuationRustChoice", frozen)]
pub struct PyWriterContinuationRustChoice {
    text: String,
    immediate_multiplicity: BigUint,
    support_count: BigUint,
    completion_count: BigUint,
    next_cursor: PyWriterContinuationRustCursor,
}

#[pyclass(name = "WriterContinuationRustProbability", frozen)]
pub struct PyWriterContinuationRustProbability {
    text: Option<String>,
    numerator: BigUint,
    denominator: BigUint,
}

fn invalid(reason: &str) -> PyErr {
    PyValueError::new_err(reason.to_owned())
}

fn node_for_cursor<'a>(
    core: &'a Arc<ContinuationCore>,
    cursor: &PyWriterContinuationRustCursor,
) -> PyResult<&'a ContinuationNode> {
    if !Arc::ptr_eq(core, &cursor.core) {
        return Err(invalid("continuation_rust_cursor_core_mismatch"));
    }
    core.nodes
        .get(cursor.node_id as usize)
        .ok_or_else(|| invalid("continuation_rust_cursor_node_invalid"))
}

fn node_choices<'a>(
    core: &'a ContinuationCore,
    node: &ContinuationNode,
) -> &'a [ContinuationChoice] {
    let first = node.first_choice as usize;
    &core.choices[first..first + node.choice_count as usize]
}

fn validate_acyclic_and_reachable(
    nodes: &[ContinuationNode],
    choices: &[ContinuationChoice],
    root: u32,
) -> PyResult<()> {
    fn visit(
        node_id: u32,
        nodes: &[ContinuationNode],
        choices: &[ContinuationChoice],
        marks: &mut [u8],
    ) -> PyResult<()> {
        match marks[node_id as usize] {
            1 => return Err(invalid("continuation_rust_cycle")),
            2 => return Ok(()),
            _ => {}
        }
        marks[node_id as usize] = 1;
        let node = &nodes[node_id as usize];
        for choice in node_choices_raw(choices, node) {
            visit(choice.successor_node_id, nodes, choices, marks)?;
        }
        marks[node_id as usize] = 2;
        Ok(())
    }

    let mut marks = vec![0; nodes.len()];
    visit(root, nodes, choices, &mut marks)?;
    if marks.iter().any(|mark| *mark != 2) {
        return Err(invalid("continuation_rust_unreachable_node"));
    }
    Ok(())
}

fn node_choices_raw<'a>(
    choices: &'a [ContinuationChoice],
    node: &ContinuationNode,
) -> &'a [ContinuationChoice] {
    let first = node.first_choice as usize;
    &choices[first..first + node.choice_count as usize]
}

fn signature_digest(
    node: &ContinuationNode,
    choices: &[ContinuationChoice],
    tokens: &[String],
) -> String {
    let mut canonical = format!(
        "[{},{},{},[",
        node.terminal_available, node.terminal_multiplicity, node.terminal_completion_count
    );
    for (index, choice) in node_choices_raw(choices, node).iter().enumerate() {
        if index != 0 {
            canonical.push(',');
        }
        canonical.push_str(&format!(
            "[{},{},{},{}]",
            serde_json::to_string(&tokens[choice.token_id as usize])
                .expect("a Rust string always serializes as JSON"),
            choice.immediate_multiplicity,
            choice.successor_scale,
            choice.successor_node_id,
        ));
    }
    canonical.push_str("]]");
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

fn build_core(
    manifest_digest: String,
    root_node_id: u32,
    root_scale: BigUint,
    node_terms: Vec<NodeTerm>,
) -> PyResult<ContinuationCore> {
    if manifest_digest.is_empty() {
        return Err(invalid("continuation_rust_manifest_digest_empty"));
    }
    if root_scale.is_zero() {
        return Err(invalid("continuation_rust_root_scale_nonpositive"));
    }
    if node_terms.is_empty() {
        return Err(invalid("continuation_rust_nodes_empty"));
    }
    if root_node_id as usize >= node_terms.len() {
        return Err(invalid("continuation_rust_root_node_invalid"));
    }

    let mut token_set = BTreeSet::new();
    for (_, _, _, _, _, choices, _, _) in &node_terms {
        for (text, _, _, _, _, _) in choices {
            token_set.insert(text.clone());
        }
    }
    let tokens: Vec<_> = token_set.into_iter().collect();
    let token_ids: BTreeMap<_, _> = tokens
        .iter()
        .enumerate()
        .map(|(index, text)| (text.as_str(), index as u32))
        .collect();

    let node_count = node_terms.len();
    let mut nodes = Vec::with_capacity(node_count);
    let mut signature_digests = Vec::with_capacity(node_count);
    let mut choices = Vec::new();
    for (expected_id, term) in node_terms.into_iter().enumerate() {
        let (
            node_id,
            provided_signature_digest,
            terminal_available,
            terminal_multiplicity,
            terminal_completion_count,
            choice_terms,
            support_count,
            completion_count,
        ) = term;
        if node_id as usize != expected_id {
            return Err(invalid("continuation_rust_node_id_not_contiguous"));
        }
        signature_digests.push(provided_signature_digest);
        if terminal_available == terminal_multiplicity.is_zero() {
            return Err(invalid("continuation_rust_terminal_multiplicity_mismatch"));
        }
        if terminal_completion_count != terminal_multiplicity {
            return Err(invalid("continuation_rust_terminal_completion_mismatch"));
        }
        let first_choice = choices.len() as u32;
        let mut previous_text: Option<String> = None;
        for (
            text,
            immediate_multiplicity,
            successor_node_id,
            successor_scale,
            choice_support_count,
            choice_completion_count,
        ) in choice_terms
        {
            if text.is_empty() {
                return Err(invalid("continuation_rust_choice_text_empty"));
            }
            if previous_text
                .as_ref()
                .is_some_and(|previous| previous >= &text)
            {
                return Err(invalid("continuation_rust_choice_text_order_mismatch"));
            }
            if immediate_multiplicity.is_zero() || successor_scale.is_zero() {
                return Err(invalid("continuation_rust_choice_scale_nonpositive"));
            }
            if successor_node_id as usize >= node_count {
                return Err(invalid("continuation_rust_successor_node_invalid"));
            }
            let token_id = *token_ids
                .get(text.as_str())
                .ok_or_else(|| invalid("continuation_rust_token_missing"))?;
            previous_text = Some(text);
            choices.push(ContinuationChoice {
                token_id,
                immediate_multiplicity,
                successor_node_id,
                successor_scale,
                support_count: choice_support_count,
                completion_count: choice_completion_count,
            });
        }
        nodes.push(ContinuationNode {
            terminal_available,
            terminal_multiplicity,
            terminal_completion_count,
            first_choice,
            choice_count: choices.len() as u32 - first_choice,
            support_count,
            completion_count,
        });
    }

    for node in &nodes {
        let node_choices = node_choices_raw(&choices, node);
        let mut expected_support = if node.terminal_available {
            BigUint::one()
        } else {
            BigUint::zero()
        };
        let mut expected_completion = node.terminal_completion_count.clone();
        for choice in node_choices {
            let successor = &nodes[choice.successor_node_id as usize];
            if choice.support_count != successor.support_count {
                return Err(invalid("continuation_rust_choice_support_count_mismatch"));
            }
            if choice.completion_count
                != choice.successor_scale.clone() * successor.completion_count.clone()
            {
                return Err(invalid(
                    "continuation_rust_choice_completion_count_mismatch",
                ));
            }
            expected_support += &choice.support_count;
            expected_completion += &choice.completion_count;
        }
        if node.support_count != expected_support {
            return Err(invalid("continuation_rust_node_support_count_mismatch"));
        }
        if node.completion_count != expected_completion {
            return Err(invalid("continuation_rust_node_completion_count_mismatch"));
        }
    }
    validate_acyclic_and_reachable(&nodes, &choices, root_node_id)?;

    let mut signatures = HashSet::new();
    let mut previous_depth = 0;
    let mut previous_signature: Option<SemanticSignature> = None;
    let mut depths = Vec::with_capacity(nodes.len());
    for (node_id, node) in nodes.iter().enumerate() {
        if node_choices_raw(&choices, node)
            .iter()
            .any(|choice| choice.successor_node_id as usize >= node_id)
        {
            return Err(invalid("continuation_rust_canonical_child_order_mismatch"));
        }
        let depth = 1 + node_choices_raw(&choices, node)
            .iter()
            .map(|choice| depths[choice.successor_node_id as usize])
            .max()
            .unwrap_or(0);
        let signature = SemanticSignature {
            terminal_available: node.terminal_available,
            terminal_multiplicity: node.terminal_multiplicity.clone(),
            terminal_completion_count: node.terminal_completion_count.clone(),
            choices: node_choices_raw(&choices, node)
                .iter()
                .map(|choice| {
                    (
                        choice.token_id,
                        choice.immediate_multiplicity.clone(),
                        choice.successor_scale.clone(),
                        choice.successor_node_id,
                    )
                })
                .collect(),
        };
        if signature_digests[node_id] != signature_digest(node, &choices, &tokens) {
            return Err(invalid("continuation_rust_signature_digest_mismatch"));
        }
        if !signatures.insert(signature.clone()) {
            return Err(invalid("continuation_rust_duplicate_semantic_class"));
        }
        if depth < previous_depth
            || (depth == previous_depth
                && previous_signature
                    .as_ref()
                    .is_some_and(|previous| previous >= &signature))
        {
            return Err(invalid("continuation_rust_canonical_node_order_mismatch"));
        }
        previous_depth = depth;
        previous_signature = Some(signature);
        depths.push(depth);
    }

    Ok(ContinuationCore {
        manifest_digest,
        tokens,
        nodes,
        choices,
        root_node_id,
        root_scale,
    })
}

#[pyfunction]
pub fn _writer_continuation_rust_core_from_verified_terms(
    manifest_digest: String,
    root_node_id: u32,
    root_scale: BigUint,
    nodes: Vec<NodeTerm>,
) -> PyResult<PyWriterContinuationRustCore> {
    Ok(PyWriterContinuationRustCore {
        core: Arc::new(build_core(
            manifest_digest,
            root_node_id,
            root_scale,
            nodes,
        )?),
    })
}

#[pymethods]
impl PyWriterContinuationRustCore {
    #[getter]
    fn manifest_digest(&self) -> &str {
        &self.core.manifest_digest
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.core.nodes.len()
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.core.choices.len()
    }

    #[getter]
    fn resident_bytes(&self) -> usize {
        let bigint_heap_bytes = |value: &BigUint| value.bits().div_ceil(8) as usize;
        std::mem::size_of::<ContinuationCore>()
            + self.core.tokens.len() * std::mem::size_of::<String>()
            + self.core.tokens.iter().map(String::len).sum::<usize>()
            + self.core.nodes.len() * std::mem::size_of::<ContinuationNode>()
            + self.core.choices.len() * std::mem::size_of::<ContinuationChoice>()
            + bigint_heap_bytes(&self.core.root_scale)
            + self
                .core
                .nodes
                .iter()
                .map(|node| {
                    bigint_heap_bytes(&node.terminal_multiplicity)
                        + bigint_heap_bytes(&node.terminal_completion_count)
                        + bigint_heap_bytes(&node.support_count)
                        + bigint_heap_bytes(&node.completion_count)
                })
                .sum::<usize>()
            + self
                .core
                .choices
                .iter()
                .map(|choice| {
                    bigint_heap_bytes(&choice.immediate_multiplicity)
                        + bigint_heap_bytes(&choice.successor_scale)
                        + bigint_heap_bytes(&choice.support_count)
                        + bigint_heap_bytes(&choice.completion_count)
                })
                .sum::<usize>()
    }

    fn root_cursor(&self) -> PyWriterContinuationRustCursor {
        PyWriterContinuationRustCursor {
            core: Arc::clone(&self.core),
            node_id: self.core.root_node_id,
            completion_scale: self.core.root_scale.clone(),
        }
    }

    fn cursor(
        &self,
        node_id: u32,
        completion_scale: BigUint,
    ) -> PyResult<PyWriterContinuationRustCursor> {
        if completion_scale.is_zero() {
            return Err(invalid("continuation_rust_cursor_scale_nonpositive"));
        }
        if node_id as usize >= self.core.nodes.len() {
            return Err(invalid("continuation_rust_cursor_node_invalid"));
        }
        Ok(PyWriterContinuationRustCursor {
            core: Arc::clone(&self.core),
            node_id,
            completion_scale,
        })
    }

    fn choices(
        &self,
        cursor: &PyWriterContinuationRustCursor,
    ) -> PyResult<Vec<PyWriterContinuationRustChoice>> {
        let node = node_for_cursor(&self.core, cursor)?;
        Ok(node_choices(&self.core, node)
            .iter()
            .map(|choice| PyWriterContinuationRustChoice {
                text: self.core.tokens[choice.token_id as usize].clone(),
                immediate_multiplicity: cursor.completion_scale.clone()
                    * choice.immediate_multiplicity.clone(),
                support_count: choice.support_count.clone(),
                completion_count: cursor.completion_scale.clone() * choice.completion_count.clone(),
                next_cursor: PyWriterContinuationRustCursor {
                    core: Arc::clone(&self.core),
                    node_id: choice.successor_node_id,
                    completion_scale: cursor.completion_scale.clone()
                        * choice.successor_scale.clone(),
                },
            })
            .collect())
    }

    fn advance(
        &self,
        cursor: &PyWriterContinuationRustCursor,
        emitted_text: &str,
    ) -> PyResult<PyWriterContinuationRustCursor> {
        let node = node_for_cursor(&self.core, cursor)?;
        let choice = node_choices(&self.core, node)
            .iter()
            .find(|choice| self.core.tokens[choice.token_id as usize] == emitted_text)
            .ok_or_else(|| invalid("continuation_rust_emitted_text_not_available"))?;
        Ok(PyWriterContinuationRustCursor {
            core: Arc::clone(&self.core),
            node_id: choice.successor_node_id,
            completion_scale: cursor.completion_scale.clone() * choice.successor_scale.clone(),
        })
    }

    fn is_terminal(&self, cursor: &PyWriterContinuationRustCursor) -> PyResult<bool> {
        Ok(node_for_cursor(&self.core, cursor)?.terminal_available)
    }

    fn support_count(&self, cursor: &PyWriterContinuationRustCursor) -> PyResult<BigUint> {
        Ok(node_for_cursor(&self.core, cursor)?.support_count.clone())
    }

    fn completion_count(&self, cursor: &PyWriterContinuationRustCursor) -> PyResult<BigUint> {
        Ok(cursor.completion_scale.clone()
            * node_for_cursor(&self.core, cursor)?
                .completion_count
                .clone())
    }

    fn terminal_completion_count(
        &self,
        cursor: &PyWriterContinuationRustCursor,
    ) -> PyResult<BigUint> {
        Ok(cursor.completion_scale.clone()
            * node_for_cursor(&self.core, cursor)?
                .terminal_completion_count
                .clone())
    }

    fn probabilities(
        &self,
        cursor: &PyWriterContinuationRustCursor,
    ) -> PyResult<Vec<PyWriterContinuationRustProbability>> {
        let node = node_for_cursor(&self.core, cursor)?;
        let denominator = cursor.completion_scale.clone() * node.completion_count.clone();
        if denominator.is_zero() {
            return Err(invalid("continuation_rust_has_no_completion"));
        }
        let mut probabilities: Vec<_> = node_choices(&self.core, node)
            .iter()
            .map(|choice| PyWriterContinuationRustProbability {
                text: Some(self.core.tokens[choice.token_id as usize].clone()),
                numerator: cursor.completion_scale.clone() * choice.completion_count.clone(),
                denominator: denominator.clone(),
            })
            .collect();
        if node.terminal_available {
            probabilities.push(PyWriterContinuationRustProbability {
                text: None,
                numerator: cursor.completion_scale.clone() * node.terminal_completion_count.clone(),
                denominator,
            });
        }
        Ok(probabilities)
    }
}

#[pymethods]
impl PyWriterContinuationRustCursor {
    #[getter]
    fn node_id(&self) -> u32 {
        self.node_id
    }

    #[getter]
    fn completion_scale(&self) -> BigUint {
        self.completion_scale.clone()
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

#[pymethods]
impl PyWriterContinuationRustChoice {
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    #[getter]
    fn immediate_multiplicity(&self) -> BigUint {
        self.immediate_multiplicity.clone()
    }

    #[getter]
    fn support_count(&self) -> BigUint {
        self.support_count.clone()
    }

    #[getter]
    fn completion_count(&self) -> BigUint {
        self.completion_count.clone()
    }

    #[getter]
    fn next_cursor(&self) -> PyWriterContinuationRustCursor {
        self.next_cursor.clone()
    }
}

#[pymethods]
impl PyWriterContinuationRustProbability {
    #[getter]
    fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    #[getter]
    fn numerator(&self) -> BigUint {
        self.numerator.clone()
    }

    #[getter]
    fn denominator(&self) -> BigUint {
        self.denominator.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(node_id: u32, terminal_count: BigUint) -> NodeTerm {
        let terminal_available = !terminal_count.is_zero();
        let digest = format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "[{},{},{},[]]",
                    terminal_available, terminal_count, terminal_count
                )
                .as_bytes()
            )
        );
        (
            node_id,
            digest,
            terminal_available,
            terminal_count.clone(),
            terminal_count.clone(),
            vec![],
            if terminal_count.is_zero() {
                BigUint::zero()
            } else {
                BigUint::one()
            },
            terminal_count,
        )
    }

    #[test]
    fn accepts_counts_beyond_u128() {
        let huge = BigUint::one() << 160usize;
        let core = build_core(
            "asset".into(),
            0,
            BigUint::one(),
            vec![leaf(0, huge.clone())],
        )
        .unwrap();
        assert_eq!(core.nodes[0].completion_count, huge);
    }

    #[test]
    fn rejects_duplicate_semantic_classes() {
        let branch = |text: &str, successor: u32| {
            (
                text.to_owned(),
                BigUint::one(),
                successor,
                BigUint::one(),
                BigUint::one(),
                BigUint::one(),
            )
        };
        let root = (
            2,
            String::new(),
            false,
            BigUint::zero(),
            BigUint::zero(),
            vec![branch("a", 0), branch("b", 1)],
            BigUint::from(2u8),
            BigUint::from(2u8),
        );
        assert!(build_core(
            "asset".into(),
            2,
            BigUint::one(),
            vec![leaf(0, BigUint::one()), leaf(1, BigUint::one()), root],
        )
        .is_err());
    }
}
