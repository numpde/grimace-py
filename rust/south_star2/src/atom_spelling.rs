//! Incremental OpenSMILES atom spellings for one fixed atom fact.
//!
//! This module does not choose traversal or bond text. An accepted state is
//! one complete atom spelling; molecular end-of-string belongs to its caller.

use std::fmt;

const ELEMENT_SYMBOLS: &[&str] = &[
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];

/// An element symbol accepted by this lexer: OpenSMILES 1.0 symbols plus the
/// four symbols IUPAC approved in November 2016 (Nh, Mc, Ts, and Og).
///
/// OpenSMILES 1.0 predates those approvals, so this table intentionally does
/// not claim to be the exact 2016 grammar symbol set. See IUPAC's approval:
/// <https://iupac.org/iupac-announces-the-names-of-elements-113-115-117-and-118/>.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElementSymbol(&'static str);

impl ElementSymbol {
    pub fn new(symbol: &str) -> Result<Self, AtomFactsError> {
        if symbol == "*" {
            return Ok(Self("*"));
        }
        ELEMENT_SYMBOLS
            .iter()
            .copied()
            .find(|candidate| *candidate == symbol)
            .map(Self)
            .ok_or_else(|| AtomFactsError::UnknownElement(symbol.to_owned()))
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

/// One of the two tetrahedral parity markers represented by this atom field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TetrahedralMarker {
    At,
    AtAt,
}

/// Spelling decisions selected by the caller from graph and stereo constraints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AtomSpellingContext {
    pub aromatic_spelling: AromaticSpelling,
    pub tetrahedral_marker: Option<TetrahedralMarker>,
    pub bonds: AtomBondContext,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AromaticSpelling {
    #[default]
    Source,
    Kekule,
}

/// Native-derived incident bond kinds; totals are derived from these counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AtomBondContext {
    pub single_bonds: u8,
    pub double_bonds: u8,
    pub triple_bonds: u8,
    pub aromatic_bonds: u8,
}

impl AtomBondContext {
    fn degree(self) -> u16 {
        u16::from(self.single_bonds)
            + u16::from(self.double_bonds)
            + u16::from(self.triple_bonds)
            + u16::from(self.aromatic_bonds)
    }

    fn exocyclic_multiple_bond(self) -> bool {
        self.double_bonds != 0 || self.triple_bonds != 0
    }

    fn integer_valence(self) -> Option<u16> {
        (self.aromatic_bonds == 0).then_some(
            u16::from(self.single_bonds)
                + 2 * u16::from(self.double_bonds)
                + 3 * u16::from(self.triple_bonds),
        )
    }
}

/// Chemical atom facts used to generate standard atom spellings.
///
/// `non_graph_hydrogens` counts hydrogens in an atom field, not graph vertices.
/// An absent isotope differs from
/// isotope zero. OpenSMILES defines an omitted atom class as numeric zero, so
/// `atom_class` stores that normalized value; for zero both omission and an
/// explicit class field are generated.
/// Numeric fields are carried as bounded molecule facts (`u32` isotope and
/// `u64` class); these representation limits do not bound the lexical zero
/// prefixes accepted by the grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AtomFacts {
    pub element: ElementSymbol,
    pub aromatic: bool,
    pub isotope: Option<u32>,
    pub formal_charge: i8,
    pub non_graph_hydrogens: u8,
    pub atom_class: u64,
    pub aromatic_total_valence: Option<u32>,
}

impl AtomFacts {
    pub fn new(
        element: ElementSymbol,
        aromatic: bool,
        isotope: Option<u32>,
        formal_charge: i8,
        non_graph_hydrogens: u8,
        atom_class: u64,
        aromatic_total_valence: Option<u32>,
    ) -> Result<Self, AtomFactsError> {
        if aromatic != aromatic_total_valence.is_some() {
            return Err(AtomFactsError::AromaticValencePresence { aromatic });
        }
        if !(-99..=99).contains(&formal_charge) {
            return Err(AtomFactsError::ChargeOutOfRange(formal_charge));
        }
        if non_graph_hydrogens > 9 {
            return Err(AtomFactsError::HydrogenCountOutOfRange(non_graph_hydrogens));
        }
        if element.as_str() == "H" && non_graph_hydrogens != 0 {
            return Err(AtomFactsError::HydrogenAtomHasHydrogens(
                non_graph_hydrogens,
            ));
        }
        if aromatic
            && !matches!(
                element.as_str(),
                "B" | "C" | "N" | "O" | "P" | "S" | "As" | "Se" | "*"
            )
        {
            return Err(AtomFactsError::UnsupportedAromaticElement(element));
        }
        Ok(Self {
            element,
            aromatic,
            isotope,
            formal_charge,
            non_graph_hydrogens,
            atom_class,
            aromatic_total_valence,
        })
    }

    /// Return a bare organic-subset spelling only when its inferred H count agrees.
    fn bare_spelling(&self, context: AtomSpellingContext) -> Option<&'static str> {
        if self.isotope.is_some()
            || self.formal_charge != 0
            || self.atom_class != 0
            || context.tetrahedral_marker.is_some()
        {
            return None;
        }
        let symbol = if self.aromatic && context.aromatic_spelling == AromaticSpelling::Source {
            match self.element.as_str() {
                "B" => "b",
                "C" => "c",
                "N" => "n",
                "P" => "p",
                "*" => "*",
                "O" => "o",
                "S" => "s",
                _ => return None,
            }
        } else {
            match self.element.as_str() {
                "B" | "C" | "N" | "O" | "P" | "S" | "F" | "Cl" | "Br" | "I" => {
                    self.element.as_str()
                }
                "*" => "*",
                _ => return None,
            }
        };
        let eligible_hydrogens = if self.aromatic && context.aromatic_spelling == AromaticSpelling::Source {
            match self.element.as_str() {
                "C" if (2..=3).contains(&context.bonds.degree())
                    && context.bonds.aromatic_bonds >= 2
                    && context.bonds.triple_bonds == 0
                    && self.aromatic_total_valence == Some(4) =>
                {
                    Some((3 - context.bonds.degree()) as u8)
                }
                "B" if context.bonds.aromatic_bonds == 2
                    && (2..=3).contains(&context.bonds.degree())
                    && !context.bonds.exocyclic_multiple_bond()
                    && self.aromatic_total_valence == Some(3) =>
                {
                    Some(0)
                }
                "*" => Some(0),
                "N" | "P" if context.bonds.aromatic_bonds == 2
                    && !context.bonds.exocyclic_multiple_bond()
                    && (context.bonds.degree() == 2
                        || (context.bonds.degree() == 3 && context.bonds.single_bonds == 1)) =>
                {
                    Some(0)
                }
                "O" | "S" if context.bonds.aromatic_bonds == 2 && context.bonds.degree() == 2 => {
                    Some(0)
                }
                _ => None,
            }
        } else {
            if self.aromatic && context.aromatic_spelling == AromaticSpelling::Kekule {
                self.aromatic_total_valence
                    .expect("Kekule spelling requires aromatic total valence")
                    .checked_sub(u32::from(self.non_graph_hydrogens))
                    .and_then(|sum| u16::try_from(sum).ok())
            } else {
                context.bonds.integer_valence()
            }
                .and_then(|valence| normal_valence_hydrogens(self.element, valence))
        };
        (eligible_hydrogens == Some(self.non_graph_hydrogens)).then_some(symbol)
    }
}

/// One finite next-character transition and its immutable successor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomSpellingChoice {
    pub text: char,
    pub field: AtomSpellingField,
    pub successor: AtomSpellingState,
}

/// Atom-grammar field that owns one emitted character.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomSpellingField {
    BareElement,
    OpenBracket,
    Isotope,
    Element,
    TetrahedralMarker,
    HydrogenCount,
    Charge,
    AtomClass,
    CloseBracket,
}

/// Native atom spelling policy selected independently of traversal constraints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AtomSpellingPolicy {
    #[default]
    Maximal,
    Compact,
}

/// Incremental spelling language for one fixed `AtomFacts` and selected context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomSpellingState {
    facts: AtomFacts,
    context: AtomSpellingContext,
    phase: Phase,
    policy: AtomSpellingPolicy,
    atom_class_present: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Phase {
    Start,
    AfterOpen,
    ElementStart,
    Number(NumberField),
    Element {
        text: String,
        offset: usize,
        bracketed: bool,
    },
    AfterElement,
    AfterAt,
    Marker {
        text: &'static str,
        offset: usize,
    },
    AfterMarker,
    HydrogenDigit,
    ChargeStart,
    ChargeAfterSign,
    ChargeDigits {
        text: String,
        offset: usize,
    },
    ClassStart,
    Close,
    Complete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NumberField {
    target: String,
    offset: usize,
    kind: NumberKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumberKind {
    Isotope,
    Class,
}

impl AtomSpellingState {
    pub fn initial(facts: AtomFacts, context: AtomSpellingContext) -> Self {
        Self::initial_with_policy(facts, context, AtomSpellingPolicy::Maximal, false)
    }

    /// Source map-property presence distinguishes absent class zero from `:0`.
    pub fn initial_with_policy(
        facts: AtomFacts,
        context: AtomSpellingContext,
        policy: AtomSpellingPolicy,
        atom_class_present: bool,
    ) -> Self {
        Self {
            facts,
            context,
            phase: Phase::Start,
            policy,
            atom_class_present,
        }
    }

    /// Whether exactly one complete atom spelling has been emitted.
    pub fn is_complete(&self) -> bool {
        matches!(self.phase, Phase::Complete)
    }

    /// Return finite next characters without searching or enumerating suffixes.
    /// Duplicate characters may lead to distinct valid spellings, for example
    /// when an explicit zero-valued field can either finish or take another zero.
    pub fn choices(&self) -> Vec<AtomSpellingChoice> {
        let mut choices = Vec::new();
        macro_rules! push {
            ($field:expr, $text:expr, $phase:expr $(,)?) => {
                push_choice(&mut choices, self, $field, $text, $phase)
            };
        }
        match &self.phase {
            Phase::Start => {
                let bare = self.facts.bare_spelling(self.context);
                if self.policy == AtomSpellingPolicy::Maximal
                    || self.atom_class_present
                    || bare.is_none()
                {
                    push!(AtomSpellingField::OpenBracket, '[', Phase::AfterOpen);
                }
                if let Some(bare) = bare.filter(|_| {
                    self.policy == AtomSpellingPolicy::Maximal || !self.atom_class_present
                }) {
                    push_text_first(
                        bare,
                        false,
                        Phase::Complete,
                        AtomSpellingField::BareElement,
                        self,
                        &mut choices,
                    );
                }
            }
            Phase::AfterOpen => {
                if let Some(isotope) = self.facts.isotope {
                    push_number_start(&mut choices, self, isotope as u64, NumberKind::Isotope);
                }
                if self.facts.isotope.is_none() {
                    push_element_start(&mut choices, self, true);
                }
            }
            Phase::ElementStart => push_element_start(&mut choices, self, true),
            Phase::Number(field) => {
                push_number_choices(&mut choices, self, field);
            }
            Phase::Element {
                text,
                offset,
                bracketed,
            } => {
                let ch = text
                    .chars()
                    .nth(*offset)
                    .expect("element text offset is valid");
                let next_offset = *offset + ch.len_utf8();
                let phase = if next_offset == text.len() {
                    if *bracketed {
                        Phase::AfterElement
                    } else {
                        Phase::Complete
                    }
                } else {
                    Phase::Element {
                        text: text.clone(),
                        offset: next_offset,
                        bracketed: *bracketed,
                    }
                };
                push!(AtomSpellingField::Element, ch, phase);
            }
            Phase::AfterElement => {
                if self.context.tetrahedral_marker.is_some() {
                    push!(AtomSpellingField::TetrahedralMarker, '@', Phase::AfterAt);
                } else {
                    push_after_marker(&mut choices, self);
                }
            }
            Phase::AfterAt => match self.context.tetrahedral_marker {
                Some(TetrahedralMarker::At) => {
                    push_after_marker(&mut choices, self);
                    if self.policy == AtomSpellingPolicy::Maximal {
                        push!(
                            AtomSpellingField::TetrahedralMarker,
                            'T',
                            Phase::Marker {
                                text: "H1",
                                offset: 0,
                            },
                        );
                    }
                }
                Some(TetrahedralMarker::AtAt) => {
                    push!(
                        AtomSpellingField::TetrahedralMarker,
                        '@',
                        Phase::AfterMarker,
                    );
                    if self.policy == AtomSpellingPolicy::Maximal {
                        push!(
                            AtomSpellingField::TetrahedralMarker,
                            'T',
                            Phase::Marker {
                                text: "H2",
                                offset: 0,
                            },
                        );
                    }
                }
                None => unreachable!("AfterAt requires a selected tetrahedral marker"),
            },
            Phase::Marker { text, offset } => {
                let ch = text.chars().nth(*offset).expect("marker offset is valid");
                let next_offset = *offset + ch.len_utf8();
                let phase = if next_offset == text.len() {
                    Phase::AfterMarker
                } else {
                    Phase::Marker {
                        text,
                        offset: next_offset,
                    }
                };
                push!(AtomSpellingField::TetrahedralMarker, ch, phase);
            }
            Phase::AfterMarker => push_after_marker(&mut choices, self),
            Phase::HydrogenDigit => {
                if self.facts.non_graph_hydrogens == 1 {
                    // H without a digit and H1 have the same meaning.
                    push_after_hydrogen(&mut choices, self);
                    if self.policy == AtomSpellingPolicy::Maximal {
                        push!(AtomSpellingField::HydrogenCount, '1', Phase::ChargeStart);
                    }
                } else {
                    // OpenSMILES' bracket H-count production has one digit.
                    push!(
                        AtomSpellingField::HydrogenCount,
                        char::from(b'0' + self.facts.non_graph_hydrogens),
                        Phase::ChargeStart,
                    );
                }
            }
            Phase::ChargeStart => {
                if self.facts.formal_charge == 0 {
                    push_class_start(&mut choices, self);
                    if self.policy == AtomSpellingPolicy::Maximal {
                        push!(AtomSpellingField::Charge, '+', Phase::ChargeAfterSign);
                        push!(AtomSpellingField::Charge, '-', Phase::ChargeAfterSign);
                    }
                } else {
                    push!(
                        AtomSpellingField::Charge,
                        charge_sign(self.facts),
                        Phase::ChargeAfterSign
                    );
                }
            }
            Phase::ChargeAfterSign => {
                let magnitude = self.facts.formal_charge.unsigned_abs();
                let sign = charge_sign(self.facts);
                if magnitude == 2 && self.policy == AtomSpellingPolicy::Maximal {
                    push!(AtomSpellingField::Charge, sign, Phase::ClassStart);
                }
                if magnitude == 1 {
                    push_class_start(&mut choices, self);
                }
                let digits = magnitude.to_string();
                if !(magnitude == 1 && self.policy == AtomSpellingPolicy::Compact) {
                    if digits.len() == 1 {
                        push!(
                            AtomSpellingField::Charge,
                            digits.chars().next().unwrap(),
                            Phase::ClassStart
                        );
                        if self.policy == AtomSpellingPolicy::Maximal {
                            push!(
                                AtomSpellingField::Charge,
                                '0',
                                Phase::ChargeDigits {
                                    text: digits,
                                    offset: 0
                                }
                            );
                        }
                    } else {
                        let first = digits.chars().next().unwrap();
                        push!(
                            AtomSpellingField::Charge,
                            first,
                            Phase::ChargeDigits {
                                text: digits,
                                offset: 1
                            }
                        );
                    }
                }
            }
            Phase::ChargeDigits { text, offset } => {
                let ch = text
                    .chars()
                    .nth(*offset)
                    .expect("charge digit offset is valid");
                let next_offset = *offset + ch.len_utf8();
                let phase = if next_offset == text.len() {
                    Phase::ClassStart
                } else {
                    Phase::ChargeDigits {
                        text: text.clone(),
                        offset: next_offset,
                    }
                };
                push!(AtomSpellingField::Charge, ch, phase);
            }
            Phase::ClassStart => {
                if self.facts.atom_class == 0
                    && (self.policy == AtomSpellingPolicy::Maximal || !self.atom_class_present)
                {
                    push!(AtomSpellingField::CloseBracket, ']', Phase::Complete);
                }
                if self.policy == AtomSpellingPolicy::Maximal
                    || self.facts.atom_class != 0
                    || self.atom_class_present
                {
                    push!(
                        AtomSpellingField::AtomClass,
                        ':',
                        Phase::Number(NumberField {
                            target: self.facts.atom_class.to_string(),
                            offset: 0,
                            kind: NumberKind::Class,
                        }),
                    );
                }
            }
            Phase::Close => push!(AtomSpellingField::CloseBracket, ']', Phase::Complete),
            Phase::Complete => {}
        }
        choices
    }
}

fn push_after_marker(choices: &mut Vec<AtomSpellingChoice>, state: &AtomSpellingState) {
    let facts = state.facts;
    if facts.element.as_str() == "H" {
        push_after_hydrogen(choices, state);
        return;
    }
    if facts.non_graph_hydrogens > 0 {
        push_choice(
            choices,
            state,
            AtomSpellingField::HydrogenCount,
            'H',
            Phase::HydrogenDigit,
        );
    } else {
        push_after_hydrogen(choices, state);
        // H0 has the same non-graph hydrogen count as an omitted H field.
        if state.policy == AtomSpellingPolicy::Maximal {
            push_choice(
                choices,
                state,
                AtomSpellingField::HydrogenCount,
                'H',
                Phase::HydrogenDigit,
            );
        }
    }
}

fn push_after_hydrogen(choices: &mut Vec<AtomSpellingChoice>, state: &AtomSpellingState) {
    let facts = state.facts;
    if facts.formal_charge != 0 {
        push_choice(
            choices,
            state,
            AtomSpellingField::Charge,
            charge_sign(facts),
            Phase::ChargeAfterSign,
        );
    } else {
        push_class_start(choices, state);
        if state.policy == AtomSpellingPolicy::Maximal {
            push_choice(
                choices,
                state,
                AtomSpellingField::Charge,
                '+',
                Phase::ChargeAfterSign,
            );
            push_choice(
                choices,
                state,
                AtomSpellingField::Charge,
                '-',
                Phase::ChargeAfterSign,
            );
        }
    }
}

fn push_class_start(choices: &mut Vec<AtomSpellingChoice>, state: &AtomSpellingState) {
    let facts = state.facts;
    if facts.atom_class == 0
        && (state.policy == AtomSpellingPolicy::Maximal || !state.atom_class_present)
    {
        push_choice(
            choices,
            state,
            AtomSpellingField::CloseBracket,
            ']',
            Phase::Complete,
        );
    }
    if facts.atom_class != 0 {
        push_choice(
            choices,
            state,
            AtomSpellingField::AtomClass,
            ':',
            Phase::Number(NumberField {
                target: facts.atom_class.to_string(),
                offset: 0,
                kind: NumberKind::Class,
            }),
        );
    }
    if facts.atom_class == 0
        && (state.policy == AtomSpellingPolicy::Maximal || state.atom_class_present)
    {
        push_choice(
            choices,
            state,
            AtomSpellingField::AtomClass,
            ':',
            Phase::Number(NumberField {
                target: "0".to_owned(),
                offset: 0,
                kind: NumberKind::Class,
            }),
        );
    }
}

fn push_number_start(
    choices: &mut Vec<AtomSpellingChoice>,
    state: &AtomSpellingState,
    value: u64,
    kind: NumberKind,
) {
    push_number_choices(
        choices,
        state,
        &NumberField {
            target: value.to_string(),
            offset: 0,
            kind,
        },
    );
}

fn push_number_choices(
    choices: &mut Vec<AtomSpellingChoice>,
    state: &AtomSpellingState,
    field: &NumberField,
) {
    let target = field.target.as_bytes()[field.offset] as char;
    let next = if field.offset + 1 == field.target.len() {
        number_suffix_phase(field.kind)
    } else {
        Phase::Number(NumberField {
            offset: field.offset + 1,
            ..field.clone()
        })
    };
    push_choice(choices, state, number_field(field.kind), target, next);
    // Only a prefix zero is padding; internal zeros belong to the fixed value.
    if state.policy == AtomSpellingPolicy::Maximal && field.offset == 0 {
        push_choice(
            choices,
            state,
            number_field(field.kind),
            '0',
            Phase::Number(field.clone()),
        );
    }
}

fn number_suffix_phase(kind: NumberKind) -> Phase {
    match kind {
        NumberKind::Isotope => Phase::ElementStart,
        NumberKind::Class => Phase::Close,
    }
}

fn push_element_start(
    choices: &mut Vec<AtomSpellingChoice>,
    state: &AtomSpellingState,
    bracketed: bool,
) {
    let facts = state.facts;
    let context = state.context;
    let text = element_text(facts, context);
    push_text_first(
        &text,
        bracketed,
        if bracketed {
            Phase::AfterElement
        } else {
            Phase::Complete
        },
        if bracketed {
            AtomSpellingField::Element
        } else {
            AtomSpellingField::BareElement
        },
        state,
        choices,
    );
}

fn push_text_first(
    text: &str,
    bracketed: bool,
    terminal: Phase,
    field: AtomSpellingField,
    state: &AtomSpellingState,
    choices: &mut Vec<AtomSpellingChoice>,
) {
    let mut chars = text.chars();
    let first = chars.next().expect("text is nonempty");
    let tail = chars.collect::<String>();
    let phase = if tail.is_empty() {
        terminal
    } else {
        Phase::Element {
            text: tail,
            offset: 0,
            bracketed,
        }
    };
    push_choice(choices, state, field, first, phase);
}

fn element_text(facts: AtomFacts, context: AtomSpellingContext) -> String {
    if facts.aromatic && context.aromatic_spelling == AromaticSpelling::Source {
        match facts.element.as_str() {
            "B" => "b",
            "C" => "c",
            "N" => "n",
            "O" => "o",
            "P" => "p",
            "S" => "s",
            "As" => "as",
            "Se" => "se",
            "*" => "*",
            _ => unreachable!("aromatic element was validated"),
        }
        .to_owned()
    } else {
        facts.element.as_str().to_owned()
    }
}

fn charge_sign(facts: AtomFacts) -> char {
    if facts.formal_charge < 0 {
        '-'
    } else {
        '+'
    }
}

fn push_choice(
    choices: &mut Vec<AtomSpellingChoice>,
    state: &AtomSpellingState,
    field: AtomSpellingField,
    text: char,
    phase: Phase,
) {
    choices.push(AtomSpellingChoice {
        text,
        field,
        successor: AtomSpellingState {
            facts: state.facts,
            context: state.context,
            phase,
            policy: state.policy,
            atom_class_present: state.atom_class_present,
        },
    });
}

fn number_field(kind: NumberKind) -> AtomSpellingField {
    match kind {
        NumberKind::Isotope => AtomSpellingField::Isotope,
        NumberKind::Class => AtomSpellingField::AtomClass,
    }
}

/// OpenSMILES §3.1.5 organic-subset H inference for neutral common elements.
fn normal_valence_hydrogens(element: ElementSymbol, bond_order_sum: u16) -> Option<u8> {
    let normal_valences: &[u16] = match element.as_str() {
        "B" => &[3],
        "C" => &[4],
        "N" | "P" => &[3, 5],
        "O" => &[2],
        "S" => &[2, 4, 6],
        "F" | "Cl" | "Br" | "I" => &[1],
        "*" => return Some(0),
        _ => return None,
    };
    if normal_valences.contains(&bond_order_sum) {
        return Some(0);
    }
    let hydrogens = normal_valences
        .iter()
        .copied()
        .find(|valence| *valence > bond_order_sum)
        .map(|valence| valence - bond_order_sum)
        .unwrap_or(0);
    u8::try_from(hydrogens).ok()
}

/// Errors in atom facts that this OpenSMILES atom lexer can represent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomFactsError {
    AromaticValencePresence { aromatic: bool },
    UnknownElement(String),
    ChargeOutOfRange(i8),
    HydrogenCountOutOfRange(u8),
    HydrogenAtomHasHydrogens(u8),
    UnsupportedAromaticElement(ElementSymbol),
}

impl fmt::Display for AtomFactsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AromaticValencePresence { aromatic } => write!(
                f, "aromatic total valence must be {} for {} atoms",
                if *aromatic { "present" } else { "absent" },
                if *aromatic { "aromatic" } else { "nonaromatic" }
            ),
            Self::UnknownElement(symbol) => {
                write!(f, "unknown OpenSMILES element symbol {symbol:?}")
            }
            Self::ChargeOutOfRange(charge) => write!(
                f,
                "OpenSMILES charge grammar supports at most two digits, got {charge}"
            ),
            Self::HydrogenCountOutOfRange(count) => write!(
                f,
                "OpenSMILES bracket hydrogen count must be one digit, got {count}"
            ),
            Self::HydrogenAtomHasHydrogens(count) => write!(
                f,
                "hydrogen atom cannot carry a bracket H-count field, got H{count}"
            ),
            Self::UnsupportedAromaticElement(element) => write!(
                f,
                "OpenSMILES has no aromatic spelling for {}",
                element.as_str()
            ),
        }
    }
}

impl std::error::Error for AtomFactsError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(
        symbol: &str, aromatic: bool, h: u8, atom_class: u64,
        aromatic_total_valence: Option<u32>,
    ) -> AtomFacts {
        AtomFacts::new(
            ElementSymbol::new(symbol).unwrap(),
            aromatic,
            None,
            0,
            h,
            atom_class,
            aromatic_total_valence,
        )
        .unwrap()
    }

    fn compact_support(
        facts: AtomFacts,
        context: AtomSpellingContext,
        map_present: bool,
    ) -> Vec<String> {
        let initial = AtomSpellingState::initial_with_policy(
            facts,
            context,
            AtomSpellingPolicy::Compact,
            map_present,
        );
        let mut pending = vec![(String::new(), initial)];
        let mut complete = Vec::new();
        while let Some((text, state)) = pending.pop() {
            assert!(text.len() < 64, "compact atom grammar must terminate");
            if state.is_complete() {
                complete.push(text);
            } else {
                let choices = state.choices();
                assert!(
                    !choices.is_empty(),
                    "compact prefix {text:?} must remain live"
                );
                for choice in choices {
                    pending.push((format!("{text}{}", choice.text), choice.successor));
                }
            }
        }
        complete.sort();
        complete
    }

    fn accepts_with_policy(
        facts: AtomFacts,
        context: AtomSpellingContext,
        policy: AtomSpellingPolicy,
        map_present: bool,
        text: &str,
    ) -> bool {
        let mut states = vec![AtomSpellingState::initial_with_policy(
            facts,
            context,
            policy,
            map_present,
        )];
        for ch in text.chars() {
            states = states
                .into_iter()
                .flat_map(|state| state.choices())
                .filter(|choice| choice.text == ch)
                .map(|choice| choice.successor)
                .collect();
        }
        states.iter().any(AtomSpellingState::is_complete)
    }

    fn finite_bracket_suffixes(
        atom: AtomFacts,
        context: AtomSpellingContext,
        policy: AtomSpellingPolicy,
    ) -> std::collections::BTreeSet<String> {
        let mut states = vec![AtomSpellingState::initial_with_policy(
            atom, context, policy, false,
        )];
        for ch in "[13C".chars() {
            states = states
                .into_iter()
                .flat_map(|state| state.choices())
                .filter(|choice| choice.text == ch)
                .map(|choice| choice.successor)
                .collect();
            assert!(
                !states.is_empty(),
                "fixed isotope and element prefix must exist"
            );
        }
        let mut pending = states
            .into_iter()
            .map(|state| (String::new(), state))
            .collect::<Vec<_>>();
        let mut complete = std::collections::BTreeSet::new();
        while let Some((suffix, state)) = pending.pop() {
            assert!(
                suffix.len() <= 12,
                "finite H/charge/marker suffix exceeded grammar width"
            );
            if state.is_complete() {
                complete.insert(suffix);
                continue;
            }
            // Class zero is optional here. Exclude its entire production at its entry.
            let choices = state
                .choices()
                .into_iter()
                .filter(|choice| choice.field != AtomSpellingField::AtomClass)
                .collect::<Vec<_>>();
            assert!(
                !choices.is_empty(),
                "published finite prefix has no completion"
            );
            for choice in choices {
                pending.push((format!("{suffix}{}", choice.text), choice.successor));
            }
        }
        complete
    }

    fn grammar_charge_spellings(value: i8) -> std::collections::BTreeSet<String> {
        let mut spellings = std::collections::BTreeSet::new();
        if value == 0 {
            spellings.insert(String::new());
        }
        for (sign, polarity) in [('+', 1_i16), ('-', -1_i16)] {
            if i16::from(value) == polarity {
                spellings.insert(sign.to_string());
            }
            if i16::from(value) == 2 * polarity {
                spellings.insert(format!("{sign}{sign}"));
            }
            for magnitude in 0..=99 {
                if i16::from(value) != polarity * magnitude {
                    continue;
                }
                spellings.insert(format!("{sign}{magnitude}"));
                if magnitude < 10 {
                    spellings.insert(format!("{sign}0{magnitude}"));
                }
            }
        }
        spellings
    }

    #[test]
    fn bracket_hydrogen_and_charge_productions_match_complete_finite_language() {
        for h in 0..=9 {
            let hydrogens = match h {
                0 => vec![String::new(), "H0".into()],
                1 => vec!["H".into(), "H1".into()],
                _ => vec![format!("H{h}")],
            };
            for charge in -99..=99 {
                let mut atom = facts("C", false, h, 0, None);
                atom.isotope = Some(13); // Fixes bracket form in both profiles.
                atom.formal_charge = charge;
                let expected = hydrogens
                    .iter()
                    .flat_map(|hydrogen| {
                        grammar_charge_spellings(charge)
                            .into_iter()
                            .map(move |charge_text| format!("{hydrogen}{charge_text}]"))
                    })
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(
                    finite_bracket_suffixes(
                        atom,
                        AtomSpellingContext::default(),
                        AtomSpellingPolicy::Maximal
                    ),
                    expected,
                    "H={h}, charge={charge}"
                );

                let short_h = if h == 0 {
                    String::new()
                } else if h == 1 {
                    "H".into()
                } else {
                    format!("H{h}")
                };
                let short_charge = if charge == 0 {
                    String::new()
                } else if charge == 1 {
                    "+".into()
                } else if charge == -1 {
                    "-".into()
                } else {
                    format!("{:+}", charge)
                };
                let compact = [format!("{short_h}{short_charge}]")].into_iter().collect();
                assert_eq!(
                    finite_bracket_suffixes(
                        atom,
                        AtomSpellingContext::default(),
                        AtomSpellingPolicy::Compact
                    ),
                    compact,
                    "compact H={h}, charge={charge}"
                );
            }
        }
        let hydrogen = facts("H", false, 0, 0, None);
        for invalid in ["[HH]", "[HH0]", "[HH1]"] {
            assert!(!accepts_with_policy(
                hydrogen,
                AtomSpellingContext::default(),
                AtomSpellingPolicy::Maximal,
                false,
                invalid
            ));
        }
        let mut isotope_carbon = facts("C", false, 0, 0, None);
        isotope_carbon.isotope = Some(13);
        for noncompact in ["[013C]", "[13CH0]", "[13C+0]", "[13C:0]"] {
            assert!(!accepts_with_policy(
                isotope_carbon,
                AtomSpellingContext::default(),
                AtomSpellingPolicy::Compact,
                false,
                noncompact
            ));
        }
    }

    #[test]
    fn bracket_tetrahedral_and_optional_field_order_is_fixed() {
        let mut finite_atom = facts("C", false, 1, 0, None);
        finite_atom.isotope = Some(13);
        finite_atom.formal_charge = 2;
        for (marker, spellings) in [
            (TetrahedralMarker::At, ["@", "@TH1"]),
            (TetrahedralMarker::AtAt, ["@@", "@TH2"]),
        ] {
            let context = AtomSpellingContext {
                tetrahedral_marker: Some(marker),
                ..AtomSpellingContext::default()
            };
            let expected = spellings
                .into_iter()
                .flat_map(|spelling| {
                    ["H", "H1"].into_iter().flat_map(move |hydrogen| {
                        grammar_charge_spellings(2)
                            .into_iter()
                            .map(move |charge| format!("{spelling}{hydrogen}{charge}]"))
                    })
                })
                .collect();
            assert_eq!(
                finite_bracket_suffixes(finite_atom, context, AtomSpellingPolicy::Maximal),
                expected
            );
            let compact = [format!("{}H+2]", spellings[0])].into_iter().collect();
            assert_eq!(
                finite_bracket_suffixes(finite_atom, context, AtomSpellingPolicy::Compact),
                compact
            );
        }
        let mut atom = facts("C", false, 1, 5, None);
        atom.isotope = Some(13);
        atom.formal_charge = 2;
        for (marker, short, long) in [
            (TetrahedralMarker::At, "@", "@TH1"),
            (TetrahedralMarker::AtAt, "@@", "@TH2"),
        ] {
            let context = AtomSpellingContext {
                tetrahedral_marker: Some(marker),
                ..AtomSpellingContext::default()
            };
            for spelling in [short, long] {
                let text = format!("[0013C{spelling}H1+02:005]");
                assert!(
                    accepts_with_policy(atom, context, AtomSpellingPolicy::Maximal, false, &text),
                    "missing {text}"
                );
            }
            let compact = format!("[13C{short}H+2:5]");
            assert!(accepts_with_policy(
                atom,
                context,
                AtomSpellingPolicy::Compact,
                true,
                &compact
            ));
            for invalid in [
                format!("[0013CH1{short}+02:005]"),
                format!("[0013C{short}+02H1:005]"),
                format!("[0013C{short}H1:005+02]"),
                format!("[0013C{short}{short}H1+02:005]"),
            ] {
                assert!(
                    !accepts_with_policy(
                        atom,
                        context,
                        AtomSpellingPolicy::Maximal,
                        false,
                        &invalid
                    ),
                    "admitted {invalid}"
                );
            }
        }
    }

    #[test]
    fn compact_complete_atom_support_preserves_metadata_and_internal_zeros() {
        let mut atom = facts("C", false, 4, 0, None);
        let context = AtomSpellingContext::default();
        assert_eq!(compact_support(atom, context, false), ["C"]);
        assert_eq!(compact_support(atom, context, true), ["[CH4:0]"]);
        atom.isotope = Some(0);
        assert_eq!(compact_support(atom, context, false), ["[0CH4]"]);
        atom.isotope = Some(101);
        atom.atom_class = 1002;
        assert_eq!(compact_support(atom, context, true), ["[101CH4:1002]"]);
        assert!(supports(atom, context, "[101CH4:1002]"));
        assert!(supports(atom, context, "[00101CH4:001002]"));
        assert!(!supports(atom, context, "[1001CH4:1002]"));
    }

    #[test]
    fn compact_complete_atom_support_uses_short_hydrogen_charge_and_stereo() {
        let mut atom = facts("N", false, 1, 0, None);
        let mut context = AtomSpellingContext::default();
        assert_eq!(compact_support(atom, context, false), ["[NH]"]);
        atom.non_graph_hydrogens = 0;
        assert_eq!(compact_support(atom, context, false), ["[N]"]);
        for (charge, suffix) in [(1, "+"), (-1, "-"), (2, "+2"), (-2, "-2"), (10, "+10")] {
            atom.formal_charge = charge;
            assert_eq!(
                compact_support(atom, context, false),
                [format!("[N{suffix}]")]
            );
        }
        atom.formal_charge = 0;
        atom.non_graph_hydrogens = 1;
        for (marker, text) in [
            (TetrahedralMarker::At, "[N@H]"),
            (TetrahedralMarker::AtAt, "[N@@H]"),
        ] {
            context.tetrahedral_marker = Some(marker);
            assert_eq!(compact_support(atom, context, false), [text]);
            assert!(supports(atom, context, text));
        }
        let donor = facts("N", true, 1, 0, Some(3));
        let aromatic = bond_context(0, 0, 0, 2);
        assert_eq!(compact_support(donor, aromatic, false), ["[nH]"]);
        let carbon = facts("C", true, 1, 0, Some(4));
        assert_eq!(compact_support(carbon, aromatic, false), ["c"]);
    }

    #[test]
    fn aromatic_total_valence_presence_matches_chemical_identity() {
        let carbon = ElementSymbol::new("C").unwrap();
        assert_eq!(
            AtomFacts::new(carbon, true, None, 0, 1, 0, None),
            Err(AtomFactsError::AromaticValencePresence { aromatic: true })
        );
        assert_eq!(
            AtomFacts::new(carbon, false, None, 0, 4, 0, Some(4)),
            Err(AtomFactsError::AromaticValencePresence { aromatic: false })
        );
        let aromatic = AtomFacts::new(carbon, true, None, 0, 1, 0, Some(4)).unwrap();
        assert!(aromatic.aromatic);
        assert_eq!(aromatic.aromatic_total_valence, Some(4));
    }

    fn states_after(
        facts: AtomFacts,
        context: AtomSpellingContext,
        text: &str,
    ) -> Vec<AtomSpellingState> {
        let mut states = vec![AtomSpellingState::initial(facts, context)];
        for character in text.chars() {
            states = states
                .iter()
                .flat_map(AtomSpellingState::choices)
                .filter(|choice| choice.text == character)
                .map(|choice| choice.successor)
                .collect();
            states.sort_by_key(|state| format!("{state:?}"));
            states.dedup();
            if states.is_empty() {
                break;
            }
        }
        states
    }

    fn supports(facts: AtomFacts, context: AtomSpellingContext, text: &str) -> bool {
        states_after(facts, context, text)
            .iter()
            .any(AtomSpellingState::is_complete)
    }

    fn bond_context(single: u8, double: u8, triple: u8, aromatic: u8) -> AtomSpellingContext {
        AtomSpellingContext {
            bonds: AtomBondContext {
                single_bonds: single,
                double_bonds: double,
                triple_bonds: triple,
                aromatic_bonds: aromatic,
            },
            ..AtomSpellingContext::default()
        }
    }

    #[test]
    fn organic_subset_uses_standard_normal_valences_for_common_elements() {
        for (symbol, h, context) in [
            ("B", 0, bond_context(3, 0, 0, 0)),
            ("C", 1, bond_context(3, 0, 0, 0)),
            ("N", 1, bond_context(2, 0, 0, 0)),
            ("O", 1, bond_context(1, 0, 0, 0)),
            ("P", 1, bond_context(2, 0, 0, 0)),
            ("S", 1, bond_context(1, 0, 0, 0)),
            ("F", 0, bond_context(1, 0, 0, 0)),
            ("Cl", 0, bond_context(1, 0, 0, 0)),
            ("Br", 0, bond_context(1, 0, 0, 0)),
            ("I", 0, bond_context(1, 0, 0, 0)),
        ] {
            assert!(supports(facts(symbol, false, h, 0, None), context, symbol));
        }
        assert!(!supports(
            facts("Cl", false, 1, 0, None),
            bond_context(1, 0, 0, 0),
            "Cl"
        ));
        assert!(supports(
            facts("C", false, 1, 0, None),
            bond_context(0, 0, 1, 0),
            "C"
        ));
    }

    #[test]
    fn source_carbon_bare_hydrogens_preserve_valid_exocyclic_valence() {
        assert!(supports(facts("C", true, 0, 0, Some(4)), bond_context(0, 1, 0, 2), "c"));
        for (h, valence, context) in [
            (1, 4, bond_context(0, 1, 0, 2)),
            (0, 3, bond_context(0, 1, 0, 2)),
            (0, 4, bond_context(0, 0, 1, 2)),
            (0, 4, bond_context(0, 2, 0, 2)),
            (0, 4, bond_context(1, 1, 0, 1)),
        ] {
            assert!(!supports(facts("C", true, h, 0, Some(valence)), context, "c"));
        }
        assert!(!supports(facts("N", true, 1, 0, Some(3)), bond_context(0, 0, 0, 2), "n"));
    }

    #[test]
    fn aromatic_bare_spelling_uses_supported_bond_context() {
        let aromatic_ring = bond_context(0, 0, 0, 2);
        assert!(supports(facts("C", true, 1, 0, Some(4)), aromatic_ring, "c"));
        assert!(!supports(facts("C", true, 0, 0, Some(4)), aromatic_ring, "c"));
        assert!(supports(
            facts("C", true, 0, 0, Some(4)),
            bond_context(1, 0, 0, 2),
            "c"
        ));
        assert!(supports(facts("N", true, 0, 0, Some(3)), aromatic_ring, "n"));
        let aromatic_tertiary_n = bond_context(1, 0, 0, 2);
        assert!(supports(
            facts("N", true, 0, 0, Some(3)),
            aromatic_tertiary_n,
            "n"
        ));
        assert!(!supports(
            facts("N", true, 1, 0, Some(3)),
            aromatic_tertiary_n,
            "n"
        ));
        assert!(supports(
            facts("N", true, 1, 0, Some(3)),
            aromatic_tertiary_n,
            "[nH]"
        ));
        assert!(!supports(
            facts("N", true, 0, 0, Some(3)),
            bond_context(0, 1, 0, 2),
            "n"
        ));
        let charged_n = AtomFacts::new(
            ElementSymbol::new("N").unwrap(),
            true,
            None,
            1,
            0,
            0,
            Some(4),
        )
        .unwrap();
        assert!(!supports(charged_n, aromatic_tertiary_n, "n"));
        assert!(!supports(facts("N", true, 1, 0, Some(3)), aromatic_ring, "n"));
        assert!(supports(facts("N", true, 1, 0, Some(3)), aromatic_ring, "[nH]"));
        assert!(supports(facts("N", true, 0, 0, Some(3)), aromatic_ring, "[nH0]"));
        assert!(supports(facts("O", true, 0, 0, Some(2)), aromatic_ring, "o"));
        assert!(supports(facts("S", true, 0, 0, Some(2)), aromatic_ring, "s"));
        assert!(!supports(
            facts("O", true, 0, 0, Some(2)),
            bond_context(1, 0, 0, 1),
            "o"
        ));
        assert!(supports(facts("P", true, 0, 0, Some(3)), aromatic_ring, "[p]"));
        assert!(supports(facts("P", true, 0, 0, Some(3)), aromatic_ring, "p"));
        assert!(!supports(
            facts("C", true, 1, 0, Some(4)),
            bond_context(0, 1, 0, 2),
            "c"
        ));
    }

    #[test]
    fn bare_aromatic_boron_phosphorus_and_wildcard_preserve_fixed_hydrogens() {
        let source = bond_context(0, 0, 0, 2);
        assert!(supports(facts("B", true, 0, 0, Some(3)), source, "b"));
        assert!(!supports(facts("B", true, 1, 0, Some(3)), source, "b"));
        assert!(!supports(facts("B", true, 0, 0, Some(4)), source, "b"));
        for symbol in ["N", "P"] {
            let bare = symbol.to_ascii_lowercase();
            assert!(supports(facts(symbol, true, 0, 0, Some(3)), source, &bare));
            assert!(!supports(facts(symbol, true, 1, 0, Some(3)), source, &bare));
            assert!(supports(facts(symbol, true, 1, 0, Some(3)), source, &format!("[{bare}H]")));
            assert!(!supports(facts(symbol, true, 0, 0, Some(3)), bond_context(0, 1, 0, 2), &bare));
        }
        for notation in [AromaticSpelling::Source, AromaticSpelling::Kekule] {
            let context = AtomSpellingContext { aromatic_spelling: notation, ..source };
            let wildcard = facts("*", true, 0, 0, Some(3));
            assert!(supports(wildcard, context, "*"));
            assert!(!supports(facts("*", true, 1, 0, Some(3)), context, "*"));
            for restricted in [
                AtomFacts { isotope: Some(13), ..wildcard },
                AtomFacts { formal_charge: 1, ..wildcard },
                AtomFacts { atom_class: 1, ..wildcard },
            ] {
                assert!(!supports(restricted, context, "*"));
            }
            assert!(!supports(wildcard, AtomSpellingContext {
                tetrahedral_marker: Some(TetrahedralMarker::At), ..context
            }, "*"));
        }
    }

    #[test]
    fn bracket_fields_spell_isotope_hydrogen_charge_class_and_tetrahedral_context() {
        let atom =
            AtomFacts::new(ElementSymbol::new("N").unwrap(), false, Some(15), 1, 2, 0, None).unwrap();
        let context = AtomSpellingContext {
            tetrahedral_marker: Some(TetrahedralMarker::At),
            bonds: bond_context(4, 0, 0, 0).bonds,
            aromatic_spelling: AromaticSpelling::Source,
        };
        assert!(supports(atom, context, "[15N@H2+]"));
        assert!(supports(atom, context, "[15N@TH1H2+1]"));
        let classed = facts("C", false, 4, 7, None);
        assert!(supports(classed, AtomSpellingContext::default(), "[CH4:7]"));
    }

    #[test]
    fn shared_at_prefix_keeps_selected_short_markers_live_after_style_filtering() {
        let atom = facts("C", false, 1, 0, None);
        for marker in [TetrahedralMarker::At, TetrahedralMarker::AtAt] {
            let context = AtomSpellingContext {
                tetrahedral_marker: Some(marker),
                bonds: bond_context(3, 0, 0, 0).bonds,
                aromatic_spelling: AromaticSpelling::Source,
            };
            let after_shared_at = states_after(atom, context, "[C@")
                .into_iter()
                .next()
                .expect("selected marker has a shared @ prefix");
            let unrestricted = after_shared_at.choices();
            assert!(unrestricted.iter().any(|choice| {
                choice.field == AtomSpellingField::TetrahedralMarker && choice.text == 'T'
            }));

            // This models the downstream field/character restriction without
            // making a writer regime part of the generic atom grammar.
            let allowed = unrestricted
                .iter()
                .filter(|choice| {
                    choice.field != AtomSpellingField::TetrahedralMarker || choice.text != 'T'
                })
                .collect::<Vec<_>>();
            assert!(!allowed.is_empty());
            assert!(allowed.iter().all(|choice| {
                choice.field != AtomSpellingField::TetrahedralMarker || choice.text != 'T'
            }));

            match marker {
                TetrahedralMarker::At => {
                    assert!(allowed.iter().any(|choice| choice.text == 'H'));
                    assert!(supports(atom, context, "[C@H]"));
                    assert!(supports(atom, context, "[C@TH1H]"));
                    assert!(!supports(atom, context, "[C@@H]"));
                }
                TetrahedralMarker::AtAt => {
                    assert!(allowed.iter().any(|choice| choice.text == '@'));
                    let after_second_at = allowed
                        .iter()
                        .find(|choice| choice.text == '@')
                        .unwrap()
                        .successor
                        .clone();
                    assert!(after_second_at
                        .choices()
                        .iter()
                        .any(|choice| choice.text == 'H'));
                    assert!(supports(atom, context, "[C@@H]"));
                    assert!(supports(atom, context, "[C@TH2H]"));
                    assert!(!supports(atom, context, "[C@H]"));
                }
            }
        }
    }

    #[test]
    fn neutral_charge_zero_aliases_and_hydrogen_atom_fields_follow_grammar() {
        let neutral_carbon = facts("C", false, 0, 0, None);
        let tetravalent = bond_context(4, 0, 0, 0);
        assert!(supports(neutral_carbon, tetravalent, "[C+0]"));
        assert!(supports(neutral_carbon, tetravalent, "[C-0]"));
        assert!(supports(neutral_carbon, tetravalent, "[C+00]"));
        assert!(supports(neutral_carbon, tetravalent, "[C-00]"));
        assert!(!supports(neutral_carbon, tetravalent, "[C+]"));
        assert!(!supports(neutral_carbon, tetravalent, "[C-]"));

        let hydrogen = facts("H", false, 0, 0, None);
        assert!(supports(hydrogen, AtomSpellingContext::default(), "[H]"));
        assert!(!supports(hydrogen, AtomSpellingContext::default(), "[HH]"));
        assert!(!supports(hydrogen, AtomSpellingContext::default(), "[HH0]"));
        assert!(matches!(
            AtomFacts::new(ElementSymbol::new("H").unwrap(), false, None, 0, 1, 0, None),
            Err(AtomFactsError::HydrogenAtomHasHydrogens(1))
        ));
    }

    #[test]
    fn zero_isotope_and_class_accept_long_leading_zero_prefixes() {
        let isotope_zero =
            AtomFacts::new(ElementSymbol::new("H").unwrap(), false, Some(0), 0, 0, 0, None).unwrap();
        let long_isotope = format!("[{}0H]", "0".repeat(128));
        assert!(supports(
            isotope_zero,
            AtomSpellingContext::default(),
            &long_isotope
        ));
        assert!(!supports(isotope_zero, AtomSpellingContext::default(), "[H]"));
        let isotope_absent = AtomFacts { isotope: None, ..isotope_zero };
        assert!(supports(isotope_absent, AtomSpellingContext::default(), "[H]"));
        assert!(!supports(isotope_absent, AtomSpellingContext::default(), "[0H]"));
        assert!(!supports(
            isotope_absent,
            AtomSpellingContext::default(),
            &long_isotope
        ));

        let class_zero = facts("C", false, 4, 0, None);
        let long_class = format!("[CH4:{}0]", "0".repeat(128));
        assert!(supports(
            class_zero,
            AtomSpellingContext::default(),
            &long_class
        ));
        assert!(supports(class_zero, AtomSpellingContext::default(), "[CH4]"));
    }

    #[test]
    fn bracket_spelling_preserves_nonzero_isotope() {
        let isotope_carbon = AtomFacts::new(
            ElementSymbol::new("C").unwrap(), false, Some(13), 0, 4, 0, None,
        ).unwrap();
        let context = AtomSpellingContext::default();
        assert!(supports(isotope_carbon, context, "[13CH4]"));
        assert!(supports(isotope_carbon, context, "[0013CH4]"));
        assert!(!supports(isotope_carbon, context, "[CH4]"));

        let ordinary_carbon = AtomFacts { isotope: None, ..isotope_carbon };
        assert!(supports(ordinary_carbon, context, "[CH4]"));
        assert!(!supports(ordinary_carbon, context, "[13CH4]"));
    }

    #[test]
    fn zero_class_is_optional_and_sibling_states_are_immutable() {
        let atom = facts("C", false, 4, 0, None);
        assert!(supports(atom, AtomSpellingContext::default(), "C"));
        assert!(supports(atom, AtomSpellingContext::default(), "[CH4]"));
        assert!(supports(atom, AtomSpellingContext::default(), "[CH4:0]"));

        let root = AtomSpellingState::initial(atom, AtomSpellingContext::default());
        let before = root.choices();
        let selected = before[0].successor.clone();
        assert_eq!(before, root.choices());
        assert_eq!(selected.choices(), before[0].successor.choices());
    }

    #[test]
    fn choices_identify_the_atom_field_emitting_each_character() {
        let state = AtomSpellingState::initial(facts("C", false, 0, 0, None), bond_context(4, 0, 0, 0));
        let choices = state.choices();
        assert!(choices.iter().any(|choice| {
            choice.text == '[' && choice.field == AtomSpellingField::OpenBracket
        }));
        assert!(choices.iter().any(|choice| {
            choice.text == 'C' && choice.field == AtomSpellingField::BareElement
        }));

        let bracket = choices
            .iter()
            .find(|choice| choice.field == AtomSpellingField::OpenBracket)
            .unwrap()
            .successor
            .clone();
        let element = bracket
            .choices()
            .into_iter()
            .find(|choice| choice.text == 'C')
            .unwrap();
        assert_eq!(element.field, AtomSpellingField::Element);
        assert!(element.successor.choices().iter().any(|choice| {
            choice.text == ']' && choice.field == AtomSpellingField::CloseBracket
        }));
    }

    #[test]
    fn element_table_includes_the_post_2016_iupac_symbols_and_wildcard() {
        for symbol in ELEMENT_SYMBOLS.iter().copied().chain(std::iter::once("*")) {
            assert_eq!(ElementSymbol::new(symbol).unwrap().as_str(), symbol);
        }
        assert!(supports(
            facts("Xe", false, 0, 0, None),
            AtomSpellingContext::default(),
            "[Xe]"
        ));
        assert!(supports(
            facts("*", false, 0, 0, None),
            AtomSpellingContext::default(),
            "*"
        ));
    }
}
