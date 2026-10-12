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

/// Incremental spelling language for one fixed `AtomFacts` and selected context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomSpellingState {
    facts: AtomFacts,
    context: AtomSpellingContext,
    phase: Phase,
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
        Self {
            facts,
            context,
            phase: Phase::Start,
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
                push_choice(
                    &mut choices,
                    self.facts,
                    self.context,
                    $field,
                    $text,
                    $phase,
                )
            };
        }
        match &self.phase {
            Phase::Start => {
                push!(AtomSpellingField::OpenBracket, '[', Phase::AfterOpen);
                if let Some(bare) = self.facts.bare_spelling(self.context) {
                    push_text_first(
                        bare,
                        false,
                        Phase::Complete,
                        AtomSpellingField::BareElement,
                        self.facts,
                        self.context,
                        &mut choices,
                    );
                }
            }
            Phase::AfterOpen => {
                if let Some(isotope) = self.facts.isotope {
                    push_number_start(
                        &mut choices,
                        self.facts,
                        self.context,
                        isotope as u64,
                        NumberKind::Isotope,
                    );
                }
                push_element_start(&mut choices, self.facts, self.context, true);
            }
            Phase::ElementStart => push_element_start(&mut choices, self.facts, self.context, true),
            Phase::Number(field) => {
                push_number_choices(&mut choices, self.facts, self.context, field);
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
                    push_after_marker(&mut choices, self.facts, self.context);
                }
            }
            Phase::AfterAt => match self.context.tetrahedral_marker {
                Some(TetrahedralMarker::At) => {
                    push_after_marker(&mut choices, self.facts, self.context);
                    push!(
                        AtomSpellingField::TetrahedralMarker,
                        'T',
                        Phase::Marker {
                            text: "H1",
                            offset: 0,
                        },
                    );
                }
                Some(TetrahedralMarker::AtAt) => {
                    push!(
                        AtomSpellingField::TetrahedralMarker,
                        '@',
                        Phase::AfterMarker,
                    );
                    push!(
                        AtomSpellingField::TetrahedralMarker,
                        'T',
                        Phase::Marker {
                            text: "H2",
                            offset: 0,
                        },
                    );
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
            Phase::AfterMarker => push_after_marker(&mut choices, self.facts, self.context),
            Phase::HydrogenDigit => {
                if self.facts.non_graph_hydrogens == 1 {
                    // H without a digit and H1 have the same meaning.
                    push_after_hydrogen(&mut choices, self.facts, self.context);
                    push!(AtomSpellingField::HydrogenCount, '1', Phase::ChargeStart);
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
                    push_class_start(&mut choices, self.facts, self.context);
                    push!(AtomSpellingField::Charge, '+', Phase::ChargeAfterSign);
                    push!(AtomSpellingField::Charge, '-', Phase::ChargeAfterSign);
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
                if magnitude == 2 {
                    push!(AtomSpellingField::Charge, sign, Phase::ClassStart);
                }
                if magnitude == 1 {
                    push_class_start(&mut choices, self.facts, self.context);
                }
                let digits = magnitude.to_string();
                if digits.len() == 1 {
                    push!(
                        AtomSpellingField::Charge,
                        digits.chars().next().unwrap(),
                        Phase::ClassStart
                    );
                    push!(
                        AtomSpellingField::Charge,
                        '0',
                        Phase::ChargeDigits {
                            text: digits,
                            offset: 0
                        }
                    );
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
                if self.facts.atom_class == 0 {
                    push!(AtomSpellingField::CloseBracket, ']', Phase::Complete);
                }
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
            Phase::Close => push!(AtomSpellingField::CloseBracket, ']', Phase::Complete),
            Phase::Complete => {}
        }
        choices
    }
}

fn push_after_marker(
    choices: &mut Vec<AtomSpellingChoice>,
    facts: AtomFacts,
    context: AtomSpellingContext,
) {
    if facts.element.as_str() == "H" {
        push_after_hydrogen(choices, facts, context);
        return;
    }
    if facts.non_graph_hydrogens > 0 {
        push_choice(
            choices,
            facts,
            context,
            AtomSpellingField::HydrogenCount,
            'H',
            Phase::HydrogenDigit,
        );
    } else {
        push_after_hydrogen(choices, facts, context);
        // H0 has the same non-graph hydrogen count as an omitted H field.
        push_choice(
            choices,
            facts,
            context,
            AtomSpellingField::HydrogenCount,
            'H',
            Phase::HydrogenDigit,
        );
    }
}

fn push_after_hydrogen(
    choices: &mut Vec<AtomSpellingChoice>,
    facts: AtomFacts,
    context: AtomSpellingContext,
) {
    if facts.formal_charge != 0 {
        push_choice(
            choices,
            facts,
            context,
            AtomSpellingField::Charge,
            charge_sign(facts),
            Phase::ChargeAfterSign,
        );
    } else {
        push_class_start(choices, facts, context);
        push_choice(
            choices,
            facts,
            context,
            AtomSpellingField::Charge,
            '+',
            Phase::ChargeAfterSign,
        );
        push_choice(
            choices,
            facts,
            context,
            AtomSpellingField::Charge,
            '-',
            Phase::ChargeAfterSign,
        );
    }
}

fn push_class_start(
    choices: &mut Vec<AtomSpellingChoice>,
    facts: AtomFacts,
    context: AtomSpellingContext,
) {
    if facts.atom_class == 0 {
        push_choice(
            choices,
            facts,
            context,
            AtomSpellingField::CloseBracket,
            ']',
            Phase::Complete,
        );
    }
    if facts.atom_class != 0 {
        push_choice(
            choices,
            facts,
            context,
            AtomSpellingField::AtomClass,
            ':',
            Phase::Number(NumberField {
                target: facts.atom_class.to_string(),
                offset: 0,
                kind: NumberKind::Class,
            }),
        );
    }
    if facts.atom_class == 0 {
        push_choice(
            choices,
            facts,
            context,
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
    facts: AtomFacts,
    context: AtomSpellingContext,
    value: u64,
    kind: NumberKind,
) {
    push_number_choices(
        choices,
        facts,
        context,
        &NumberField {
            target: value.to_string(),
            offset: 0,
            kind,
        },
    );
}

fn push_number_choices(
    choices: &mut Vec<AtomSpellingChoice>,
    facts: AtomFacts,
    context: AtomSpellingContext,
    field: &NumberField,
) {
    let target = field
        .target
        .chars()
        .nth(field.offset)
        .expect("numeric target offset is valid");
    let continuation = |offset| NumberField {
        target: field.target.clone(),
        offset,
        kind: field.kind,
    };
    if target != '0' {
        let next = if field.offset + 1 == field.target.len() {
            number_suffix_phase(field.kind)
        } else {
            Phase::Number(continuation(field.offset + 1))
        };
        push_choice(
            choices,
            facts,
            context,
            number_field(field.kind),
            target,
            next,
        );
        push_choice(
            choices,
            facts,
            context,
            number_field(field.kind),
            '0',
            Phase::Number(field.clone()),
        );
    } else {
        // For numeric zero, the same character can finish the field or begin
        // another leading zero. The two successors preserve both languages.
        push_choice(
            choices,
            facts,
            context,
            number_field(field.kind),
            '0',
            number_suffix_phase(field.kind),
        );
        push_choice(
            choices,
            facts,
            context,
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
    facts: AtomFacts,
    context: AtomSpellingContext,
    bracketed: bool,
) {
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
        facts,
        context,
        choices,
    );
}

fn push_text_first(
    text: &str,
    bracketed: bool,
    terminal: Phase,
    field: AtomSpellingField,
    facts: AtomFacts,
    context: AtomSpellingContext,
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
    push_choice(choices, facts, context, field, first, phase);
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
    facts: AtomFacts,
    context: AtomSpellingContext,
    field: AtomSpellingField,
    text: char,
    phase: Phase,
) {
    choices.push(AtomSpellingChoice {
        text,
        field,
        successor: AtomSpellingState {
            facts,
            context,
            phase,
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

        let class_zero = facts("C", false, 4, 0, None);
        let long_class = format!("[CH4:{}0]", "0".repeat(128));
        assert!(supports(
            class_zero,
            AtomSpellingContext::default(),
            &long_class
        ));
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
