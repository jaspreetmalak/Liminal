//! Fixed, balanced language contrasts for frozen-model diagnostics only.
//! Labels describe the printed sentences; no model training uses these cases.

pub(crate) struct ContrastPair {
    pub id: String,
    pub family: &'static str,
    pub contexts: [String; 2],
    pub candidates: [String; 2],
    pub gold: [usize; 2],
    pub query: String,
    pub first_mention: [usize; 2],
    pub last_mention: [usize; 2],
}

/// Four families x four name pairs x two entity orders x two questions.
/// Each member of a pair has the same question and opposite correct answers.
pub(crate) fn dataset() -> Vec<ContrastPair> {
    let mut pairs = Vec::with_capacity(64);
    for family in ["binding", "negation", "order", "semantic_roles"] {
        for names in [
            ["Ann", "Bob"],
            ["Eve", "Dan"],
            ["Amy", "Tom"],
            ["Sam", "Lee"],
        ] {
            for reversed in [false, true] {
                let first = usize::from(reversed);
                let last = 1 - first;
                let subject = names[first];
                let object = names[last];
                for question in 0..2 {
                    let (stories, query, candidates, gold, first_mention, last_mention) =
                        match family {
                            "binding" => {
                                let first_answer = if question == first { 0 } else { 1 };
                                (
                                    [
                                        format!("{subject} wears red. {object} wears tan."),
                                        format!("{subject} wears tan. {object} wears red."),
                                    ],
                                    format!("{} wears", names[question]),
                                    [" red.".into(), " tan.".into()],
                                    [first_answer, 1 - first_answer],
                                    [0, 1],
                                    [1, 0],
                                )
                            }
                            "negation" => {
                                let first_answer = if question == 0 { first } else { last };
                                (
                                    [
                                        format!("{subject} is wet. {object} is not wet."),
                                        format!("{subject} is not wet. {object} is wet."),
                                    ],
                                    if question == 0 {
                                        "Who is wet?"
                                    } else {
                                        "Who is not wet?"
                                    }
                                    .into(),
                                    names.map(|name| format!(" {name}.")),
                                    [first_answer, 1 - first_answer],
                                    [first; 2],
                                    [last; 2],
                                )
                            }
                            "order" => {
                                let first_answer = if question == 0 { first } else { last };
                                (
                                    [
                                        format!("{subject} came before {object}."),
                                        format!("{subject} came after {object}."),
                                    ],
                                    if question == 0 {
                                        "First came"
                                    } else {
                                        "Last came"
                                    }
                                    .into(),
                                    names.map(|name| format!(" {name}.")),
                                    [first_answer, 1 - first_answer],
                                    [first; 2],
                                    [last; 2],
                                )
                            }
                            "semantic_roles" => {
                                let first_answer = if question == 0 { first } else { last };
                                (
                                    [
                                        format!("{subject} thanked {object}."),
                                        format!("{subject} was thanked by {object}."),
                                    ],
                                    if question == 0 {
                                        "Who gave thanks?"
                                    } else {
                                        "Who received thanks?"
                                    }
                                    .into(),
                                    names.map(|name| format!(" {name}.")),
                                    [first_answer, 1 - first_answer],
                                    [first; 2],
                                    [last; 2],
                                )
                            }
                            _ => unreachable!("fixed family inventory"),
                        };
                    pairs.push(ContrastPair {
                        id: format!(
                            "{family}_{}_{}_order{first}_query{question}",
                            names[0], names[1]
                        ),
                        family,
                        contexts: stories.map(|story| format!("{story} {query}")),
                        candidates,
                        gold,
                        query,
                        first_mention,
                        last_mention,
                    });
                }
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn story<'a>(pair: &ContrastPair, context: &'a str) -> &'a str {
        context.strip_suffix(&pair.query).unwrap().trim_end()
    }

    // Parse the actual language rather than recovering labels from case IDs,
    // generator loop indices, or the supplied gold/heuristic arrays.
    fn answer_from_printed_text(pair: &ContrastPair, context: &str) -> usize {
        let story = story(pair, context);
        let answer = match pair.family {
            "binding" => {
                let queried_person = pair.query.strip_suffix(" wears").unwrap();
                story
                    .split('.')
                    .filter(|clause| !clause.trim().is_empty())
                    .find_map(|clause| {
                        let (person, color) = clause.trim().split_once(" wears ").unwrap();
                        (person == queried_person).then_some(color)
                    })
                    .unwrap()
            }
            "negation" => {
                let wants_wet = match pair.query.as_str() {
                    "Who is wet?" => true,
                    "Who is not wet?" => false,
                    _ => panic!("unexpected wetness question"),
                };
                story
                    .split('.')
                    .filter(|clause| !clause.trim().is_empty())
                    .find_map(|clause| {
                        let (person, predicate) = clause.trim().split_once(" is ").unwrap();
                        let wet = match predicate {
                            "wet" => true,
                            "not wet" => false,
                            _ => panic!("unexpected wetness predicate"),
                        };
                        (wet == wants_wet).then_some(person)
                    })
                    .unwrap()
            }
            "order" => {
                let words: Vec<_> = story.trim_end_matches('.').split_whitespace().collect();
                assert_eq!(words.len(), 4);
                assert_eq!(words[1], "came");
                let chronological = match words[2] {
                    "before" => [words[0], words[3]],
                    "after" => [words[3], words[0]],
                    _ => panic!("unexpected ordering relation"),
                };
                match pair.query.as_str() {
                    "First came" => chronological[0],
                    "Last came" => chronological[1],
                    _ => panic!("unexpected ordering question"),
                }
            }
            "semantic_roles" => {
                let sentence = story.trim_end_matches('.');
                let (giver, receiver) =
                    if let Some((receiver, giver)) = sentence.split_once(" was thanked by ") {
                        (giver, receiver)
                    } else {
                        sentence.split_once(" thanked ").unwrap()
                    };
                match pair.query.as_str() {
                    "Who gave thanks?" => giver,
                    "Who received thanks?" => receiver,
                    _ => panic!("unexpected role question"),
                }
            }
            _ => panic!("unexpected family"),
        };
        pair.candidates
            .iter()
            .position(|candidate| candidate == &format!(" {answer}."))
            .unwrap()
    }

    #[test]
    fn printed_sentence_oracle_verifies_every_label_and_opposite_answer_pair() {
        let pairs = dataset();
        assert_eq!(pairs.len(), 64);
        for pair in pairs {
            for (context, &gold) in pair.contexts.iter().zip(&pair.gold) {
                assert_eq!(
                    answer_from_printed_text(&pair, context),
                    gold,
                    "{}: {context}",
                    pair.id
                );
            }
            assert_eq!(pair.gold[0] + pair.gold[1], 1);
        }
    }

    #[test]
    fn full_factorial_and_story_mention_controls_are_balanced_per_family() {
        let mut counts: BTreeMap<&str, [usize; 5]> = BTreeMap::new();
        for pair in dataset() {
            let row = counts.entry(pair.family).or_default();
            for (index, context) in pair.contexts.iter().enumerate() {
                let story = story(&pair, context);
                let mentions: Vec<_> = pair
                    .candidates
                    .iter()
                    .map(|candidate| {
                        let word = candidate.trim().trim_end_matches('.');
                        (story.find(word).unwrap(), story.rfind(word).unwrap())
                    })
                    .collect();
                let first = usize::from(mentions[1].0 < mentions[0].0);
                let last = usize::from(mentions[1].1 > mentions[0].1);
                assert_eq!(pair.first_mention[index], first);
                assert_eq!(pair.last_mention[index], last);
                row[0] += 1;
                row[1] += usize::from(pair.gold[index] == 0);
                row[2] += usize::from(pair.gold[index] == 1);
                row[3] += usize::from(first == pair.gold[index]);
                row[4] += usize::from(last == pair.gold[index]);
            }
        }
        assert_eq!(counts.len(), 4);
        for (family, row) in counts {
            assert_eq!(row, [32, 16, 16, 16, 16], "imbalanced family: {family}");
        }
    }

    #[test]
    fn unique_bounded_ascii_contexts_have_common_queries_and_equal_length_candidates() {
        let mut ids = BTreeSet::new();
        let mut contexts = BTreeSet::new();
        for pair in dataset() {
            assert!(ids.insert(pair.id.clone()));
            assert_eq!(pair.candidates[0].chars().count(), 5);
            assert_eq!(pair.candidates[1].chars().count(), 5);
            assert_ne!(pair.candidates[0], pair.candidates[1]);
            for context in &pair.contexts {
                assert!(contexts.insert(context.clone()));
                assert!(context.is_ascii());
                assert!(context.ends_with(&pair.query));
                let printed_story = story(&pair, context);
                assert!(printed_story.ends_with('.'));
                assert_eq!(context, &format!("{printed_story} {}", pair.query));
                assert!(
                    !context.contains("gold")
                        && !context.contains("query0")
                        && !context.contains("query1")
                );
                for candidate in &pair.candidates {
                    assert!(candidate.starts_with(' ') && candidate.ends_with('.'));
                    assert!(context.chars().count() + candidate.chars().count() <= 64);
                }
            }
        }
        assert_eq!(contexts.len(), 128);
        assert_eq!(ids.len(), 64);
    }
}
