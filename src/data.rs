//! Deterministic tokenization fitted only to training documents: Unicode
//! characters, optionally extended with byte-pair-style merges of characters.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Token zero represents characters absent from the training vocabulary.
pub const UNKNOWN_CHARACTER: char = '\u{fffd}';

/// Base characters (token 0 is unknown), then one token per learned merge.
/// Token `vocab().len() + rank` is the concatenation of merge `rank`'s pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tokenizer {
    vocabulary: Vec<char>,
    merges: Vec<(usize, usize)>,
    /// Decoded text of every token, in token order.
    pieces: Vec<String>,
    ranks: HashMap<(usize, usize), usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharacterClass {
    Letter,
    Digit,
    Space,
    Break,
    Other,
}

fn character_class(character: char) -> CharacterClass {
    if character.is_alphabetic() {
        CharacterClass::Letter
    } else if character.is_numeric() {
        CharacterClass::Digit
    } else if character == ' ' {
        CharacterClass::Space
    } else if character.is_whitespace() {
        CharacterClass::Break
    } else {
        CharacterClass::Other
    }
}

/// Pre-tokenization for merges: runs of one character class, except that the
/// last space before a letter or digit run joins it (" word"). Merges never
/// cross chunks, so tokens stay within words, numbers, or punctuation runs.
fn chunks(text: &str) -> Vec<&str> {
    let characters: Vec<(usize, char)> = text.char_indices().collect();
    if characters.is_empty() {
        return Vec::new();
    }
    let mut starts = vec![0];
    for index in 1..characters.len() {
        let previous = character_class(characters[index - 1].1);
        let current = character_class(characters[index].1);
        if previous == current {
            continue;
        }
        if previous == CharacterClass::Space
            && matches!(current, CharacterClass::Letter | CharacterClass::Digit)
        {
            if starts.last() != Some(&(index - 1)) {
                starts.push(index - 1);
            }
        } else {
            starts.push(index);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(position, &start)| {
            let end = starts
                .get(position + 1)
                .map_or(text.len(), |&next| characters[next].0);
            &text[characters[start].0..end]
        })
        .collect()
}

/// Replace every non-overlapping left-to-right occurrence of `pair`.
fn merge_pair(symbols: &[usize], pair: (usize, usize), merged: usize) -> Vec<usize> {
    let mut output = Vec::with_capacity(symbols.len());
    let mut index = 0;
    while index < symbols.len() {
        if index + 1 < symbols.len() && (symbols[index], symbols[index + 1]) == pair {
            output.push(merged);
            index += 2;
        } else {
            output.push(symbols[index]);
            index += 1;
        }
    }
    output
}

/// Greedy merge learning over weighted chunks: repeatedly merge the most
/// frequent adjacent pair (ties: smallest token pair), updating only the
/// chunks that contain it. Stops early when no pair occurs at least twice.
fn learn_merges(
    words: &mut [(Vec<usize>, u64)],
    first_token: usize,
    count: usize,
) -> Vec<(usize, usize)> {
    let mut pair_counts: HashMap<(usize, usize), u64> = HashMap::new();
    let mut occurrences: HashMap<(usize, usize), HashSet<usize>> = HashMap::new();
    for (index, (symbols, weight)) in words.iter().enumerate() {
        for pair in symbols.windows(2) {
            let pair = (pair[0], pair[1]);
            *pair_counts.entry(pair).or_default() += weight;
            occurrences.entry(pair).or_default().insert(index);
        }
    }
    let mut heap: BinaryHeap<(u64, Reverse<(usize, usize)>)> = pair_counts
        .iter()
        .map(|(&pair, &weight)| (weight, Reverse(pair)))
        .collect();
    let mut merges = Vec::with_capacity(count);
    while merges.len() < count {
        let Some((weight, Reverse(pair))) = heap.pop() else {
            break;
        };
        let current = pair_counts.get(&pair).copied().unwrap_or(0);
        if weight != current {
            // A stale heap entry; requeue the pair at its current count.
            if current > 0 {
                heap.push((current, Reverse(pair)));
            }
            continue;
        }
        if current < 2 {
            break;
        }
        let merged = first_token + merges.len();
        merges.push(pair);
        let mut affected: Vec<usize> = occurrences
            .remove(&pair)
            .unwrap_or_default()
            .into_iter()
            .collect();
        affected.sort_unstable();
        let mut changed = HashSet::new();
        for index in affected {
            let (symbols, weight) = &mut words[index];
            for window in symbols.windows(2) {
                let window = (window[0], window[1]);
                *pair_counts.get_mut(&window).expect("counted pair") -= *weight;
                changed.insert(window);
            }
            *symbols = merge_pair(symbols, pair, merged);
            for window in symbols.windows(2) {
                let window = (window[0], window[1]);
                *pair_counts.entry(window).or_default() += *weight;
                occurrences.entry(window).or_default().insert(index);
                changed.insert(window);
            }
        }
        for window in changed {
            let weight = pair_counts[&window];
            if weight > 0 {
                heap.push((weight, Reverse(window)));
            }
        }
    }
    merges
}

impl Tokenizer {
    /// Collect Unicode scalar values in sorted order, reserving index zero.
    pub fn from_text(text: &str) -> Self {
        Self::from_characters(text.chars())
    }

    /// Fit all training documents without adding artificial separator tokens.
    pub fn from_documents(documents: &[String]) -> Self {
        Self::from_characters(documents.iter().flat_map(|text| text.chars()))
    }

    fn from_characters(characters: impl Iterator<Item = char>) -> Self {
        let characters: BTreeSet<char> = characters
            .filter(|character| *character != UNKNOWN_CHARACTER)
            .collect();
        let mut vocabulary = Vec::with_capacity(characters.len() + 1);
        vocabulary.push(UNKNOWN_CHARACTER);
        vocabulary.extend(characters);
        Self::assemble(vocabulary, Vec::new())
    }

    fn assemble(vocabulary: Vec<char>, merges: Vec<(usize, usize)>) -> Self {
        let mut pieces: Vec<String> = vocabulary.iter().map(char::to_string).collect();
        for &(left, right) in &merges {
            let piece = format!("{}{}", pieces[left], pieces[right]);
            pieces.push(piece);
        }
        let ranks = merges
            .iter()
            .enumerate()
            .map(|(rank, &pair)| (pair, rank))
            .collect();
        Self {
            vocabulary,
            merges,
            pieces,
            ranks,
        }
    }

    /// Learn up to `count` merges from these (training) documents. Fewer are
    /// learned when no remaining pair occurs at least twice.
    pub fn with_merges(self, documents: &[String], count: usize) -> Result<Self, String> {
        if !self.merges.is_empty() {
            return Err("merges are learned once, from a character vocabulary".into());
        }
        let mut chunk_counts: HashMap<&str, u64> = HashMap::new();
        for document in documents {
            for chunk in chunks(document) {
                *chunk_counts.entry(chunk).or_default() += 1;
            }
        }
        let mut words: Vec<(Vec<usize>, u64)> = chunk_counts
            .into_iter()
            .map(|(chunk, weight)| (self.character_ids(chunk).collect(), weight))
            .filter(|(symbols, _): &(Vec<usize>, u64)| symbols.len() > 1)
            .collect();
        words.sort_unstable();
        let merges = learn_merges(&mut words, self.vocabulary.len(), count);
        Ok(Self::assemble(self.vocabulary, merges))
    }

    /// Restore an exact vocabulary without silently changing token indices.
    pub fn from_vocab(vocabulary: Vec<char>) -> Result<Self, String> {
        Self::from_parts(vocabulary, Vec::new())
    }

    /// Restore base characters and ranked merges exactly.
    pub fn from_parts(vocabulary: Vec<char>, merges: Vec<(usize, usize)>) -> Result<Self, String> {
        if vocabulary.first() != Some(&UNKNOWN_CHARACTER) {
            return Err("vocabulary must reserve token 0 for U+FFFD".to_string());
        }
        if vocabulary[1..].contains(&UNKNOWN_CHARACTER) {
            return Err("vocabulary contains a duplicate unknown token".to_string());
        }
        if vocabulary[1..].windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(
                "vocabulary characters after token 0 must be unique and sorted".to_string(),
            );
        }
        for (rank, &(left, right)) in merges.iter().enumerate() {
            let available = vocabulary.len() + rank;
            if left >= available || right >= available {
                return Err(format!("merge {rank} uses a token defined after it"));
            }
        }
        if merges.iter().collect::<HashSet<_>>().len() != merges.len() {
            return Err("merges must be distinct pairs".to_string());
        }
        Ok(Self::assemble(vocabulary, merges))
    }

    /// Base characters; token 0 is U+FFFD.
    pub fn vocab(&self) -> &[char] {
        &self.vocabulary
    }

    pub fn merges(&self) -> &[(usize, usize)] {
        &self.merges
    }

    /// Model vocabulary: base characters plus merged tokens.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Number of Unicode scalar values a valid token decodes to.
    pub fn token_characters(&self, token: usize) -> usize {
        self.pieces[token].chars().count()
    }

    fn character_ids<'a>(&'a self, text: &'a str) -> impl Iterator<Item = usize> + 'a {
        text.chars().map(|character| {
            if character == UNKNOWN_CHARACTER {
                0
            } else {
                self.vocabulary[1..]
                    .binary_search(&character)
                    .map_or(0, |index| index + 1)
            }
        })
    }

    /// Encode unknown characters as token zero, then apply merges in rank
    /// order within each chunk. No text normalization is applied.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if self.merges.is_empty() {
            return self.character_ids(text).collect();
        }
        let mut cache: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut tokens = Vec::new();
        for chunk in chunks(text) {
            let ids = cache.entry(chunk).or_insert_with(|| {
                let mut symbols: Vec<usize> = self.character_ids(chunk).collect();
                while let Some((rank, pair)) = symbols
                    .windows(2)
                    .filter_map(|pair| {
                        let pair = (pair[0], pair[1]);
                        self.ranks.get(&pair).map(|&rank| (rank, pair))
                    })
                    .min()
                {
                    symbols = merge_pair(&symbols, pair, self.vocabulary.len() + rank);
                }
                symbols
            });
            tokens.extend_from_slice(ids);
        }
        tokens
    }

    /// Decode exact Unicode scalar values; reject invalid token IDs.
    pub fn decode(&self, tokens: &[usize]) -> Result<String, String> {
        tokens
            .iter()
            .map(|&token| {
                self.pieces
                    .get(token)
                    .map(String::as_str)
                    .ok_or_else(|| format!("token {token} is outside the vocabulary"))
            })
            .collect()
    }
}

/// JSONL preserves one independent document per row; plain UTF-8 is one document.
/// A malformed or short document is an error, not silently dropped training data.
pub fn read_documents(path: &Path) -> Result<Vec<String>, String> {
    let file = File::open(path)
        .map_err(|error| format!("cannot read corpus {}: {error}", path.display()))?;
    let documents = if path
        .extension()
        .is_some_and(|extension| extension == "jsonl")
    {
        let mut documents = Vec::new();
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line =
                line.map_err(|error| format!("cannot read JSONL line {}: {error}", index + 1))?;
            let row: serde_json::Value = serde_json::from_str(&line)
                .map_err(|error| format!("invalid JSONL line {}: {error}", index + 1))?;
            let text = row["text"]
                .as_str()
                .ok_or_else(|| format!("JSONL line {} needs a string text field", index + 1))?;
            documents.push(text.to_owned());
        }
        documents
    } else {
        let mut text = String::new();
        BufReader::new(file)
            .read_to_string(&mut text)
            .map_err(|error| format!("corpus is not readable UTF-8: {error}"))?;
        vec![text]
    };
    if documents.is_empty() || documents.iter().any(|text| text.chars().count() < 2) {
        return Err("corpus requires nonempty documents of at least two characters each".into());
    }
    Ok(documents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_split_by_class_and_attach_one_space_to_the_next_word() {
        assert_eq!(
            chunks("Hi there, 42 cats!\n\nOk  go"),
            [
                "Hi", " there", ",", " 42", " cats", "!", "\n\n", "Ok", " ", " go"
            ]
        );
        assert!(chunks("").is_empty());
    }

    #[test]
    fn merges_are_learned_greedily_deterministically_and_encode_losslessly() {
        let documents = vec!["the the the cat".to_string()];
        let tokenizer = Tokenizer::from_documents(&documents)
            .with_merges(&documents, 2)
            .unwrap();
        // Base ids: ' '=1, a=2, c=3, e=4, h=5, t=6. "he" and "th" tie at three
        // occurrences, so the smaller pair (h,e) merges first, then (t,he).
        assert_eq!(tokenizer.merges(), [(5, 4), (6, 7)]);
        assert_eq!(tokenizer.vocab_size(), 9);
        assert_eq!(tokenizer.encode("the cat"), [8, 1, 3, 2, 6]);
        assert_eq!(tokenizer.token_characters(8), 3);
        // Merges stay inside chunks; unknown characters still round-trip as U+FFFD.
        let text = "thethe Z the";
        let encoded = tokenizer.encode(text);
        assert_eq!(encoded, [8, 8, 1, 0, 1, 8]);
        assert_eq!(tokenizer.decode(&encoded).unwrap(), "thethe \u{fffd} the");
        // Requesting more merges than any repeated pair supports stops early.
        let all = Tokenizer::from_documents(&documents)
            .with_merges(&documents, 100)
            .unwrap();
        assert!(all.merges().len() < 100);
        for document in &documents {
            assert_eq!(all.decode(&all.encode(document)).unwrap(), *document);
        }
        let restored =
            Tokenizer::from_parts(tokenizer.vocab().to_vec(), tokenizer.merges().to_vec()).unwrap();
        assert_eq!(restored, tokenizer);
    }

    #[test]
    fn invalid_merges_and_character_vocabularies_are_rejected() {
        let vocabulary = Tokenizer::from_text("ab").vocab().to_vec();
        assert!(Tokenizer::from_parts(vocabulary.clone(), vec![(1, 3)]).is_err());
        assert!(Tokenizer::from_parts(vocabulary.clone(), vec![(1, 2), (1, 2)]).is_err());
        assert!(Tokenizer::from_parts(vocabulary.clone(), vec![(1, 2), (3, 1)]).is_ok());
        assert!(Tokenizer::from_vocab(vec!['a']).is_err());
        let characters = Tokenizer::from_text("ab");
        assert_eq!(characters.encode("abz"), [1, 2, 0]);
        assert!(characters.decode(&[3]).is_err());
        assert!(characters.clone().with_merges(&["abab".into()], 1).is_ok());
        let merged = characters.with_merges(&["abab".into()], 1).unwrap();
        assert!(merged.with_merges(&["abab".into()], 1).is_err());
    }
}
