//! Bounded, reproducible preparation of original FineWeb documents.
//!
//! The public rows API follows its current dataset revision. Every page must have
//! the same observed revision; the manifest records it without claiming that the
//! endpoint can retrieve a historical revision.

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::blocking::{Client, Response};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::rng::Rng;

const DATASET: &str = "HuggingFaceFW/fineweb";
const ENDPOINT: &str = "https://datasets-server.huggingface.co/rows";
const DATASET_CARD: &str = "https://huggingface.co/datasets/HuggingFaceFW/fineweb";
const MAX_ROWS: usize = 100_000;
const PAGE_ROWS: usize = 100;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_ATTEMPTS: u32 = 10;
const MAX_RETRY_SECONDS: u64 = 120;
const OUTPUT_FILES: [&str; 3] = ["train.jsonl", "validation.jsonl", "manifest.json"];

/// Explicit bounds for a small, document-disjoint FineWeb corpus.
#[derive(Clone, Debug)]
pub struct FineWebConfig {
    pub subset: String,
    pub offset: usize,
    pub rows: usize,
    pub max_chars: usize,
    pub seed: u64,
    pub output_dir: PathBuf,
}

/// Counts describe original document text, excluding JSON serialization bytes.
#[derive(Clone, Debug)]
pub struct PreparationReport {
    pub revision: String,
    pub fetched_rows: usize,
    pub train_documents: usize,
    pub validation_documents: usize,
    pub train_characters: usize,
    pub validation_characters: usize,
    pub duplicates_skipped: usize,
    pub rejected_rows: usize,
}

struct Document {
    row_idx: usize,
    row: Value,
    text_sha256: String,
    characters: usize,
}

#[derive(Default)]
struct Collection {
    documents: Vec<Document>,
    ids: HashSet<String>,
    text_hashes: HashSet<String>,
    characters: usize,
    fetched_rows: usize,
    duplicates_skipped: usize,
    rejected_rows: usize,
    malformed_or_truncated_rows: usize,
    over_budget_rows: usize,
}

impl Collection {
    fn accept(&mut self, value: &Value, max_chars: usize) {
        self.fetched_rows += 1;
        let Some(document) = parse_document(value) else {
            self.rejected_rows += 1;
            self.malformed_or_truncated_rows += 1;
            return;
        };
        // parse_document has checked the ID's type and presence.
        let id = document.row["id"].as_str().expect("validated document ID");
        if self.ids.contains(id) || self.text_hashes.contains(&document.text_sha256) {
            self.duplicates_skipped += 1;
            return;
        }
        // Never cut a document to fit. JSONL boundaries need no text separator.
        if document.characters > max_chars.saturating_sub(self.characters) {
            self.rejected_rows += 1;
            self.over_budget_rows += 1;
            return;
        }
        self.ids.insert(id.to_owned());
        self.text_hashes.insert(document.text_sha256.clone());
        self.characters += document.characters;
        self.documents.push(document);
    }
}

/// Fetch a bounded current-revision sample and atomically publish a new corpus.
///
/// No existing nonempty directory is overwritten. The output has one original
/// FineWeb row object per JSONL line, preserving complete document boundaries.
pub fn prepare(config: &FineWebConfig) -> Result<PreparationReport, String> {
    validate_config(config)?;
    ensure_empty_output(&config.output_dir)?;
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(15))
        .user_agent("Liminal/FineWeb-preparation")
        .build()
        .map_err(|error| format!("cannot construct FineWeb HTTP client: {error}"))?;

    let mut collection = Collection::default();
    let mut revision = None;
    let mut pages = Vec::new();
    while collection.fetched_rows < config.rows {
        let offset = config.offset + collection.fetched_rows;
        let length = PAGE_ROWS.min(config.rows - collection.fetched_rows);
        let response = request_page(&client, config, offset, length)?;
        let page_revision = response
            .headers()
            .get("x-revision")
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.trim().is_empty())
            .ok_or("FineWeb response has no usable X-Revision header")?
            .to_owned();
        observe_revision(&mut revision, &page_revision)?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
        {
            return Err(format!("FineWeb page at row {offset} exceeds 16 MiB"));
        }
        let body = read_bounded(response)
            .map_err(|error| format!("FineWeb page at row {offset}: {error}"))?;
        let page = parse_page(&body)?;
        let rows = page["rows"].as_array().expect("validated page rows");
        if rows.len() > length {
            return Err("FineWeb returned more rows than requested".to_owned());
        }
        for (index, row) in rows.iter().enumerate() {
            // Gaps or reordered source indices would invalidate request provenance.
            if row["row_idx"].as_u64() != Some((offset + index) as u64) {
                return Err(format!(
                    "FineWeb returned an unexpected row index at {offset}"
                ));
            }
            collection.accept(row, config.max_chars);
        }
        pages.push(json!({
            "offset": offset,
            "requested_rows": length,
            "returned_rows": rows.len(),
            "revision": page_revision,
            "response_sha256": sha256(&body),
        }));
        if rows.len() < length {
            break;
        }
    }
    let revision = revision.ok_or("FineWeb returned no dataset revision")?;
    let validation = split_documents(collection.documents.len(), config.seed)?;
    let (train, validation_text) = serialize_splits(&collection.documents, &validation)?;
    let report = report(&collection, &validation, revision);
    let manifest = manifest(
        config,
        &collection,
        &validation,
        &report,
        pages,
        &train,
        &validation_text,
    );
    let manifest = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("cannot serialize FineWeb manifest: {error}"))?;
    publish(&config.output_dir, [&train, &validation_text, &manifest])?;
    Ok(report)
}

fn request_page(
    client: &Client,
    config: &FineWebConfig,
    offset: usize,
    length: usize,
) -> Result<Response, String> {
    let mut attempt = 1;
    loop {
        let mut retry_after = None;
        let response = client
            .get(ENDPOINT)
            .query(&[
                ("dataset", DATASET.to_owned()),
                ("config", config.subset.clone()),
                ("split", "train".to_owned()),
                ("offset", offset.to_string()),
                ("length", length.to_string()),
            ])
            .send();
        match response {
            Ok(response) if response.status().is_success() => return Ok(response),
            Ok(response) => {
                let retry = matches!(response.status().as_u16(), 429 | 500 | 502 | 503 | 504);
                retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.trim().parse().ok());
                if !retry || attempt == MAX_ATTEMPTS {
                    let code = response
                        .headers()
                        .get("x-error-code")
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or("unspecified");
                    return Err(format!(
                        "FineWeb request at row {offset} failed after {attempt} attempt(s) with HTTP {} (X-Error-Code: {code})",
                        response.status()
                    ));
                }
            }
            Err(error) => {
                if !(error.is_timeout() || error.is_connect()) || attempt == MAX_ATTEMPTS {
                    return Err(format!(
                        "FineWeb request at row {offset} failed after {attempt} attempt(s): {error}"
                    ));
                }
            }
        }
        std::thread::sleep(retry_delay(attempt, retry_after));
        attempt += 1;
    }
}

/// Rate-limit windows last minutes, so back off exponentially (1, 2, 4, ... s,
/// capped) and honor a numeric Retry-After header when the server sends one.
fn retry_delay(attempt: u32, retry_after: Option<u64>) -> Duration {
    let backoff = 1_u64 << (attempt - 1).min(7);
    Duration::from_secs(retry_after.unwrap_or(backoff).min(MAX_RETRY_SECONDS))
}

fn validate_config(config: &FineWebConfig) -> Result<(), String> {
    if config.subset.trim().is_empty() {
        return Err("FineWeb subset must not be empty".to_owned());
    }
    if !(2..=MAX_ROWS).contains(&config.rows) {
        return Err(format!("FineWeb rows must be between 2 and {MAX_ROWS}"));
    }
    if config.max_chars == 0 {
        return Err("FineWeb character budget must be positive".to_owned());
    }
    if config.offset.checked_add(config.rows).is_none() {
        return Err("FineWeb offset plus row count exceeds the supported range".to_owned());
    }
    if config.output_dir.as_os_str().is_empty() || config.output_dir.file_name().is_none() {
        return Err("FineWeb output must name a corpus directory".to_owned());
    }
    Ok(())
}

fn parse_page(body: &[u8]) -> Result<Value, String> {
    let page: Value = serde_json::from_slice(body)
        .map_err(|error| format!("invalid FineWeb page JSON: {error}"))?;
    if page["partial"].as_bool() != Some(false) {
        return Err("FineWeb page is partial or has no explicit partial:false flag".to_owned());
    }
    if !page["rows"].is_array() {
        return Err("FineWeb page has no rows array".to_owned());
    }
    Ok(page)
}

fn read_bounded(reader: impl Read) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    reader
        .take((MAX_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut body)
        .map_err(|error| format!("cannot read response: {error}"))?;
    if body.len() > MAX_RESPONSE_BYTES {
        return Err("response exceeds 16 MiB".to_owned());
    }
    Ok(body)
}

fn observe_revision(revision: &mut Option<String>, next: &str) -> Result<(), String> {
    if next.len() != 40 || !next.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("FineWeb X-Revision must be a 40-digit Git commit SHA".to_owned());
    }
    match revision {
        Some(current) if current != next => Err(
            "FineWeb revision changed between pages; no corpus was published, retry preparation"
                .to_owned(),
        ),
        Some(_) => Ok(()),
        None => {
            *revision = Some(next.to_owned());
            Ok(())
        }
    }
}

fn parse_document(value: &Value) -> Option<Document> {
    let truncated = value["truncated_cells"].as_array()?;
    if !truncated.is_empty() {
        return None;
    }
    let row_idx = usize::try_from(value["row_idx"].as_u64()?).ok()?;
    let row = value.get("row")?;
    row.as_object()?;
    let text = row["text"].as_str()?;
    let characters = text.chars().count();
    if characters < 2 || text.trim().is_empty() || row["id"].as_str()?.is_empty() {
        return None;
    }
    for field in ["url", "dump", "date"] {
        row[field].as_str()?;
    }
    Some(Document {
        row_idx,
        row: row.clone(),
        text_sha256: sha256(text.as_bytes()),
        characters,
    })
}

/// Shuffle membership only, keeping source order within each split.
fn split_documents(count: usize, seed: u64) -> Result<Vec<bool>, String> {
    if count < 2 {
        return Err("FineWeb preparation requires at least two complete, distinct documents; increase rows or the character budget".to_owned());
    }
    let mut indices: Vec<usize> = (0..count).collect();
    let mut rng = Rng::new(seed);
    for index in (1..count).rev() {
        indices.swap(index, rng.index(index + 1));
    }
    let validation_count = count.div_ceil(10).clamp(1, count - 1);
    let mut validation = vec![false; count];
    for &index in &indices[..validation_count] {
        validation[index] = true;
    }
    Ok(validation)
}

fn serialize_splits(
    documents: &[Document],
    validation: &[bool],
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (document, &is_validation) in documents.iter().zip(validation) {
        let destination = if is_validation {
            &mut valid
        } else {
            &mut train
        };
        serde_json::to_writer(&mut *destination, &document.row)
            .map_err(|error| format!("cannot serialize FineWeb document: {error}"))?;
        destination.push(b'\n');
    }
    Ok((train, valid))
}

fn report(collection: &Collection, validation: &[bool], revision: String) -> PreparationReport {
    let mut report = PreparationReport {
        revision,
        fetched_rows: collection.fetched_rows,
        train_documents: 0,
        validation_documents: 0,
        train_characters: 0,
        validation_characters: 0,
        duplicates_skipped: collection.duplicates_skipped,
        rejected_rows: collection.rejected_rows,
    };
    for (document, &is_validation) in collection.documents.iter().zip(validation) {
        if is_validation {
            report.validation_documents += 1;
            report.validation_characters += document.characters;
        } else {
            report.train_documents += 1;
            report.train_characters += document.characters;
        }
    }
    report
}

fn manifest(
    config: &FineWebConfig,
    collection: &Collection,
    validation: &[bool],
    report: &PreparationReport,
    pages: Vec<Value>,
    train: &[u8],
    valid: &[u8],
) -> Value {
    let documents: Vec<Value> = collection
        .documents
        .iter()
        .zip(validation)
        .map(|(document, &is_validation)| {
            json!({
                "row_idx": document.row_idx,
                "id": document.row["id"],
                "url": document.row["url"],
                "dump": document.row["dump"],
                "date": document.row["date"],
                "text_sha256": document.text_sha256,
                "characters": document.characters,
                "split": if is_validation { "validation" } else { "train" },
            })
        })
        .collect();
    json!({
        "format": "liminal.fineweb.prepared",
        "source": {
            "dataset": DATASET,
            "config": config.subset,
            "split": "train",
            "revision": report.revision,
            "revision_header": "X-Revision",
            "endpoint": ENDPOINT,
            "dataset_card": DATASET_CARD,
            "revision_semantics": "Current dataset revision observed on every page; /rows does not support requesting a historical revision.",
            "license": "odc-by",
            "license_notes": "Original web content may have additional rights; see the official dataset card.",
        },
        "request": {
            "offset": config.offset,
            "row_budget": config.rows,
            "character_budget": config.max_chars,
            "seed": config.seed,
            "characters_unit": "Unicode scalar values in complete original document text",
        },
        "split": {
            "method": "Seeded Fisher-Yates shuffle of accepted document membership; source order retained within each split",
            "validation_documents_rule": "ceil(accepted_documents / 10), clamped to [1, accepted_documents - 1]",
            "deduplication": "Exact FineWeb ID or exact UTF-8 text SHA-256, before splitting",
            "boundaries": "One complete original FineWeb row object per JSONL line; never concatenate documents for next-token pairs",
        },
        "counts": {
            "fetched_rows": report.fetched_rows,
            "accepted_documents": collection.documents.len(),
            "train_documents": report.train_documents,
            "validation_documents": report.validation_documents,
            "train_characters": report.train_characters,
            "validation_characters": report.validation_characters,
            "duplicates_skipped": report.duplicates_skipped,
            "rejected_rows": report.rejected_rows,
            "malformed_or_truncated_rows": collection.malformed_or_truncated_rows,
            "over_budget_rows": collection.over_budget_rows,
        },
        "files": {
            "train.jsonl": { "sha256": sha256(train), "bytes": train.len() },
            "validation.jsonl": { "sha256": sha256(valid), "bytes": valid.len() },
        },
        "pages": pages,
        "documents": documents,
    })
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn ensure_empty_output(path: &Path) -> Result<(), String> {
    match fs::read_dir(path) {
        Ok(mut entries) => match entries.next() {
            None => Ok(()),
            Some(Ok(_)) => Err(format!(
                "FineWeb output {} is not empty; choose a new corpus directory",
                path.display()
            )),
            Some(Err(error)) => Err(format!(
                "cannot inspect FineWeb output {}: {error}",
                path.display()
            )),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "cannot inspect FineWeb output {}: {error}",
            path.display()
        )),
    }
}

/// Stage all three files together; failed writes never expose a prepared corpus.
fn publish(path: &Path, contents: [&[u8]; 3]) -> Result<(), String> {
    ensure_empty_output(path)?;
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("cannot create FineWeb output parent: {error}"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("cannot choose FineWeb staging directory: {error}"))?
        .as_nanos();
    let staging = parent.join(format!(".fineweb-prepare-{}-{nonce}", std::process::id()));
    fs::create_dir(&staging)
        .map_err(|error| format!("cannot create FineWeb staging directory: {error}"))?;
    let result = (|| {
        for (name, bytes) in OUTPUT_FILES.iter().zip(contents) {
            fs::write(staging.join(name), bytes)
                .map_err(|error| format!("cannot write FineWeb {name}: {error}"))?;
        }
        ensure_empty_output(path)?;
        if path.exists() {
            // remove_dir can only remove an empty directory, even if it changes.
            fs::remove_dir(path).map_err(|error| {
                format!("cannot replace empty FineWeb output directory: {error}")
            })?;
        }
        fs::rename(&staging, path)
            .map_err(|error| format!("cannot publish FineWeb corpus: {error}"))
    })();
    if result.is_err() {
        for name in OUTPUT_FILES {
            let _ = fs::remove_file(staging.join(name));
        }
        let _ = fs::remove_dir(&staging);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(index: usize, id: &str, text: &str) -> Value {
        json!({
            "row_idx": index,
            "truncated_cells": [],
            "row": { "id": id, "text": text, "url": "https://example.com/", "dump": "CC-MAIN-example", "date": "2024-01-01", "language": "en" },
        })
    }

    #[test]
    fn unicode_budget_counts_scalars_and_keeps_whole_documents() {
        let mut collection = Collection::default();
        collection.accept(&row(0, "a", "é🦀"), 5);
        collection.accept(&row(1, "b", "ab\nc"), 5);
        collection.accept(&row(2, "c", "水火木"), 5);
        assert_eq!(collection.characters, 5);
        assert_eq!(collection.documents.len(), 2);
        assert_eq!(collection.documents[0].row["text"], "é🦀");
        assert_eq!(collection.documents[1].row["text"], "水火木");
        assert_eq!(collection.over_budget_rows, 1);
    }

    #[test]
    fn duplicate_ids_and_exact_text_cannot_cross_splits() {
        let mut collection = Collection::default();
        collection.accept(&row(0, "a", "original"), 1000);
        collection.accept(&row(1, "a", "changed text same id"), 1000);
        collection.accept(&row(2, "b", "original"), 1000);
        collection.accept(&row(3, "c", "different"), 1000);
        assert_eq!(collection.documents.len(), 2);
        assert_eq!(collection.duplicates_skipped, 2);
        let membership = split_documents(collection.documents.len(), 42).unwrap();
        let report = report(&collection, &membership, "revision".to_owned());
        assert_eq!(report.train_documents, 1);
        assert_eq!(report.validation_documents, 1);
        assert_eq!(report.fetched_rows, 4);
        assert_eq!(report.rejected_rows, 0);
    }

    #[test]
    fn splitting_is_deterministic_disjoint_and_preserves_original_text() {
        let mut collection = Collection::default();
        for index in 0..21 {
            collection.accept(
                &row(
                    index,
                    &index.to_string(),
                    &format!("doc {index}\n\ncomplete\n"),
                ),
                1000,
            );
        }
        let membership = split_documents(21, 42).unwrap();
        assert_eq!(membership, split_documents(21, 42).unwrap());
        assert_eq!(membership.iter().filter(|&&value| value).count(), 3);
        let (train, valid) = serialize_splits(&collection.documents, &membership).unwrap();
        let mut ids = HashSet::new();
        for bytes in [train, valid] {
            let text = String::from_utf8(bytes).unwrap();
            let rows: Vec<Value> = text
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let indices: Vec<usize> = rows
                .iter()
                .map(|value| {
                    let id = value["id"].as_str().unwrap();
                    assert!(ids.insert(id.to_owned()));
                    let index = id.parse::<usize>().unwrap();
                    assert_eq!(value["text"], format!("doc {index}\n\ncomplete\n"));
                    assert_eq!(value["language"], "en");
                    index
                })
                .collect();
            assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        }
        assert_eq!(ids.len(), 21);
    }

    #[test]
    fn malformed_truncated_blank_and_short_documents_are_rejected() {
        let mut collection = Collection::default();
        let mut truncated = row(0, "a", "complete text");
        truncated["truncated_cells"] = json!(["text"]);
        collection.accept(&truncated, 1000);
        collection.accept(&json!({ "row_idx": 1, "row": { "text": "bad" } }), 1000);
        collection.accept(&row(2, "b", " \n\t"), 1000);
        collection.accept(&row(3, "c", "🦀"), 1000);
        let mut malformed = row(4, "d", "text");
        malformed["row"]["date"] = Value::Null;
        collection.accept(&malformed, 1000);
        assert_eq!(collection.rejected_rows, 5);
        assert!(collection.documents.is_empty());
        assert!(split_documents(0, 42).is_err());
        assert!(split_documents(1, 42).is_err());
    }

    #[test]
    fn partial_pages_and_changing_revisions_fail() {
        assert!(parse_page(br#"{"partial":true,"rows":[]}"#).is_err());
        assert!(parse_page(br#"{"rows":[]}"#).is_err());
        assert!(parse_page(br#"{"partial":false,"rows":[]}"#).is_ok());
        assert!(parse_page(br#"{"partial":false,"rows":{}}"#).is_err());
        let mut revision = None;
        let first = "0123456789012345678901234567890123456789";
        observe_revision(&mut revision, first).unwrap();
        observe_revision(&mut revision, first).unwrap();
        assert!(
            observe_revision(&mut revision, "abcdef0123456789012345678901234567890123").is_err()
        );
        assert!(observe_revision(&mut revision, "invalid").is_err());
        assert_eq!(revision.as_deref(), Some(first));
    }

    #[test]
    fn sha256_uses_the_standard_exact_byte_digest() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn configuration_requires_finite_row_and_character_bounds() {
        let mut config = FineWebConfig {
            subset: "sample-10BT".to_owned(),
            offset: 0,
            rows: 128,
            max_chars: 1000,
            seed: 42,
            output_dir: PathBuf::from("data/fineweb"),
        };
        assert!(validate_config(&config).is_ok());
        config.rows = usize::MAX;
        assert!(validate_config(&config).is_err());
        config.rows = 2;
        config.offset = usize::MAX;
        assert!(validate_config(&config).is_err());
        config.offset = 0;
        config.max_chars = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn response_reader_enforces_the_byte_limit_without_network() {
        assert_eq!(read_bounded(&b"small"[..]).unwrap(), b"small");
        let oversized = vec![0_u8; MAX_RESPONSE_BYTES + 1];
        assert!(read_bounded(oversized.as_slice()).is_err());
    }

    #[test]
    fn retries_back_off_exponentially_and_honor_retry_after_with_a_cap() {
        let seconds = |attempt, after| retry_delay(attempt, after).as_secs();
        assert_eq!(
            [1, 2, 3, 4].map(|attempt| seconds(attempt, None)),
            [1, 2, 4, 8]
        );
        assert_eq!(seconds(9, None), MAX_RETRY_SECONDS);
        assert_eq!(seconds(1, Some(30)), 30);
        assert_eq!(seconds(1, Some(3600)), MAX_RETRY_SECONDS);
    }

    #[test]
    fn publication_refuses_to_overwrite_an_existing_corpus() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!(
                "fineweb-publish-test-{}-{nonce}",
                std::process::id()
            ));
        publish(&root, [b"train", b"validation", b"manifest"]).unwrap();
        assert!(publish(&root, [b"changed", b"changed", b"changed"]).is_err());
        assert_eq!(fs::read(root.join("train.jsonl")).unwrap(), b"train");
        assert_eq!(
            fs::read(root.join("validation.jsonl")).unwrap(),
            b"validation"
        );
        for name in OUTPUT_FILES {
            fs::remove_file(root.join(name)).unwrap();
        }
        fs::remove_dir(root).unwrap();
    }
}
