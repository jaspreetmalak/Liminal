use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use liminal::checkpoint;
use liminal::data::{Tokenizer, read_documents};
use liminal::fineweb::{self, FineWebConfig};
use liminal::model::{FeedbackMode, Model, ModelConfig, VoltagePenalty};
use liminal::runtime;
use liminal::training::{self, Decoding, LearningRateSchedule, TrainConfig};

const USAGE: &str = "Liminal: a Rust spiking neural network for character language modeling

Usage:
  liminal prepare-fineweb [OPTIONS]
  liminal train [OPTIONS]
  liminal generate [OPTIONS]
  liminal evaluate [OPTIONS]
  liminal --help

prepare-fineweb options:
  --output PATH            Prepared corpus directory (default: data/fineweb)
  --subset NAME            FineWeb subset (default: sample-10BT)
  --offset INTEGER         First source row (default: 0)
  --rows INTEGER           Maximum source rows to inspect (default: 128)
  --max-chars INTEGER      Maximum retained characters (default: 200000)
  --seed INTEGER           Document split seed (default: 42)

train options:
  --data PATH              UTF-8 or JSONL corpus (default: data/fineweb/train.jsonl)
  --validation-data PATH   Separate validation corpus (default with FineWeb: data/fineweb/validation.jsonl)
  --checkpoint PATH        Save model and vocabulary (default: artifacts/liminal.bin)
  --steps INTEGER          Optimizer updates (default: 500)
  --batch-size INTEGER     Independent windows per update (default: 16)
  --device gpu|cpu         Burn WGPU or Flex device (default: gpu)
  --hidden INTEGER         Spiking hidden neurons (default: 48)
  --feedback MODE          both|top-down|bottom-up|neither (default: both)
  --sequence-length INT    Truncated training sequence length (default: 32)
  --learning-rate FLOAT    Adam learning rate (default: 0.003)
  --lr-schedule NAME      constant|cosine; cosine decays to 10% by the last update (default: constant)
  --merges INTEGER        Subword merges learned from training documents; 0 keeps characters (default: 0)
  --gradient-clip FLOAT    Maximum global gradient norm (default: 1.0)
  --voltage-penalty FLOAT  Training-only voltage penalty coefficient (default: 0)
  --voltage-band-widths F  Allowed distance from threshold in surrogate widths (default: 3)
  --seed INTEGER           Initialization and training seed (default: 42)

generate options:
  --device gpu|cpu         Burn WGPU or Flex device (default: gpu)
  --checkpoint PATH        Model and vocabulary (default: artifacts/liminal.bin)
  --prompt TEXT            Initial text (default: The )
  --tokens INTEGER         Number of additional tokens: characters, or subword tokens with merges (default: 160)
  --temperature FLOAT      Sampling temperature; 0 is greedy (default: 0.8)
  --top-p FLOAT            Nucleus sampling mass in (0, 1]; 1 keeps all tokens (default: 1)
  --repetition-penalty F   Penalty >= 1 for tokens in the last 64; 1 disables it (default: 1)
  --seed INTEGER           Sampling seed (default: 42)

evaluate options:
  --device gpu|cpu         Burn WGPU or Flex device (default: gpu)
  --batch-size INTEGER     Windows per evaluation batch (default: 16)
  --checkpoint PATH        Model and vocabulary (default: artifacts/liminal.bin)
  --data PATH              Evaluation corpus (default: data/fineweb/validation.jsonl)
  --sequence-length INT    Independent evaluation window length (default: 32)

FineWeb uses separate whole-document splits; JSONL training needs validation data.
Custom plain text without validation data is split 90/10 by character. Vocabulary
is fitted only to training data. Model state and targets never cross documents.
Evaluation loss is next-character cross-entropy in nats; perplexity is exp(loss).
Optional voltage regularization affects training only; inference is unchanged.
";

type Options = HashMap<String, String>;

fn main() {
    if let Err(error) = run(env::args().skip(1).collect()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(arguments: Vec<String>) -> Result<(), String> {
    let Some(command) = arguments.first() else {
        print!("{USAGE}");
        return Ok(());
    };
    if arguments.len() == 1 && matches!(command.as_str(), "--help" | "-h" | "help") {
        print!("{USAGE}");
        return Ok(());
    }
    if !matches!(
        command.as_str(),
        "prepare-fineweb" | "train" | "generate" | "evaluate"
    ) {
        return Err(format!("unknown command {command:?}; run liminal --help"));
    }
    if arguments.len() == 2 && matches!(arguments[1].as_str(), "--help" | "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    match command.as_str() {
        "prepare-fineweb" => prepare_fineweb(&arguments[1..]),
        "train" => train(&arguments[1..]),
        "generate" => generate(&arguments[1..]),
        "evaluate" => evaluate(&arguments[1..]),
        _ => unreachable!(),
    }
}

fn parse_options(arguments: &[String], allowed: &[&str]) -> Result<Options, String> {
    let mut options = HashMap::new();
    let mut index = 0;
    while index < arguments.len() {
        let name = &arguments[index];
        if !allowed.contains(&name.as_str()) {
            return Err(format!("unknown option {name:?}; run liminal --help"));
        }
        if options.contains_key(name) {
            return Err(format!("duplicate option {name}"));
        }
        let value = arguments
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("option {name} requires a value"))?;
        options.insert(name.clone(), value.clone());
        index += 2;
    }
    Ok(options)
}

fn value<'a>(options: &'a Options, name: &str, default: &'a str) -> &'a str {
    options.get(name).map_or(default, String::as_str)
}

fn number<T: FromStr>(options: &Options, name: &str, default: &str) -> Result<T, String> {
    let raw = value(options, name, default);
    raw.parse()
        .map_err(|_| format!("invalid value {raw:?} for {name}"))
}

fn positive_integer(options: &Options, name: &str, default: &str) -> Result<usize, String> {
    let result = number(options, name, default)?;
    if result == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(result)
}

fn positive_float(options: &Options, name: &str, default: &str) -> Result<f64, String> {
    let result: f64 = number(options, name, default)?;
    if !result.is_finite() || result <= 0.0 {
        return Err(format!("{name} must be finite and greater than zero"));
    }
    Ok(result)
}

fn read_text(path: &Path) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|error| format!("cannot read UTF-8 text {}: {error}", path.display()))
}

fn report_unknown(tokenizer: &Tokenizer, text: &str, context: &str) {
    let count = text
        .chars()
        .filter(|character| !tokenizer.vocab().contains(character))
        .count();
    if count > 0 {
        eprintln!(
            "{context}: {count} characters are outside the training vocabulary and map to U+FFFD."
        );
    }
}

/// Report mean loss per predicted token in nats per character. For character
/// tokens the scale is exactly 1 and the two coincide.
fn print_loss(label: &str, loss: f64, characters_per_target: f64) {
    let per_character = loss / characters_per_target;
    if characters_per_target == 1.0 {
        println!(
            "{label}: cross-entropy {per_character:.6} nats/character; perplexity {:.6}",
            per_character.exp()
        );
    } else {
        println!(
            "{label}: cross-entropy {per_character:.6} nats/character ({loss:.6} nats/token, {characters_per_target:.4} characters/token); per-character perplexity {:.6}",
            per_character.exp()
        );
    }
}

/// Characters covered by scored targets, per scored target. Each document's
/// first token is only context, so its characters are excluded, matching
/// `training::evaluate`.
fn characters_per_target(tokenizer: &Tokenizer, documents: &[Vec<usize>]) -> f64 {
    let (characters, targets) =
        documents
            .iter()
            .fold((0_usize, 0_usize), |(characters, targets), tokens| {
                let scored: usize = tokens[1..]
                    .iter()
                    .map(|&token| tokenizer.token_characters(token))
                    .sum();
                (characters + scored, targets + tokens.len() - 1)
            });
    characters as f64 / targets as f64
}

fn character_count(documents: &[String]) -> usize {
    documents.iter().map(|text| text.chars().count()).sum()
}

fn prepare_fineweb(arguments: &[String]) -> Result<(), String> {
    let options = parse_options(
        arguments,
        &[
            "--output",
            "--subset",
            "--offset",
            "--rows",
            "--max-chars",
            "--seed",
        ],
    )?;
    let config = FineWebConfig {
        subset: value(&options, "--subset", "sample-10BT").to_owned(),
        offset: number(&options, "--offset", "0")?,
        rows: positive_integer(&options, "--rows", "128")?,
        max_chars: positive_integer(&options, "--max-chars", "200000")?,
        seed: number(&options, "--seed", "42")?,
        output_dir: PathBuf::from(value(&options, "--output", "data/fineweb")),
    };
    println!(
        "Preparing HuggingFaceFW/fineweb, subset={}, offset={}, row budget={}, character budget={}, seed={}.",
        config.subset, config.offset, config.rows, config.max_chars, config.seed
    );
    let report = fineweb::prepare(&config)?;
    println!("Source revision: {}", report.revision);
    println!(
        "Prepared {} training documents ({} characters), {} validation documents ({} characters); inspected {} rows, skipped {} duplicates and {} rejected rows.",
        report.train_documents,
        report.train_characters,
        report.validation_documents,
        report.validation_characters,
        report.fetched_rows,
        report.duplicates_skipped,
        report.rejected_rows
    );
    println!(
        "Corpus and provenance manifest: {}",
        config.output_dir.display()
    );
    Ok(())
}

fn report_corpus_unknown(tokenizer: &Tokenizer, documents: &[String], context: &str) {
    let total = documents
        .iter()
        .map(|text| text.chars().count())
        .sum::<usize>();
    let unknown = documents
        .iter()
        .flat_map(|text| tokenizer.encode(text))
        .filter(|&token| token == 0)
        .count();
    println!("{context}: {unknown}/{total} unknown characters map to U+FFFD.");
}

fn train(arguments: &[String]) -> Result<(), String> {
    let options = parse_options(
        arguments,
        &[
            "--data",
            "--validation-data",
            "--checkpoint",
            "--steps",
            "--batch-size",
            "--device",
            "--hidden",
            "--feedback",
            "--sequence-length",
            "--learning-rate",
            "--lr-schedule",
            "--merges",
            "--gradient-clip",
            "--voltage-penalty",
            "--voltage-band-widths",
            "--seed",
        ],
    )?;
    let steps = positive_integer(&options, "--steps", "500")?;
    let batch_size = positive_integer(&options, "--batch-size", "16")?;
    let hidden_size = positive_integer(&options, "--hidden", "48")?;
    let sequence_length = positive_integer(&options, "--sequence-length", "32")?;
    let learning_rate = positive_float(&options, "--learning-rate", "0.003")?;
    let learning_rate_schedule =
        LearningRateSchedule::parse(value(&options, "--lr-schedule", "constant"))?;
    let merges: usize = number(&options, "--merges", "0")?;
    let gradient_clip = positive_float(&options, "--gradient-clip", "1.0")?;
    let voltage_penalty = VoltagePenalty {
        coefficient: number(&options, "--voltage-penalty", "0")?,
        band_widths: number(&options, "--voltage-band-widths", "3")?,
    };
    voltage_penalty.validate()?;
    let seed = number(&options, "--seed", "42")?;
    let feedback = FeedbackMode::parse(value(&options, "--feedback", "both"))?;
    let device = runtime::device(value(&options, "--device", "gpu"))?;
    println!(
        "Burn 0.22 device: {device:?}; identity: {:?}",
        device.identity()
    );
    let data_path = Path::new(value(&options, "--data", "data/fineweb/train.jsonl"));
    let checkpoint_path = PathBuf::from(value(&options, "--checkpoint", "artifacts/liminal.bin"));
    if !options.contains_key("--data") && !data_path.exists() {
        return Err("FineWeb is not prepared; run liminal prepare-fineweb first".into());
    }
    let validation_path = options
        .get("--validation-data")
        .map(PathBuf::from)
        .or_else(|| {
            (!options.contains_key("--data"))
                .then(|| PathBuf::from("data/fineweb/validation.jsonl"))
        });
    let (training_documents, validation_documents) = if let Some(path) = validation_path {
        (read_documents(data_path)?, read_documents(&path)?)
    } else {
        if data_path
            .extension()
            .is_some_and(|extension| extension == "jsonl")
        {
            return Err(
                "JSONL training requires --validation-data with a separate document split".into(),
            );
        }
        let text = read_text(data_path)?;
        let character_count = text.chars().count();
        let split_character = character_count - character_count.div_ceil(10);
        if split_character < 2 || character_count - split_character < 2 {
            return Err("training text needs at least two characters in each 90/10 split".into());
        }
        let split_byte = text
            .char_indices()
            .nth(split_character)
            .map(|(offset, _)| offset)
            .ok_or("cannot split the corpus")?;
        let (training, validation) = text.split_at(split_byte);
        (vec![training.to_owned()], vec![validation.to_owned()])
    };
    let training_texts: HashSet<&str> = training_documents.iter().map(String::as_str).collect();
    if validation_documents
        .iter()
        .any(|text| training_texts.contains(text.as_str()))
    {
        return Err("training and validation contain an identical document".into());
    }
    // Merges are learned from training documents only, like the characters.
    let tokenizer =
        Tokenizer::from_documents(&training_documents).with_merges(&training_documents, merges)?;
    let tokens: Vec<Vec<usize>> = training_documents
        .iter()
        .map(|text| tokenizer.encode(text))
        .collect();
    let validation_tokens: Vec<Vec<usize>> = validation_documents
        .iter()
        .map(|text| tokenizer.encode(text))
        .collect();
    let train_sample = training::training_evaluation_sample(&tokens);
    let train_scale = characters_per_target(&tokenizer, &train_sample);
    let validation_scale = characters_per_target(&tokenizer, &validation_tokens);
    let unigram_validation_loss =
        training::unigram_loss(&tokens, &validation_tokens, tokenizer.vocab_size())?;
    report_corpus_unknown(&tokenizer, &validation_documents, "Validation split");
    println!(
        "Corpus: {} training documents ({} characters, {} tokens), {} validation documents ({} characters, {} tokens), {} vocabulary entries ({} characters + {} merges).",
        tokens.len(),
        character_count(&training_documents),
        tokens.iter().map(Vec::len).sum::<usize>(),
        validation_tokens.len(),
        character_count(&validation_documents),
        validation_tokens.iter().map(Vec::len).sum::<usize>(),
        tokenizer.vocab_size(),
        tokenizer.vocab().len(),
        tokenizer.merges().len()
    );
    if train_sample.len() < tokens.len() {
        println!(
            "Training-split scores use an evenly spaced sample of {} of {} documents; validation is scored in full.",
            train_sample.len(),
            tokens.len()
        );
    }
    println!(
        "Training: hidden={hidden_size}, sequence-length={sequence_length}, batch-size={batch_size}, steps={steps}, learning-rate={learning_rate}, lr-schedule={}, gradient-clip={gradient_clip}, voltage-penalty={}, voltage-band-widths={}, seed={seed}, precision=f32.",
        learning_rate_schedule.as_str(),
        voltage_penalty.coefficient,
        voltage_penalty.band_widths
    );
    let mut model = Model::new(
        ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_size,
            threshold: 1.0,
            surrogate_width: 1.0,
        },
        seed,
        &device,
    )?;
    model.feedback = feedback;
    println!(
        "Model: {} parameters; feedback={}, decays=learned per neuron, threshold=1.0, surrogate-width=1.0.",
        model.parameter_count(),
        model.feedback.as_str()
    );
    let report = training::train(
        &mut model,
        &tokens,
        &validation_tokens,
        &TrainConfig {
            steps,
            batch_size,
            sequence_length,
            learning_rate,
            learning_rate_schedule,
            gradient_clip,
            seed,
            voltage_penalty,
        },
    )?;
    print_loss("Initial training", report.initial_train_loss, train_scale);
    print_loss("Final training", report.final_train_loss, train_scale);
    print_loss(
        "Train-frequency validation unigram (additive-one smoothing)",
        unigram_validation_loss,
        validation_scale,
    );
    print_loss(
        "Initial validation",
        report.initial_validation_loss,
        validation_scale,
    );
    print_loss(
        "Final validation",
        report.final_validation_loss,
        validation_scale,
    );
    checkpoint::save(&checkpoint_path, &model, &tokenizer)?;
    println!(
        "Saved model after {} updates and {} non-padding training targets to {}.",
        report.steps,
        report.trained_targets,
        checkpoint_path.display()
    );
    Ok(())
}

fn generate(arguments: &[String]) -> Result<(), String> {
    let options = parse_options(
        arguments,
        &[
            "--checkpoint",
            "--device",
            "--prompt",
            "--tokens",
            "--temperature",
            "--top-p",
            "--repetition-penalty",
            "--seed",
        ],
    )?;
    let count = number(&options, "--tokens", "160")?;
    let decoding = Decoding {
        temperature: number(&options, "--temperature", "0.8")?,
        top_p: number(&options, "--top-p", "1")?,
        repetition_penalty: number(&options, "--repetition-penalty", "1")?,
    };
    let seed = number(&options, "--seed", "42")?;
    let prompt = value(&options, "--prompt", "The ");
    if prompt.is_empty() {
        return Err("--prompt must contain at least one character".to_string());
    }
    let checkpoint_path = Path::new(value(&options, "--checkpoint", "artifacts/liminal.bin"));
    let device = runtime::device(value(&options, "--device", "gpu"))?;
    let (model, tokenizer) = checkpoint::load(checkpoint_path, &device)?;
    report_unknown(&tokenizer, prompt, "Prompt");
    let generated = training::generate(&model, &tokenizer.encode(prompt), count, decoding, seed)?;
    println!("{prompt}{}", tokenizer.decode(&generated)?);
    Ok(())
}

fn evaluate(arguments: &[String]) -> Result<(), String> {
    let options = parse_options(
        arguments,
        &[
            "--checkpoint",
            "--data",
            "--sequence-length",
            "--device",
            "--batch-size",
        ],
    )?;
    let sequence_length = positive_integer(&options, "--sequence-length", "32")?;
    let batch_size = positive_integer(&options, "--batch-size", "16")?;
    let data_path = value(&options, "--data", "data/fineweb/validation.jsonl");
    let checkpoint_path = Path::new(value(&options, "--checkpoint", "artifacts/liminal.bin"));
    let documents = read_documents(Path::new(data_path))?;
    let device = runtime::device(value(&options, "--device", "gpu"))?;
    println!(
        "Burn 0.22 device: {device:?}; identity: {:?}",
        device.identity()
    );
    let (model, tokenizer) = checkpoint::load(checkpoint_path, &device)?;
    report_corpus_unknown(&tokenizer, &documents, "Evaluation corpus");
    let tokens: Vec<Vec<usize>> = documents
        .iter()
        .map(|text| tokenizer.encode(text))
        .collect();
    let loss = training::evaluate(&model, &tokens, sequence_length, batch_size)?;
    println!(
        "Evaluation: {} documents, {} characters, {} tokens, sequence-length={sequence_length}, batch-size={batch_size}, precision=f32.",
        tokens.len(),
        character_count(&documents),
        tokens.iter().map(Vec::len).sum::<usize>()
    );
    print_loss(
        "Evaluation",
        loss,
        characters_per_target(&tokenizer, &tokens),
    );
    Ok(())
}
