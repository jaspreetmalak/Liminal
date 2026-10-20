use std::process::Command;

use liminal::data::{Tokenizer, UNKNOWN_CHARACTER, read_documents};

#[test]
fn unicode_tokenization_preserves_scalars_and_training_only_vocabulary() {
    let tokenizer = Tokenizer::from_text("café λ café");
    assert_eq!(tokenizer.vocab()[0], UNKNOWN_CHARACTER);
    assert!(
        tokenizer.vocab()[1..]
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    );
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("café λ")).unwrap(),
        "café λ"
    );
    assert_eq!(tokenizer.encode("🦀"), vec![0]);
    assert_eq!(tokenizer.decode(&[0]).unwrap(), "\u{fffd}");
    assert!(tokenizer.decode(&[tokenizer.vocab_size()]).is_err());
    assert_eq!(Tokenizer::from_text("λ éfac").vocab(), tokenizer.vocab());
}

#[test]
fn restored_vocabulary_rejects_ambiguous_indices() {
    assert!(Tokenizer::from_vocab(vec![]).is_err());
    assert!(Tokenizer::from_vocab(vec!['a']).is_err());
    assert!(Tokenizer::from_vocab(vec![UNKNOWN_CHARACTER, 'a', 'a']).is_err());
    assert!(Tokenizer::from_vocab(vec![UNKNOWN_CHARACTER, 'b', 'a']).is_err());
    assert!(Tokenizer::from_vocab(vec![UNKNOWN_CHARACTER, UNKNOWN_CHARACTER]).is_err());
    let original = Tokenizer::from_text("spikes and words");
    assert_eq!(
        Tokenizer::from_vocab(original.vocab().to_vec()).unwrap(),
        original
    );
}

#[test]
fn cli_help_and_bad_options_have_correct_exit_status() {
    let binary = env!("CARGO_BIN_EXE_liminal");
    let help = Command::new(binary).arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(
        String::from_utf8(help.stdout)
            .unwrap()
            .contains("cross-entropy")
    );
    for arguments in [
        vec!["train", "--steps", "1", "--steps", "2"],
        vec!["train", "--steps", "0"],
        vec!["train", "--learning-rate", "NaN"],
        vec!["train", "--voltage-penalty", "-1"],
        vec!["train", "--voltage-penalty", "NaN"],
        vec!["train", "--voltage-penalty", "inf"],
        vec!["train", "--voltage-penalty", "1e-300"],
        vec!["train", "--voltage-band-widths", "0"],
        vec!["train", "--voltage-band-widths", "-1"],
        vec!["train", "--voltage-band-widths", "NaN"],
        vec!["train", "--voltage-band-widths", "1e300"],
        vec!["evaluate", "--bogus", "x"],
        vec!["generate", "--temperature", "-1"],
        vec!["prepare-fineweb", "--rows", "0"],
    ] {
        let output = Command::new(binary).args(&arguments).output().unwrap();
        assert!(!output.status.success(), "arguments: {arguments:?}");
        assert!(String::from_utf8(output.stderr).unwrap().contains("error:"));
    }
}

#[test]
fn jsonl_documents_stay_separate_and_validation_never_fits_vocabulary() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/test-artifacts")
        .join(format!("document-cli-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let training = directory.join("train.jsonl");
    let validation = directory.join("validation.jsonl");
    let checkpoint = directory.join("model.bin");
    std::fs::write(
        &training,
        "{\"text\":\"ab\\nβab\",\"id\":\"first\"}\n{\"text\":\"baba\",\"id\":\"second\"}\n",
    )
    .unwrap();
    std::fs::write(&validation, "{\"text\":\"aλba\",\"id\":\"heldout\"}\n").unwrap();
    let documents = read_documents(&training).unwrap();
    assert_eq!(documents, vec!["ab\nβab", "baba"]);
    let output = Command::new(env!("CARGO_BIN_EXE_liminal"))
        .args(["train", "--data"])
        .arg(&training)
        .arg("--validation-data")
        .arg(&validation)
        .arg("--checkpoint")
        .arg(&checkpoint)
        .args([
            "--steps",
            "2",
            "--hidden",
            "4",
            "--sequence-length",
            "3",
            "--batch-size",
            "2",
            "--voltage-penalty",
            "0.001",
            "--voltage-band-widths",
            "3",
            "--device",
            "cpu",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (_, tokenizer) =
        liminal::checkpoint::load(&checkpoint, &burn::tensor::Device::flex()).unwrap();
    assert_eq!(tokenizer.encode("λ"), vec![0]);
    assert_ne!(tokenizer.encode("β"), vec![0]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("2 training documents"));
    assert!(stdout.contains("1/4 unknown characters"));
    assert!(stdout.contains("voltage-penalty=0.001, voltage-band-widths=3"));
    let progress = String::from_utf8(output.stderr).unwrap();
    for term in [
        "batch CE",
        "raw voltage penalty",
        "weighted voltage penalty",
        "total",
        "gradient norm",
    ] {
        assert!(
            progress.contains(term),
            "missing objective progress field {term:?}"
        );
    }
    let evaluation = Command::new(env!("CARGO_BIN_EXE_liminal"))
        .arg("evaluate")
        .arg("--data")
        .arg(&validation)
        .arg("--checkpoint")
        .arg(&checkpoint)
        .args(["--device", "cpu", "--batch-size", "2"])
        .output()
        .unwrap();
    assert!(evaluation.status.success());
    assert!(
        String::from_utf8(evaluation.stdout)
            .unwrap()
            .contains("1 documents, 4 characters")
    );

    let invalid = directory.join("invalid.jsonl");
    std::fs::write(&invalid, "{\"missing\":\"text\"}\n").unwrap();
    assert!(read_documents(&invalid).is_err());
}
