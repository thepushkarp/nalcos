use super::*;
use std::collections::BTreeMap;

#[test]
fn resolver_requires_a_selected_model() {
    let result = resolve(
        &ModelRequest::default(),
        ResolveOptions::default(),
        &Execution::unlimited(),
    );
    assert_eq!(result.unwrap_err().code, "model_required");
}

#[test]
fn explicit_device_errors_preserve_detail_and_control_flow_errors() {
    let error = explicit_device_error(
        AppError::new(
            "device_unavailable",
            "CUDA execution provider is unavailable",
        )
        .action("Install a CUDA-enabled runtime"),
        Device::Cuda,
    );
    assert_eq!(error.code, "explicit_device_unavailable");
    assert!(error.message.contains("device_unavailable"));
    assert!(
        error
            .message
            .contains("CUDA execution provider is unavailable")
    );
    assert_eq!(
        error.action.as_deref(),
        Some("Install a CUDA-enabled runtime")
    );
    let repeated = explicit_device_error(error.clone(), Device::Cuda);
    assert_eq!(repeated.message, error.message);
    for code in [
        "timeout",
        "interrupted",
        "invalid_input",
        "invalid_config",
        "token_limit",
        "tokenization_failed",
    ] {
        let error = explicit_device_error(AppError::new(code, "Original detail"), Device::Metal);
        assert_eq!(error.code, code);
        assert_eq!(error.message, "Original detail");
    }
    for (code, detail) in [
        ("inference_failed", "Failed to allocate GGUF context"),
        (
            "inference_failed",
            "Encoder returned an unexpected number of vectors",
        ),
        ("dimension_mismatch", "Expected 384 dimensions, received 1"),
        ("invalid_embedding", "Embedding contains non-finite values"),
    ] {
        let error = AppError::new(code, detail);
        let automatic = explicit_device_error(error.clone(), Device::Auto);
        assert_eq!(automatic.code, code);
        assert_eq!(automatic.message, detail);
        let explicit = explicit_device_error(error, Device::Metal);
        assert_eq!(explicit.code, "explicit_device_unavailable");
        assert!(explicit.message.contains(code));
        assert!(explicit.message.contains(detail));
    }
}

#[test]
fn auto_device_requires_every_query_and_document_sample_to_match() {
    let reference = vec![vec![1.0, 0.0]; 3];
    verify_device_parity(&reference, &reference, Device::Metal, InputKind::Query).unwrap();
    let mut documents = reference.clone();
    let cosine = 0.99974f32;
    documents[2] = vec![cosine, (1.0 - cosine * cosine).sqrt()];
    let error = verify_device_parity(&reference, &documents, Device::Metal, InputKind::Document)
        .unwrap_err();
    assert_eq!(error.code, "device_unavailable");
    assert!(error.message.contains("Document sample 3"));
    let cosine = 0.99995f32;
    documents[2] = vec![cosine, (1.0 - cosine * cosine).sqrt()];
    verify_device_parity(&reference, &documents, Device::Metal, InputKind::Document).unwrap();
}

#[test]
fn document_identity_excludes_query_and_execution_settings() {
    let mut profile = profile("minilm").unwrap();
    let original = hub::fingerprint(&profile, &BTreeMap::new()).unwrap();
    profile.query_prefix = "Find changes: ".into();
    profile.device = Device::Cuda;
    assert_eq!(
        original,
        hub::fingerprint(&profile, &BTreeMap::new()).unwrap()
    );
    profile.document_prefix = "Document: ".into();
    assert_ne!(
        original,
        hub::fingerprint(&profile, &BTreeMap::new()).unwrap()
    );
    profile.document_prefix.clear();
    profile.artifact = "onnx/model_quantized.onnx".into();
    assert_ne!(
        original,
        hub::fingerprint(&profile, &BTreeMap::new()).unwrap()
    );
}

#[test]
fn persisted_gguf_contract_must_be_rebuilt_before_loading() {
    use sha2::{Digest, Sha256};
    let profile = profile("embeddinggemma").unwrap();
    let old = serde_json::to_vec(&(
        "nalcos-embedding-contract-v1-l2",
        profile.semantic_profile(),
        BTreeMap::<String, (String, u64)>::new(),
    ))
    .unwrap();
    let saved = ResolvedModel {
        profile,
        fingerprint: format!("{:x}", Sha256::digest(old)),
        artifact_path: None,
        tokenizer_path: None,
        files: BTreeMap::new(),
    };
    let error = Encoder::load(&saved, RuntimeOptions::default(), &Execution::unlimited())
        .err()
        .unwrap();
    assert_eq!(error.code, "encoder_contract_changed");
    assert!(error.action.unwrap().contains("sync"));
}

#[test]
fn chunk_ranges_preserve_all_utf8_and_whitespace() {
    let text = "  修复 cache invalidation\n\nfn main() { 🦀(); }\n   ".repeat(3);
    let count = |value: &str| Ok(value.chars().count() + 3);
    let chunks = tokenize::chunk_text(
        &text,
        ChunkOptions {
            max_tokens: 24,
            overlap_tokens: 3,
        },
        64,
        count,
    )
    .unwrap();
    let mut covered = 0;
    for chunk in &chunks {
        assert!(chunk.byte_start <= covered);
        assert!(chunk.byte_end > covered);
        assert_eq!(chunk.text, text[chunk.byte_start..chunk.byte_end]);
        assert!(chunk.token_count <= 24);
        covered = chunk.byte_end;
    }
    assert_eq!(covered, text.len());
    assert!(
        tokenize::chunk_text(
            "a",
            ChunkOptions {
                max_tokens: 3,
                overlap_tokens: 0
            },
            64,
            count
        )
        .is_err()
    );
}

#[test]
fn pooling_excludes_padding_and_normalization_rejects_bad_outputs() {
    let values = vec![1.0, 3.0, 3.0, 5.0, 1000.0, 1000.0];
    let mean = onnx::pool(&values, &[1, 3, 2], 1, 3, 2, &[1, 1, 0], Pooling::Mean).unwrap();
    assert_eq!(mean, vec![vec![2.0, 4.0]]);
    let last = onnx::pool(&values, &[1, 3, 2], 1, 3, 2, &[1, 1, 0], Pooling::Last).unwrap();
    assert_eq!(last, vec![vec![3.0, 5.0]]);
    let mut vector = vec![3.0, 4.0];
    normalize(&mut vector, 2).unwrap();
    assert_eq!(vector, vec![0.6, 0.8]);
    assert!(normalize(&mut [f32::NAN], 1).is_err());
    assert!(normalize(&mut [0.0, 0.0], 2).is_err());
    assert!(normalize(&mut [1.0], 2).is_err());
}

#[test]
fn changed_tokenizer_content_invalidates_resolved_profile() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("tokenizer.json");
    std::fs::write(&path, b"abc").unwrap();
    let mut profile = profile("minilm").unwrap();
    profile.artifact = "tokenizer.json".into();
    let mut files = BTreeMap::new();
    files.insert(
        "tokenizer.json".into(),
        hub::inspect_file(&path, &Execution::unlimited()).unwrap(),
    );
    let resolved = ResolvedModel {
        fingerprint: hub::fingerprint(&profile, &files).unwrap(),
        profile,
        artifact_path: Some(path.clone()),
        tokenizer_path: Some(path.clone()),
        files,
    };
    hub::validate_resolved(&resolved, &Execution::unlimited()).unwrap();
    std::fs::write(&path, b"abd").unwrap();
    assert_eq!(
        hub::validate_resolved(&resolved, &Execution::unlimited())
            .unwrap_err()
            .code,
        "model_corrupt"
    );
}

fn cached_smoke(model: &str, device: Device) {
    assert_eq!(
        std::env::var("NALCOS_RUN_MODEL_TESTS").as_deref(),
        Ok("1"),
        "Set NALCOS_RUN_MODEL_TESTS=1 only when the exact cached model/runtime prerequisites are available"
    );
    let execution = Execution::unlimited();
    // Dynamic INT8 activation scales depend on other padded inputs in the batch.
    // This smoke bound records that sensitivity; it is not a retrieval-quality gate.
    let cross_batch_minimum = if model == "minilm-int8" { 0.99 } else { 0.9999 };
    let model = resolve(
        &ModelRequest {
            model: Some(model.into()),
            profile: None,
        },
        ResolveOptions {
            allow_download: false,
            offline: true,
        },
        &execution,
    )
    .unwrap();
    let mut encoder = Encoder::load(
        &model,
        RuntimeOptions {
            device: Some(device),
            calibrate: false,
            ..RuntimeOptions::default()
        },
        &execution,
    )
    .unwrap();
    if let NativeEncoder::Gguf(native) = &encoder.native
        && model.profile.id == "ggml-org/embeddinggemma-300M-GGUF"
    {
        // Independently captured with llama-tokenize from the pinned server reference.
        assert_eq!(
            native.token_ids("title: none | text: <pad><eos><start_of_turn>hello<end_of_turn>"),
            vec![
                2, 3250, 236787, 7293, 1109, 1816, 236787, 236743, 0, 1, 105, 23391, 106, 1
            ]
        );
    }
    let text = "Fix cache invalidation after changing configuration";
    let texts = [text.into(), "Render a blue button".into()];
    let vectors = encoder
        .encode(&texts, InputKind::Document, &execution)
        .unwrap();
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].len(), model.profile.dimensions);
    assert_eq!(encoder.info().selected_device, device);
    let repeated = encoder
        .encode(&texts, InputKind::Document, &execution)
        .unwrap();
    for (first, second) in vectors.iter().zip(&repeated) {
        let cosine: f64 = first
            .iter()
            .zip(second)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        assert!(
            cosine > 0.9999,
            "Identical-batch inference drifted: {cosine}"
        );
    }
    for (sequence, text) in texts.iter().enumerate() {
        let again = encoder
            .encode(std::slice::from_ref(text), InputKind::Document, &execution)
            .unwrap();
        let cosine: f64 = vectors[sequence]
            .iter()
            .zip(&again[0])
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        assert!(
            cosine > cross_batch_minimum,
            "Sequence {sequence} repeated/batched output drifted: {cosine}"
        );
        eprintln!("Sequence {sequence} cross-batch cosine: {cosine}");
    }
    let chunks = encoder
        .chunk_documents(
            &text.repeat(30),
            ChunkOptions {
                max_tokens: 64,
                overlap_tokens: 8,
            },
        )
        .unwrap();
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        assert!(
            encoder
                .count_tokens(&chunk.text, InputKind::Document)
                .unwrap()
                <= 64
        );
    }
}

#[test]
#[ignore = "requires explicitly provisioned cached MiniLM and ONNX Runtime; never downloads"]
fn cached_minilm_cpu_smoke() {
    cached_smoke("minilm", Device::Cpu);
}

#[test]
#[ignore = "requires explicitly provisioned cached MiniLM INT8 and ONNX Runtime; never downloads"]
fn cached_minilm_int8_cpu_smoke() {
    cached_smoke("minilm-int8", Device::Cpu);
}

#[test]
#[ignore = "requires explicitly provisioned cached Jina and ONNX Runtime; never downloads"]
fn cached_jina_cpu_smoke() {
    cached_smoke("jina-code", Device::Cpu);
}

#[test]
#[ignore = "requires explicitly provisioned cached EmbeddingGemma; never downloads"]
fn cached_embeddinggemma_cpu_smoke() {
    cached_smoke("embeddinggemma", Device::Cpu);
}

#[test]
#[ignore = "requires Apple Silicon Metal and explicitly provisioned cached EmbeddingGemma; never downloads"]
fn cached_embeddinggemma_metal_smoke() {
    cached_smoke("embeddinggemma", Device::Metal);
}

#[test]
#[ignore = "explicit runtime installer check; downloads the pinned CPU pack to NALCOS_RUNTIME_CACHE"]
fn managed_ort_install_smoke() {
    assert_eq!(
        std::env::var("NALCOS_RUN_RUNTIME_INSTALL_TESTS").as_deref(),
        Ok("1")
    );
    assert!(
        std::env::var_os("NALCOS_RUNTIME_CACHE").is_some(),
        "Set an isolated NALCOS_RUNTIME_CACHE for this installer check"
    );
    let library = runtime::ensure_cpu_runtime(
        ResolveOptions {
            allow_download: true,
            offline: false,
        },
        &Execution::unlimited(),
    )
    .unwrap();
    assert!(library.is_file());
    assert!(library.starts_with(std::env::var_os("NALCOS_RUNTIME_CACHE").unwrap()));
    onnx::initialize(&RuntimeOptions {
        ort_library: Some(library),
        ..RuntimeOptions::default()
    })
    .unwrap();
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ReferenceCase {
    model: String,
    device: Device,
    #[serde(default)]
    local_gguf: Option<std::path::PathBuf>,
    query_texts: Vec<String>,
    query_vectors: Vec<Vec<f32>>,
    document_texts: Vec<String>,
    document_vectors: Vec<Vec<f32>>,
}

#[test]
#[ignore = "requires an external reference fixture plus cached real models/runtimes; never downloads"]
fn cached_native_reference_parity() {
    let path = std::env::var_os("NALCOS_EMBEDDING_REFERENCE_JSON").expect(
        "Set NALCOS_EMBEDDING_REFERENCE_JSON to an independently generated reference fixture",
    );
    let cases: Vec<ReferenceCase> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let execution = Execution::unlimited();
    let mut failures = Vec::new();
    let mut captured = Vec::new();
    for mut case in cases {
        let model = if let Some(path) = &case.local_gguf {
            // A converted artifact can be validated before publishing it to a Hub repository.
            // Its content hash is still part of the exact model identity used by the runtime.
            let mut profile = profile(&case.model).unwrap();
            profile.backend = Backend::Gguf;
            profile.artifact = path.file_name().unwrap().to_str().unwrap().into();
            profile.tokenizer = None;
            profile.extra_files.clear();
            let mut files = BTreeMap::new();
            files.insert(
                profile.artifact.clone(),
                hub::inspect_file(path, &execution).unwrap(),
            );
            ResolvedModel {
                fingerprint: hub::fingerprint(&profile, &files).unwrap(),
                profile,
                artifact_path: Some(path.clone()),
                tokenizer_path: None,
                files,
            }
        } else {
            resolve(
                &ModelRequest {
                    model: Some(case.model.clone()),
                    profile: None,
                },
                ResolveOptions {
                    allow_download: false,
                    offline: true,
                },
                &execution,
            )
            .unwrap()
        };
        let mut encoder = Encoder::load(
            &model,
            RuntimeOptions {
                device: Some(case.device),
                calibrate: false,
                ..RuntimeOptions::default()
            },
            &execution,
        )
        .unwrap();
        for (kind, texts, expected) in [
            (InputKind::Query, &case.query_texts, &mut case.query_vectors),
            (
                InputKind::Document,
                &case.document_texts,
                &mut case.document_vectors,
            ),
        ] {
            let actual = encoder.encode(texts, kind, &execution).unwrap();
            assert_eq!(actual.len(), expected.len());
            let mut minimum = 1.0f64;
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert_eq!(actual.len(), expected.len());
                let cosine = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>();
                minimum = minimum.min(cosine);
                if cosine < 0.9999 {
                    failures.push(format!(
                        "{} {} {kind:?} differs from reference: {cosine}",
                        case.model, case.device
                    ));
                }
            }
            eprintln!(
                "Reference parity: {} {} {kind:?}, {} vectors, minimum cosine {minimum:.8}",
                case.model,
                case.device,
                actual.len()
            );
            *expected = actual;
        }
        captured.push(case);
    }
    if let Some(path) = std::env::var_os("NALCOS_EMBEDDING_ACTUAL_JSON") {
        std::fs::write(path, serde_json::to_vec(&captured).unwrap()).unwrap();
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
