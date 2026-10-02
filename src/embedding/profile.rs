use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Onnx,
    Gguf,
    #[serde(alias = "openai")]
    OpenAi,
    Ollama,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    #[default]
    Mean,
    Cls,
    Last,
    /// The graph or remote provider already returns one vector per input.
    Model,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
    #[serde(alias = "coreml")]
    CoreMl,
    Vulkan,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::Vulkan => "vulkan",
        })
    }
}

impl std::str::FromStr for Device {
    type Err = AppError;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            "coreml" | "core_ml" => Ok(Self::CoreMl),
            "vulkan" => Ok(Self::Vulkan),
            _ => Err(AppError::invalid(format!(
                "Unknown device '{value}'; use auto, cpu, metal, cuda, coreml, or vulkan"
            ))),
        }
    }
}

/// A model's embedding contract. Changing any semantic field creates a new index generation.
/// `revision` must be immutable after resolution, including for remote providers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub id: String,
    #[serde(default = "main_revision")]
    pub revision: String,
    #[serde(default)]
    pub artifact: String,
    pub backend: Backend,
    pub dimensions: usize,
    pub max_tokens: usize,
    #[serde(default)]
    pub pooling: Pooling,
    #[serde(default)]
    pub query_prefix: String,
    #[serde(default)]
    pub document_prefix: String,
    #[serde(default)]
    pub tokenizer: Option<String>,
    #[serde(default)]
    pub extra_files: Vec<String>,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub device: Device,
}

fn main_revision() -> String {
    "main".into()
}

impl ModelProfile {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() || self.revision.trim().is_empty() {
            return Err(AppError::invalid("Model id and revision must not be empty"));
        }
        if self.dimensions == 0 || self.dimensions > 65_536 {
            return Err(AppError::invalid(
                "Model dimensions must be between 1 and 65536",
            ));
        }
        if !(8..=131_072).contains(&self.max_tokens) {
            return Err(AppError::invalid(
                "Model max_tokens must be between 8 and 131072",
            ));
        }
        match self.backend {
            Backend::Onnx | Backend::Gguf => {
                validate_relative_path(&self.artifact)?;
                if !valid_repo_id(&self.id) {
                    return Err(AppError::invalid(
                        "A local model id must be a Hugging Face repository id",
                    ));
                }
                if self.backend == Backend::Onnx && self.tokenizer.is_none() {
                    return Err(AppError::invalid(
                        "ONNX profiles require a tokenizer.json file",
                    ));
                }
                if self.endpoint.is_some() || self.api_key_env.is_some() {
                    return Err(AppError::invalid(
                        "Local profiles cannot specify provider credentials or endpoints",
                    ));
                }
            }
            Backend::OpenAi | Backend::Ollama => {
                let endpoint = self.endpoint.as_deref().ok_or_else(|| {
                    AppError::invalid("Remote model profiles require an explicit endpoint")
                })?;
                let url = reqwest::Url::parse(endpoint)
                    .map_err(|_| AppError::invalid("Model endpoint must be a valid HTTP(S) URL"))?;
                if !matches!(url.scheme(), "https" | "http")
                    || !url.username().is_empty()
                    || url.password().is_some()
                    || url.query().is_some()
                    || url.fragment().is_some()
                {
                    return Err(AppError::invalid(
                        "Model endpoint must use HTTP(S) without embedded credentials, a query, or a fragment",
                    ));
                }
                if self.revision == "main" || self.revision == "latest" {
                    return Err(AppError::invalid(
                        "Remote profiles require an explicit model version or deployment revision; mutable 'main'/'latest' cannot identify an embedding index",
                    ));
                }
                // Arbitrary provider tokenizers are not approximated by characters or whitespace.
                if self.tokenizer.is_none() {
                    return Err(AppError::invalid(
                        "Remote profiles require tokenizer = '/absolute/path/to/tokenizer.json' for bounded, tokenizer-aware chunks",
                    ));
                }
            }
        }
        if matches!(self.backend, Backend::Onnx | Backend::Gguf) {
            if let Some(tokenizer) = &self.tokenizer {
                validate_relative_path(tokenizer)?;
            }
            for file in &self.extra_files {
                validate_relative_path(file)?;
            }
        }
        Ok(())
    }

    pub(crate) fn semantic_profile(&self) -> Self {
        let mut value = self.clone();
        // Device and secret lookup location are operational, not vector-space identities.
        value.device = Device::Auto;
        value.api_key_env = None;
        value.query_prefix.clear();
        if matches!(value.backend, Backend::OpenAi | Backend::Ollama) {
            value.tokenizer = Some("external-tokenizer-content".into());
        }
        value
    }
}

fn valid_repo_id(id: &str) -> bool {
    let parts: Vec<_> = id.split('/').collect();
    !parts.is_empty()
        && parts.len() <= 2
        && parts.iter().all(|part| {
            !part.is_empty()
                && !matches!(*part, "." | "..")
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

pub(crate) fn validate_relative_path(file: &str) -> Result<()> {
    let path = std::path::Path::new(file);
    if file.is_empty()
        || file.contains('\\')
        || path
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(AppError::invalid(format!(
            "Model artifact must be a repository-relative file: {file}"
        )));
    }
    Ok(())
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    pub model: Option<String>,
    pub profile: Option<ModelProfile>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ResolveOptions {
    pub allow_download: bool,
    pub offline: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedFile {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    #[serde(default)]
    pub modified_ns: Option<u64>,
    #[serde(default)]
    pub file_id: Option<(u64, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub profile: ModelProfile,
    pub fingerprint: String,
    pub artifact_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub files: BTreeMap<String, ResolvedFile>,
}

/// Built-in profiles with pinned artifacts and revisions.
fn known_profiles() -> Vec<ModelProfile> {
    vec![
        local_profile(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "b207367332321f8e44f96e224ef15bc607f4dbf0",
            "onnx/model.onnx",
            Backend::Onnx,
            384,
            512,
            Pooling::Mean,
        ),
        local_profile(
            "jinaai/jina-embeddings-v2-base-code",
            "516f4baf13dec4ddddda8631e019b5737c8bc250",
            "onnx/model.onnx",
            Backend::Onnx,
            768,
            8192,
            Pooling::Mean,
        ),
        local_profile(
            "BAAI/bge-small-en-v1.5",
            "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            "onnx/model.onnx",
            Backend::Onnx,
            384,
            512,
            Pooling::Cls,
        ),
        local_profile(
            "ggml-org/embeddinggemma-300M-GGUF",
            "0f741b5a6585bd53aeb15cd1372c56f2a0f65e12",
            "embeddinggemma-300M-Q8_0.gguf",
            Backend::Gguf,
            768,
            2048,
            Pooling::Mean,
        ),
        local_profile(
            "Qwen/Qwen3-Embedding-0.6B-GGUF",
            "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
            "Qwen3-Embedding-0.6B-Q8_0.gguf",
            Backend::Gguf,
            1024,
            8192,
            Pooling::Last,
        ),
    ]
}

fn local_profile(
    id: &str,
    revision: &str,
    artifact: &str,
    backend: Backend,
    dimensions: usize,
    max_tokens: usize,
    pooling: Pooling,
) -> ModelProfile {
    let (query_prefix, document_prefix) = match id {
        "BAAI/bge-small-en-v1.5" => (
            "Represent this sentence for searching relevant passages: ",
            "",
        ),
        "ggml-org/embeddinggemma-300M-GGUF" => {
            ("task: search result | query: ", "title: none | text: ")
        }
        "Qwen/Qwen3-Embedding-0.6B-GGUF" => (
            "Instruct: Given a natural language query, retrieve relevant Git history changes\nQuery: ",
            "",
        ),
        _ => ("", ""),
    };
    ModelProfile {
        id: id.into(),
        revision: revision.into(),
        artifact: artifact.into(),
        backend,
        dimensions,
        max_tokens,
        pooling,
        query_prefix: query_prefix.into(),
        document_prefix: document_prefix.into(),
        tokenizer: (backend == Backend::Onnx).then(|| "tokenizer.json".into()),
        extra_files: vec![],
        output: None,
        endpoint: None,
        api_key_env: None,
        device: Device::Auto,
    }
}

pub fn profile(id: &str) -> Result<ModelProfile> {
    if id == "minilm-int8" {
        let mut selected = profile("minilm")?;
        // Architecture-specific quantizations are distinct embedding contracts.
        // The selected artifact is persisted so moving an index never switches weights.
        selected.artifact = minilm_int8_artifact()?.into();
        selected.device = Device::Cpu;
        return Ok(selected);
    }
    let canonical = match id {
        "minilm" => "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "jina-code" | "jina" => "jinaai/jina-embeddings-v2-base-code",
        "bge-small" => "BAAI/bge-small-en-v1.5",
        "embeddinggemma" | "gemma" => "ggml-org/embeddinggemma-300M-GGUF",
        "qwen3" => "Qwen/Qwen3-Embedding-0.6B-GGUF",
        other => other,
    };
    known_profiles().into_iter().find(|p| p.id == canonical).ok_or_else(|| {
        AppError::new("unknown_model", format!("No embedding contract is registered for '{id}'; provide a custom profile with backend, artifact, dimensions, tokenizer, pooling and prefixes"))
    })
}

fn minilm_int8_artifact() -> Result<&'static str> {
    #[cfg(target_arch = "aarch64")]
    {
        Ok("onnx/model_qint8_arm64.onnx")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Ok("onnx/model_quint8_avx2.onnx");
        }
        Err(AppError::new(
            "unsupported_cpu",
            "The MiniLM INT8 preset requires ARM64 or an x86_64 CPU with AVX2",
        )
        .action("Use nalcos init --model minilm for the portable FP32 artifact, or configure a compatible profile"))
    }
}
