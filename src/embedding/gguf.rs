use super::{Device, DeviceInfo, ModelProfile, Pooling, ResolvedModel, RuntimeOptions};
use crate::error::{AppError, Result};
use crate::execution::Execution;
use llama_cpp_2::context::{
    LlamaContext,
    params::{LlamaContextParams, LlamaPoolingType},
};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{
    LlamaModel,
    params::{LlamaModelParams, LlamaSplitMode},
};
use std::num::NonZeroU32;
use std::sync::OnceLock;

static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend> {
    BACKEND
        .get_or_init(|| {
            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
            LlamaBackend::init().map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| AppError::new("runtime_failed", e.clone()))
}

pub(crate) fn devices() -> Result<Vec<DeviceInfo>> {
    backend()?;
    Ok(llama_cpp_2::list_llama_ggml_backend_devices()
        .into_iter()
        .filter_map(|value| {
            let device = match value.backend.to_ascii_lowercase().as_str() {
                "cpu" => Device::Cpu,
                "metal" | "mtl" => Device::Metal,
                "cuda" => Device::Cuda,
                "vulkan" => Device::Vulkan,
                _ => return None,
            };
            Some(DeviceInfo {
                name: value.description,
                device,
                index: value.index,
                memory_bytes: Some(value.memory_total),
                compiled: true,
                qualified: false,
            })
        })
        .collect())
}

self_cell::self_cell! {
    struct ModelContext {
        owner: LlamaModel,
        #[covariant]
        dependent: LlamaContext,
    }
}

pub(crate) struct GgufEncoder {
    state: ModelContext,
    profile: ModelProfile,
    encoder_graph: bool,
    pub device_name: String,
    threads: usize,
    device: Device,
    pooling: LlamaPoolingType,
    sequence_capacity: usize,
    tokens_per_sequence: usize,
    pub batch_size: usize,
}

impl GgufEncoder {
    pub fn load(
        model: &ResolvedModel,
        options: &RuntimeOptions,
        device: Device,
        execution: &Execution,
    ) -> Result<Self> {
        execution.check()?;
        let backend = backend()?;
        if !matches!(
            device,
            Device::Cpu | Device::Metal | Device::Cuda | Device::Vulkan
        ) {
            return Err(AppError::new(
                "device_unavailable",
                format!("GGUF cannot use {device}; choose cpu, metal, cuda, or vulkan"),
            ));
        }
        let available = if device == Device::Cpu {
            Vec::new()
        } else {
            devices()?
        };
        let selected = if device == Device::Cpu {
            None
        } else {
            Some(available.iter().filter(|d| d.device == device).nth(options.device_index)
                .ok_or_else(|| AppError::new("device_unavailable", format!("No usable {device} device {} was registered by the compiled llama.cpp backends; install a matching runtime build or use cpu", options.device_index)))?)
        };
        let device_ids: Vec<_> = selected.iter().map(|d| d.index).collect();
        let cancellation = execution.clone();
        let params = LlamaModelParams::default()
            .with_devices(&device_ids)
            .map_err(|e| AppError::new("device_unavailable", e.to_string()))?
            .with_n_gpu_layers(if device == Device::Cpu { 0 } else { u32::MAX })
            .with_split_mode(LlamaSplitMode::None)
            .with_progress_callback(move |_| cancellation.check().is_ok());
        let path = model
            .artifact_path
            .as_ref()
            .ok_or_else(|| AppError::invalid("GGUF artifact path is missing"))?;
        let loaded = LlamaModel::load_from_file(backend, path, &params);
        execution.check()?;
        let native = loaded.map_err(|e| {
            AppError::new(
                if device == Device::Cpu {
                    "model_invalid"
                } else {
                    "device_unavailable"
                },
                format!("Cannot load GGUF on {device}: {e}"),
            )
        })?;
        if usize::try_from(native.n_embd_out()).ok() != Some(model.profile.dimensions) {
            return Err(AppError::new(
                "dimension_mismatch",
                format!(
                    "GGUF output dimension {} does not match profile dimension {}",
                    native.n_embd_out(),
                    model.profile.dimensions
                ),
            ));
        }
        if model.profile.max_tokens > native.n_ctx_train() as usize {
            return Err(AppError::new(
                "token_limit",
                format!(
                    "Configured context {} exceeds GGUF training context {}",
                    model.profile.max_tokens,
                    native.n_ctx_train()
                ),
            ));
        }
        let architecture = native
            .meta_val_str("general.architecture")
            .map_err(|e| AppError::new("model_invalid", e.to_string()))?;
        let encoder_graph = matches!(architecture.as_str(), "t5" | "t5encoder");
        let pooling = match model.profile.pooling {
            Pooling::Mean => LlamaPoolingType::Mean,
            Pooling::Cls => LlamaPoolingType::Cls,
            Pooling::Last => LlamaPoolingType::Last,
            Pooling::Model => LlamaPoolingType::Unspecified,
        };
        // Each complete sequence fits one physical batch. Splitting a bidirectional encoder
        // across micro-batches changes its attention and therefore its vector space.
        let context = model.profile.max_tokens.min(512) as u32;
        let context_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(context))
            .with_n_batch(context)
            .with_n_ubatch(context)
            .with_n_threads(options.threads as i32)
            .with_n_threads_batch(options.threads as i32)
            .with_embeddings(true)
            .with_pooling_type(pooling)
            .with_offload_kqv(device != Device::Cpu)
            .with_op_offload(device != Device::Cpu);
        let state =
            ModelContext::try_new(native, |owner| owner.new_context(backend, context_params))
                .map_err(|e| {
                    AppError::new(
                        if device == Device::Cpu {
                            "inference_failed"
                        } else {
                            "device_unavailable"
                        },
                        format!("Cannot allocate GGUF context on {device}: {e}"),
                    )
                })?;
        Ok(Self {
            state,
            profile: model.profile.clone(),
            encoder_graph,
            device_name: selected.map_or_else(|| "CPU".into(), |d| d.name.clone()),
            threads: options.threads,
            device,
            pooling,
            sequence_capacity: 1,
            tokens_per_sequence: context as usize,
            batch_size: options.batch_size.min(32),
        })
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        if text.len() > i32::MAX as usize {
            return Err(AppError::new(
                "token_limit",
                "Text exceeds native tokenizer size limits",
            ));
        }
        Ok(self.tokenize(text).len())
    }

    fn tokenize(&self, text: &str) -> Vec<llama_cpp_2::token::LlamaToken> {
        // Registered literal special tokens follow the tokenizer contract used by llama.cpp's
        // embeddings endpoint, in addition to the model's automatically inserted boundaries.
        self.state
            .borrow_owner()
            .vocab()
            .tokenize(text.as_bytes(), true, true)
    }

    #[cfg(test)]
    pub(super) fn token_ids(&self, text: &str) -> Vec<i32> {
        self.tokenize(text).iter().map(|token| token.0).collect()
    }

    pub fn encode(&mut self, texts: &[String], execution: &Execution) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(texts.len());
        let mut tokenized = Vec::with_capacity(texts.len());
        for text in texts {
            execution.check()?;
            let tokens = self.tokenize(text);
            if tokens.is_empty() || tokens.len() > self.profile.max_tokens {
                return Err(AppError::new(
                    "token_limit",
                    format!(
                        "Input has {} tokens; {} accepts at most {}",
                        tokens.len(),
                        self.profile.id,
                        self.profile.max_tokens
                    ),
                ));
            }
            tokenized.push(tokens);
        }
        let token_budget = self.profile.max_tokens.max(4096);
        let mut offset = 0;
        while offset < tokenized.len() {
            execution.check()?;
            let mut end = offset;
            let mut longest = 0;
            while end < tokenized.len() && end - offset < self.batch_size {
                let next_longest = longest.max(tokenized[end].len());
                if next_longest * (end - offset + 1) > token_budget && end > offset {
                    break;
                }
                longest = next_longest;
                end += 1;
            }
            let sequences = end - offset;
            let total_tokens: usize = tokenized[offset..end].iter().map(Vec::len).sum();
            if sequences > self.sequence_capacity || longest > self.tokens_per_sequence {
                let per_sequence = longest.div_ceil(256) * 256;
                let total = per_sequence * sequences;
                let params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(total as u32))
                    .with_n_seq_max(sequences as u32)
                    .with_n_batch(total as u32)
                    .with_n_ubatch(total as u32)
                    .with_n_threads(self.threads as i32)
                    .with_n_threads_batch(self.threads as i32)
                    .with_embeddings(true)
                    .with_pooling_type(self.pooling)
                    .with_offload_kqv(self.device != Device::Cpu)
                    .with_op_offload(self.device != Device::Cpu);
                self.state.with_dependent_mut(|owner, context| -> Result<()> {
                    let replacement = owner.new_context(backend()?, params)
                        .map_err(|e| AppError::new("inference_failed", format!("Failed to allocate GGUF context for {sequences} sequences and {total} tokens: {e}")))?;
                    *context = replacement;
                    Ok(())
                })?;
                self.sequence_capacity = sequences;
                self.tokens_per_sequence = per_sequence;
            }
            let mut batch = LlamaBatch::new(total_tokens, 1);
            for (sequence, tokens) in tokenized[offset..end].iter().enumerate() {
                batch
                    .add_sequence(tokens, sequence as i32, true)
                    .map_err(|e| AppError::new("inference_failed", e.to_string()))?;
            }
            let encoder_graph = self.encoder_graph;
            let vectors = self
                .state
                .with_dependent_mut(|_, context| -> Result<Vec<Vec<f32>>> {
                    context.clear_kv_cache();
                    let inferred = if encoder_graph {
                        context
                            .encode(&mut batch)
                            .map_err(|e| AppError::new("inference_failed", e.to_string()))
                    } else {
                        context
                            .decode(&mut batch)
                            .map_err(|e| AppError::new("inference_failed", e.to_string()))
                    };
                    // Native errors must not mask a deadline or interruption that
                    // arrived while this synchronous batch was running.
                    execution.check()?;
                    inferred?;
                    (0..sequences)
                        .map(|sequence| {
                            context
                                .embeddings_seq_ith(sequence as i32)
                                .map(|v| v.to_vec())
                                .map_err(|e| AppError::new("inference_failed", e.to_string()))
                        })
                        .collect()
                })?;
            execution.check()?;
            output.extend(vectors);
            offset = end;
        }
        Ok(output)
    }
}
