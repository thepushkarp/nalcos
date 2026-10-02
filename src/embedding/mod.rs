//! Local embedding contracts, shared-cache installation, and native inference.
//! Resolution may download only when explicitly permitted; loading is always local-only.

mod gguf;
mod hub;
mod onnx;
mod profile;
mod remote;
mod runtime;
mod tokenize;

pub use hub::resolve;
pub use profile::{
    Backend, Device, ModelProfile, ModelRequest, Pooling, ResolveOptions, ResolvedModel, profile,
};
pub use runtime::{
    CALIBRATION_POLICY, Calibration, DeviceInfo, RuntimeInfo, RuntimeOptions, devices,
    prepare_runtime,
};
pub use tokenize::{ChunkOptions, DocumentChunk};

use crate::error::{AppError, Result};
use crate::execution::Execution;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Query,
    Document,
}

enum NativeEncoder {
    Onnx(Box<onnx::OnnxEncoder>),
    Gguf(Box<gguf::GgufEncoder>),
    Remote {
        provider: Box<remote::RemoteEncoder>,
        tokenizer: Box<tokenize::HfTokenizer>,
    },
}

pub struct Encoder {
    native: NativeEncoder,
    model: ResolvedModel,
    options: RuntimeOptions,
    info: RuntimeInfo,
    startup_ms: f64,
}

impl Encoder {
    pub fn load(
        model: &ResolvedModel,
        options: RuntimeOptions,
        execution: &Execution,
    ) -> Result<Self> {
        execution.check()?;
        options.validate()?;
        hub::validate_resolved(model, execution)?;
        let requested = options.device.unwrap_or(model.profile.device);
        if matches!(model.profile.backend, Backend::OpenAi | Backend::Ollama) {
            if requested != Device::Auto {
                return Err(AppError::new(
                    "explicit_device_unavailable",
                    "Remote providers manage their own devices; use device=auto in this profile",
                ));
            }
            let provider = remote::RemoteEncoder::load(&model.profile)?;
            let tokenizer = Box::new(tokenize::HfTokenizer::load(
                model
                    .tokenizer_path
                    .as_deref()
                    .ok_or_else(|| AppError::invalid("Provider tokenizer is missing"))?,
            )?);
            return Ok(Self {
                native: NativeEncoder::Remote {
                    provider: Box::new(provider),
                    tokenizer,
                },
                model: model.clone(),
                options: options.clone(),
                startup_ms: 0.0,
                info: RuntimeInfo {
                    requested_device: requested,
                    selected_device: Device::Auto,
                    provider: format!("{:?}", model.profile.backend),
                    runtime_version: model.profile.revision.clone(),
                    device_name: "provider-managed".into(),
                    fallback_reasons: vec![],
                    calibration: vec![],
                    qualified: false,
                    batch_size: options.batch_size,
                },
            });
        }
        if requested != Device::Auto {
            let mut encoder = Self::load_device(model, &options, requested, execution)
                .map_err(|error| explicit_device_error(error, requested))?;
            encoder.info.requested_device = requested;
            encoder
                .probe(execution)
                .map_err(|error| explicit_device_error(error, requested))?;
            return Ok(encoder);
        }
        let mut reasons = Vec::new();
        if !options.calibrate {
            match runtime::read_calibration(model, &options) {
                Ok(Some(cached)) => {
                    let attempted = Self::load_device(model, &options, cached.selected_device, execution).and_then(|mut encoder| {
                        encoder.probe(execution)?;
                        Ok(encoder)
                    });
                    match attempted {
                        Ok(mut encoder) => {
                            encoder.info.requested_device = Device::Auto;
                            encoder.info.calibration = cached.measurements;
                            encoder.info.fallback_reasons.extend(cached.reasons);
                            return Ok(encoder);
                        }
                        Err(error) => {
                            execution.check()?;
                            reasons.push(format!("Cached {} selection is unavailable: {error}", cached.selected_device));
                        }
                    }
                }
                Ok(None) => reasons.push("No matching device calibration is cached for this model and workload; CPU selected. Run init/sync with calibration to compare available accelerators".into()),
                Err(error) => reasons.push(format!("CPU selected because calibration could not be read: {error}")),
            }
            let mut encoder = Self::load_device(model, &options, Device::Cpu, execution)?;
            encoder.probe(execution)?;
            encoder.info.requested_device = Device::Auto;
            encoder.info.fallback_reasons.extend(reasons);
            return Ok(encoder);
        }
        let candidates = match model.profile.backend {
            Backend::Gguf => {
                let available = devices()?;
                [Device::Metal, Device::Cuda, Device::Vulkan]
                    .into_iter()
                    .filter(|device| available.iter().any(|item| item.device == *device))
                    .collect::<Vec<_>>()
            }
            Backend::Onnx if cfg!(target_os = "linux") => vec![Device::Cuda],
            Backend::Onnx => {
                reasons.push("ONNX auto uses CPU on macOS; CoreML is an explicit experimental profile option, and Metal requires a GGUF artifact".into());
                vec![]
            }
            _ => unreachable!(),
        };
        // Calibrated auto compares the same artifact and precision on each device. The selected
        // batch size expresses the caller's workload: one for queries, larger for indexing.
        let mut cpu = Self::load_device(model, &options, Device::Cpu, execution)?;
        cpu.probe(execution)?;
        if !candidates.is_empty() {
            let measurement = cpu.measure(execution)?;
            cpu.info.calibration.push(measurement);
        }
        let mut measurements = cpu.info.calibration.clone();
        for candidate in candidates {
            execution.check()?;
            let attempted = Self::load_device(model, &options, candidate, execution).and_then(|mut encoder| {
                encoder.probe(execution)?;
                encoder.compare_cpu(&mut cpu, execution)?;
                let measurement = encoder.measure(execution)?;
                measurements.push(measurement.clone());
                let baseline = measurements.iter().find(|m| m.device == Device::Cpu);
                if baseline.is_some_and(|value| score(&measurement, options.batch_size) > score(value, options.batch_size) * 0.90) {
                    reasons.push(format!("{candidate} did not improve measured startup plus workload time by at least 10% for batch size {}; CPU retained", options.batch_size));
                    return Ok(None);
                }
                Ok(Some(encoder))
            });
            match attempted {
                Ok(Some(mut encoder)) => {
                    encoder.info.requested_device = Device::Auto;
                    encoder.info.fallback_reasons.extend(reasons);
                    encoder.info.calibration = measurements;
                    runtime::save_calibration(model, &options, &encoder.info)?;
                    return Ok(encoder);
                }
                Ok(None) => {}
                Err(error) => {
                    execution.check()?;
                    reasons.push(format!("{candidate} probe failed: {error}"));
                }
            }
        }
        cpu.info.requested_device = Device::Auto;
        cpu.info.fallback_reasons.extend(reasons);
        cpu.info.calibration = measurements;
        runtime::save_calibration(model, &options, &cpu.info)?;
        Ok(cpu)
    }

    fn load_device(
        model: &ResolvedModel,
        options: &RuntimeOptions,
        device: Device,
        execution: &Execution,
    ) -> Result<Self> {
        execution.check()?;
        let started = Instant::now();
        let (native, device_name, version, batch_size) = match model.profile.backend {
            Backend::Onnx => {
                let native = onnx::OnnxEncoder::load(model, options, device)?;
                let version = native.runtime_description();
                (
                    NativeEncoder::Onnx(Box::new(native)),
                    device.to_string(),
                    version,
                    options.batch_size,
                )
            }
            Backend::Gguf => {
                let native = gguf::GgufEncoder::load(model, options, device, execution)?;
                let name = native.device_name.clone();
                let batch_size = native.batch_size;
                (
                    NativeEncoder::Gguf(Box::new(native)),
                    name,
                    "llama-cpp-2-0.1.158".into(),
                    batch_size,
                )
            }
            _ => return Err(AppError::invalid("Not a native embedding profile")),
        };
        Ok(Self {
            native,
            model: model.clone(),
            options: options.clone(),
            startup_ms: started.elapsed().as_secs_f64() * 1000.0,
            info: RuntimeInfo {
                requested_device: device,
                selected_device: device,
                provider: format!("{:?}", model.profile.backend),
                runtime_version: version,
                device_name,
                fallback_reasons: vec![],
                calibration: vec![],
                qualified: false,
                batch_size,
            },
        })
    }

    fn probe(&mut self, execution: &Execution) -> Result<()> {
        let started = Instant::now();
        self.encode(
            &["Find a change to cache invalidation".into()],
            InputKind::Query,
            execution,
        )?;
        self.encode(
            &["Fix stale cache entries after a configuration change.".into()],
            InputKind::Document,
            execution,
        )?;
        if let NativeEncoder::Onnx(native) = &mut self.native {
            self.info
                .fallback_reasons
                .extend(native.verify_acceleration()?);
        }
        self.startup_ms += started.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    fn measure(&mut self, execution: &Execution) -> Result<Calibration> {
        let start = Instant::now();
        self.encode(
            &["Find the commit that introduced retry backoff".into()],
            InputKind::Query,
            execution,
        )?;
        let query_ms = start.elapsed().as_secs_f64() * 1000.0;
        let sample = "Commit: fix retry backoff\nChange: retry network requests with an exponential delay and preserve the original error after exhausting the retry budget.\n".repeat(3);
        let count = self.options.batch_size.min(8);
        let start = Instant::now();
        self.encode(&vec![sample; count], InputKind::Document, execution)?;
        Ok(Calibration {
            device: self.info.selected_device,
            startup_ms: self.startup_ms,
            query_ms,
            batch_ms: start.elapsed().as_secs_f64() * 1000.0,
            batch_size: count,
        })
    }

    fn compare_cpu(&mut self, reference: &mut Self, execution: &Execution) -> Result<()> {
        // Query and document prefixes can select different model behavior. Every sample must
        // agree before timing can make a candidate eligible for automatic device selection.
        for (kind, samples) in [
            (InputKind::Query, AUTO_PARITY_QUERIES),
            (InputKind::Document, AUTO_PARITY_DOCUMENTS),
        ] {
            let texts: Vec<String> = samples.iter().map(|text| (*text).to_owned()).collect();
            let expected = reference.encode(&texts, kind, execution)?;
            let actual = self.encode(&texts, kind, execution)?;
            verify_device_parity(&expected, &actual, self.info.selected_device, kind)?;
        }
        Ok(())
    }

    pub fn info(&self) -> &RuntimeInfo {
        &self.info
    }

    pub fn encode(
        &mut self,
        texts: &[String],
        kind: InputKind,
        execution: &Execution,
    ) -> Result<Vec<Vec<f32>>> {
        execution.check()?;
        let mut output = Vec::with_capacity(texts.len());
        let prefix = match kind {
            InputKind::Query => &self.model.profile.query_prefix,
            InputKind::Document => &self.model.profile.document_prefix,
        };
        let prepared: Vec<String> = texts.iter().map(|text| format!("{prefix}{text}")).collect();
        // Validate every input before sending any to a provider or producing partial results.
        for text in &prepared {
            execution.check()?;
            let count = self.count_prepared(text)?;
            if count > self.model.profile.max_tokens {
                return Err(AppError::new(
                    "token_limit",
                    format!(
                        "Input including its prefix has {count} tokens; model limit is {}. Chunk documents before embedding or shorten the query",
                        self.model.profile.max_tokens
                    ),
                ));
            }
        }
        let mut offset = 0;
        let mut batch_size = self.options.batch_size;
        while offset < prepared.len() {
            execution.check()?;
            let end = (offset + batch_size).min(prepared.len());
            let batch = &prepared[offset..end];
            let result = match &mut self.native {
                NativeEncoder::Onnx(native) => native.encode(batch, execution),
                NativeEncoder::Gguf(native) => native.encode(batch, execution),
                NativeEncoder::Remote { provider, .. } => provider.encode(batch, execution),
            }
            .and_then(|mut vectors| {
                if vectors.len() != batch.len() {
                    return Err(AppError::new(
                        "inference_failed",
                        "Encoder returned an unexpected number of vectors",
                    ));
                }
                for vector in &mut vectors {
                    normalize(vector, self.model.profile.dimensions)?;
                }
                Ok(vectors)
            });
            match result {
                Ok(vectors) => {
                    output.extend(vectors);
                    offset = end;
                }
                Err(error)
                    if is_memory_error(&error)
                        && batch_size > 1
                        && !matches!(self.native, NativeEncoder::Remote { .. }) =>
                {
                    batch_size = (batch_size / 2).max(1);
                    self.info.batch_size = batch_size;
                    self.info.fallback_reasons.push(format!("Reduced inference batch size to {batch_size} after memory allocation failed"));
                }
                Err(error)
                    if is_memory_error(&error)
                        && self.info.requested_device == Device::Auto
                        && self.info.selected_device != Device::Cpu
                        && !matches!(self.native, NativeEncoder::Remote { .. }) =>
                {
                    let mut replacement =
                        Self::load_device(&self.model, &self.options, Device::Cpu, execution)?;
                    replacement.info.requested_device = Device::Auto;
                    replacement.info.fallback_reasons = self.info.fallback_reasons.clone();
                    replacement.info.fallback_reasons.push(format!("Switched to same-artifact CPU execution after {} allocation failed: {error}", self.info.selected_device));
                    replacement.info.calibration = self.info.calibration.clone();
                    *self = replacement;
                }
                Err(error) => {
                    return Err(explicit_device_error(
                        error,
                        self.options.device.unwrap_or(self.model.profile.device),
                    ));
                }
            }
        }
        Ok(output)
    }

    fn count_prepared(&self, text: &str) -> Result<usize> {
        match &self.native {
            NativeEncoder::Onnx(native) => native.tokenizer.count(text),
            NativeEncoder::Gguf(native) => native.count_tokens(text),
            NativeEncoder::Remote { tokenizer, .. } => tokenizer.count(text),
        }
    }

    pub fn count_tokens(&self, text: &str, kind: InputKind) -> Result<usize> {
        let prefix = match kind {
            InputKind::Query => &self.model.profile.query_prefix,
            InputKind::Document => &self.model.profile.document_prefix,
        };
        self.count_prepared(&format!("{prefix}{text}"))
    }

    pub fn chunk_documents(&self, text: &str, options: ChunkOptions) -> Result<Vec<DocumentChunk>> {
        tokenize::chunk_text(text, options, self.model.profile.max_tokens, |value| {
            self.count_tokens(value, InputKind::Document)
        })
    }
}

const AUTO_PARITY_QUERIES: [&str; 3] = [
    "Which change stopped stale configuration values from being returned by the cache?",
    "Find the commit that preserved the original network error after the final retry failed.",
    "When did history search start handling Unicode file names and compact result descriptions?",
];

const AUTO_PARITY_DOCUMENTS: [&str; 3] = [
    "Commit: invalidate cached configuration after reload\nPath: src/config.rs\n- return cached;\n+ cache.clear();\n+ return load_configuration(path);\n",
    "Commit: preserve network errors during retry backoff\nPath: src/client.rs\nRequests may fail after the server accepts a connection. Retry transient failures with a bounded delay, and return the original error when the budget is exhausted.\n@@ fn fetch_with_retry @@\n- for _ in 0..retries { request()?; }\n+ let mut last_error = None;\n+ for attempt in 0..retries {\n+     match request() {\n+         Ok(response) => return Ok(response),\n+         Err(error) => last_error = Some(error),\n+     }\n+     delay(base_delay * (1 << attempt));\n+ }\n+ Err(last_error.unwrap())\n",
    "Commit: retain Unicode paths and literal template tokens\nPath: src/display.rs\nDecode paths without replacing valid Unicode. Search the entire commit body while showing its first line in result rows.\n+ let path = \"修复/cache.rs\";\n+ let template = \"<start_of_turn>user\\n<end_of_turn>\";\n",
];

fn verify_device_parity(
    expected: &[Vec<f32>],
    actual: &[Vec<f32>],
    device: Device,
    kind: InputKind,
) -> Result<()> {
    if expected.is_empty() || expected.len() != actual.len() {
        return Err(AppError::new(
            "device_unavailable",
            format!("{device} {kind:?} probe returned the wrong number of vectors"),
        ));
    }
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if expected.len() != actual.len() {
            return Err(AppError::new(
                "device_unavailable",
                format!(
                    "{device} {kind:?} sample {} changed vector dimensions",
                    index + 1
                ),
            ));
        }
        let cosine: f64 = expected
            .iter()
            .zip(actual)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        if !cosine.is_finite() || cosine < 0.9999 {
            return Err(AppError::new(
                "device_unavailable",
                format!(
                    "{device} {kind:?} sample {} differs from the same-artifact CPU reference (cosine {cosine:.8}; required 0.9999)",
                    index + 1
                ),
            ));
        }
    }
    Ok(())
}

fn explicit_device_error(mut error: AppError, device: Device) -> AppError {
    if device == Device::Auto
        || matches!(
            error.code.as_str(),
            "explicit_device_unavailable"
                | "timeout"
                | "interrupted"
                | "invalid_input"
                | "invalid_config"
                | "token_limit"
                | "tokenization_failed"
        )
    {
        return error;
    }
    error.message = format!(
        "Explicit {device} execution failed ({}): {}",
        error.code, error.message
    );
    error.code = "explicit_device_unavailable".into();
    error
}

fn score(value: &Calibration, batch_size: usize) -> f64 {
    if batch_size == 1 {
        value.startup_ms + value.query_ms
    } else {
        value.startup_ms + value.batch_ms * 32.0
    }
}

fn is_memory_error(error: &AppError) -> bool {
    let message = error.message.to_ascii_lowercase();
    [
        "out of memory",
        "out-of-memory",
        "failed to allocate",
        "bad_alloc",
        "memory allocation",
    ]
    .iter()
    .any(|value| message.contains(value))
}

pub fn normalize(vector: &mut [f32], dimensions: usize) -> Result<()> {
    if vector.len() != dimensions {
        return Err(AppError::new(
            "dimension_mismatch",
            format!(
                "Expected {dimensions} dimensions, received {}",
                vector.len()
            ),
        ));
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(AppError::new(
            "invalid_embedding",
            "Embedding contains non-finite values",
        ));
    }
    let norm = vector
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(AppError::new(
            "invalid_embedding",
            "Embedding has zero magnitude",
        ));
    }
    for value in vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
