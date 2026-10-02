use super::tokenize::HfTokenizer;
use super::{Device, ModelProfile, Pooling, ResolvedModel, RuntimeOptions};
use crate::error::{AppError, Result};
use crate::execution::Execution;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
};
use ort::session::{RunOptions, Session, builder::GraphOptimizationLevel};
use ort::value::{DynValue, Tensor};
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, mpsc};
use std::time::Duration;

pub(crate) struct OnnxEncoder {
    session: Session,
    pub tokenizer: HfTokenizer,
    profile: ModelProfile,
    device: Device,
    profiling: bool,
}

struct OrtLibrary {
    path: PathBuf,
    version: String,
}

static ORT_LIBRARY: OnceLock<std::result::Result<OrtLibrary, String>> = OnceLock::new();

fn inspect_library(path: &Path) -> Result<String> {
    // Inspect the stable API entry point before creating an OrtEnv. Old macOS builds can
    // abort during process teardown once an environment exists, even if inference succeeds.
    // The owned version string remains valid after this temporary library handle is dropped.
    unsafe {
        let library = libloading::Library::new(path).map_err(|error| {
            AppError::new(
                "runtime_invalid",
                format!("Cannot load ONNX Runtime {}: {error}", path.display()),
            )
        })?;
        let entry: libloading::Symbol<unsafe extern "system" fn() -> *const ort::sys::OrtApiBase> =
            library
                .get(b"OrtGetApiBase\0")
                .map_err(|error| AppError::new("runtime_invalid", error.to_string()))?;
        let base = entry().as_ref().ok_or_else(|| {
            AppError::new(
                "runtime_invalid",
                "ONNX Runtime returned no API entry point",
            )
        })?;
        let version_ptr = (base.GetVersionString)();
        if version_ptr.is_null() {
            return Err(AppError::new(
                "runtime_invalid",
                "ONNX Runtime returned no version",
            ));
        }
        let version = std::ffi::CStr::from_ptr(version_ptr)
            .to_string_lossy()
            .into_owned();
        let numeric: Vec<u32> = version
            .split('.')
            .take(3)
            .map(|part| part.split('-').next().unwrap_or_default().parse::<u32>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| {
                AppError::new(
                    "runtime_invalid",
                    format!("Unrecognized ONNX Runtime version: {version}"),
                )
            })?;
        if cfg!(target_os = "macos") && numeric.as_slice() < [1, 23, 2].as_slice() {
            return Err(AppError::new(
                "runtime_unsupported",
                format!(
                    "ONNX Runtime {version} can abort at process shutdown on macOS; use the managed 1.23.2 pack or a newer compatible library"
                ),
            ));
        }
        if (base.GetApi)(ort::MINOR_VERSION).is_null() {
            return Err(AppError::new(
                "runtime_unsupported",
                format!(
                    "ONNX Runtime {version} does not support required API {}",
                    ort::MINOR_VERSION
                ),
            ));
        }
        Ok(version)
    }
}

pub(super) fn initialize(options: &RuntimeOptions) -> Result<()> {
    let requested = super::runtime::locate_ort_library(options.ort_library.as_deref())?;
    let version = inspect_library(&requested)?;
    let actual = ORT_LIBRARY.get_or_init(|| {
        // ort's dynamic loader reports ABI/library failures with a panic. Keep those at this boundary.
        let result = std::panic::catch_unwind(|| {
            ort::init_from(requested.to_string_lossy())
                .with_name("nalcos")
                .commit()
        });
        match result {
            Ok(Ok(_)) => Ok(OrtLibrary {
                path: requested.clone(),
                version,
            }),
            Ok(Err(error)) => Err(error.to_string()),
            Err(_) => Err(
                "ONNX Runtime library could not be loaded or exposes an incompatible ABI".into(),
            ),
        }
    });
    match actual {
        Ok(library) if library.path == requested => Ok(()),
        Ok(_) => Err(AppError::new(
            "runtime_conflict",
            "ONNX Runtime is already initialized from another library in this process; restart with the desired runtime pack",
        )),
        Err(error) => Err(AppError::new("runtime_invalid", error.clone())),
    }
}

impl OnnxEncoder {
    pub fn runtime_description(&self) -> String {
        let version = ORT_LIBRARY
            .get()
            .and_then(|value| value.as_ref().ok())
            .map(|library| library.version.as_str())
            .unwrap_or("unknown");
        format!(
            "ONNX Runtime {version}; ort-2.0.0-rc.10 (API 22); {}",
            ort::info()
        )
    }

    pub fn load(model: &ResolvedModel, options: &RuntimeOptions, device: Device) -> Result<Self> {
        if !matches!(device, Device::Cpu | Device::Cuda | Device::CoreMl) {
            return Err(AppError::new(
                "device_unavailable",
                format!(
                    "ONNX cannot use '{device}' directly; choose cpu, cuda, coreml, or a GGUF profile for Metal"
                ),
            ));
        }
        initialize(options)?;
        let artifact = model
            .artifact_path
            .as_ref()
            .ok_or_else(|| AppError::invalid("ONNX artifact path is missing"))?;
        let tokenizer = HfTokenizer::load(
            model
                .tokenizer_path
                .as_deref()
                .ok_or_else(|| AppError::invalid("ONNX tokenizer path is missing"))?,
        )?;
        let mut builder = Session::builder()
            .map_err(ort_error)?
            .with_no_environment_execution_providers()
            .map_err(ort_error)?
            .with_intra_threads(options.threads)
            .map_err(ort_error)?
            .with_inter_threads(1)
            .map_err(ort_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?;
        let ep = match device {
            Device::Cpu => CPUExecutionProvider::default().build(),
            Device::Cuda => CUDAExecutionProvider::default()
                .with_device_id(
                    i32::try_from(options.device_index)
                        .map_err(|_| AppError::invalid("CUDA device index is too large"))?,
                )
                .with_tf32(false)
                .build(),
            Device::CoreMl => CoreMLExecutionProvider::default().build(),
            _ => unreachable!(),
        };
        builder = builder
            .with_execution_providers([ep.error_on_failure()])
            .map_err(|e| {
                AppError::new(
                    "device_unavailable",
                    format!("Cannot register {device} provider: {e}"),
                )
            })?;
        let profiling = device != Device::Cpu;
        if profiling {
            let path =
                super::hub::temporary_path(&std::env::temp_dir().join("nalcos-ort-probe.json"));
            builder = builder.with_profiling(path).map_err(ort_error)?;
        }
        let session = builder.commit_from_file(artifact).map_err(|e| {
            AppError::new(
                if device == Device::Cpu {
                    "model_invalid"
                } else {
                    "device_unavailable"
                },
                format!("Cannot load ONNX model on {device}: {e}"),
            )
        })?;
        for input in &session.inputs {
            if !matches!(
                input.name.as_str(),
                "input_ids" | "attention_mask" | "token_type_ids"
            ) {
                return Err(AppError::new(
                    "model_unsupported",
                    format!(
                        "ONNX input '{}' is unsupported; this encoder accepts input_ids, attention_mask, and token_type_ids",
                        input.name
                    ),
                ));
            }
        }
        if !session.inputs.iter().any(|i| i.name == "input_ids") {
            return Err(AppError::new(
                "model_invalid",
                "ONNX graph has no input_ids tensor",
            ));
        }
        Ok(Self {
            session,
            tokenizer,
            profile: model.profile.clone(),
            device,
            profiling,
        })
    }

    pub fn encode(&mut self, texts: &[String], execution: &Execution) -> Result<Vec<Vec<f32>>> {
        execution.check()?;
        let encoded = texts
            .iter()
            .map(|text| self.tokenizer.encode(text))
            .collect::<Result<Vec<_>>>()?;
        let sequence = encoded.iter().map(|e| e.len()).max().unwrap_or(0);
        if sequence == 0 || sequence > self.profile.max_tokens {
            return Err(AppError::new(
                "token_limit",
                format!(
                    "Input has {sequence} tokens; {} accepts at most {}",
                    self.profile.id, self.profile.max_tokens
                ),
            ));
        }
        let count = texts.len();
        let mut ids = vec![i64::from(self.tokenizer.pad_id); count * sequence];
        let mut mask = vec![0i64; count * sequence];
        let mut types = vec![i64::from(self.tokenizer.pad_type_id); count * sequence];
        for (row, value) in encoded.iter().enumerate() {
            for position in 0..value.len() {
                let index = row * sequence + position;
                ids[index] = i64::from(value.get_ids()[position]);
                mask[index] = i64::from(value.get_attention_mask()[position]);
                types[index] = i64::from(value.get_type_ids()[position]);
            }
        }
        let mut inputs: Vec<(String, DynValue)> = Vec::new();
        for input in &self.session.inputs {
            let values = match input.name.as_str() {
                "input_ids" => &ids,
                "attention_mask" => &mask,
                "token_type_ids" => &types,
                _ => unreachable!(),
            };
            let tensor =
                Tensor::<i64>::from_array(([count, sequence], values.clone().into_boxed_slice()))
                    .map_err(ort_error)?
                    .into_dyn();
            inputs.push((input.name.clone(), tensor));
        }
        let output_name = if let Some(output) = &self.profile.output {
            output.clone()
        } else {
            ["last_hidden_state", "sentence_embedding", "embeddings"]
                .into_iter()
                .find(|name| self.session.outputs.iter().any(|o| o.name == *name))
                .map(str::to_owned)
                .or_else(|| self.session.outputs.first().map(|o| o.name.clone()))
                .ok_or_else(|| AppError::new("model_invalid", "ONNX model has no outputs"))?
        };
        let run_options = RunOptions::new().map_err(ort_error)?;
        let (finished, wait) = mpsc::channel();
        std::thread::scope(|scope| {
            let cancellation_options = &run_options;
            let watchdog = scope.spawn(move || -> Result<()> {
                loop {
                    match wait.recv_timeout(Duration::from_millis(10)) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
                        Err(mpsc::RecvTimeoutError::Timeout) if execution.check().is_err() => {
                            cancellation_options.terminate().map_err(ort_error)?;
                            return Ok(());
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                }
            });
            let result = self
                .session
                .run_with_options(inputs, &run_options)
                .map_err(ort_error)
                .and_then(|outputs| {
                    let value = outputs.get(&output_name).ok_or_else(|| {
                        AppError::new(
                            "model_invalid",
                            format!("ONNX output '{output_name}' is absent"),
                        )
                    })?;
                    let (shape, values) = value.try_extract_tensor::<f32>().map_err(ort_error)?;
                    pool(
                        values,
                        shape,
                        count,
                        sequence,
                        self.profile.dimensions,
                        &mask,
                        self.profile.pooling,
                    )
                });
            // A disconnected receiver means the watchdog already terminated the run.
            let _ = finished.send(());
            watchdog.join().map_err(|_| {
                AppError::new("runtime_failed", "ONNX cancellation monitor failed")
            })??;
            execution.check()?;
            result
        })
    }

    /// Registration alone is not proof of acceleration; inspect the actual warm-up node events.
    pub fn verify_acceleration(&mut self) -> Result<Vec<String>> {
        if !self.profiling {
            return Ok(vec![]);
        }
        self.profiling = false;
        let path = self.session.end_profiling().map_err(ort_error)?;
        let bytes = std::fs::read(&path)?;
        let events: serde_json::Value = serde_json::from_slice(&bytes)?;
        let mut providers = std::collections::BTreeSet::new();
        for event in events.as_array().into_iter().flatten() {
            if let Some(provider) = event.pointer("/args/provider").and_then(|v| v.as_str()) {
                providers.insert(provider.to_owned());
            }
        }
        // This file is generated probe output in the OS temporary directory, not user data.
        std::fs::remove_file(&path)?;
        let expected = match self.device {
            Device::Cuda => "CUDAExecutionProvider",
            Device::CoreMl => "CoreMLExecutionProvider",
            _ => "CPUExecutionProvider",
        };
        if !providers.contains(expected) {
            return Err(AppError::new(
                "device_unavailable",
                format!(
                    "{expected} registered but executed no graph nodes; observed providers: {providers:?}"
                ),
            ));
        }
        Ok(if providers.contains("CPUExecutionProvider") {
            vec![format!(
                "{expected} executes the supported subgraph; remaining ONNX nodes execute on CPU"
            )]
        } else {
            vec![]
        })
    }
}

fn ort_error(error: ort::Error) -> AppError {
    AppError::new("inference_failed", error.to_string())
}

pub(crate) fn pool(
    values: &[f32],
    shape: &[i64],
    count: usize,
    sequence: usize,
    dimensions: usize,
    mask: &[i64],
    pooling: Pooling,
) -> Result<Vec<Vec<f32>>> {
    if shape == [count as i64, dimensions as i64] {
        if pooling != Pooling::Model {
            return Err(AppError::new(
                "model_invalid",
                "ONNX graph returns already pooled vectors; set pooling = 'model' explicitly",
            ));
        }
        return Ok(values
            .chunks_exact(dimensions)
            .map(|v| v.to_vec())
            .collect());
    }
    if shape != [count as i64, sequence as i64, dimensions as i64] || pooling == Pooling::Model {
        return Err(AppError::new(
            "model_invalid",
            format!(
                "ONNX output shape {shape:?} does not match the configured token embeddings ({count}, {sequence}, {dimensions}) and pooling {pooling:?}"
            ),
        ));
    }
    let mut output = Vec::with_capacity(count);
    for row in 0..count {
        let positions: Vec<usize> = (0..sequence)
            .filter(|p| mask[row * sequence + p] != 0)
            .collect();
        if positions.is_empty() {
            return Err(AppError::new(
                "model_invalid",
                "ONNX attention mask contains an empty sequence",
            ));
        }
        let mut vector = vec![0f32; dimensions];
        match pooling {
            Pooling::Mean => {
                for &position in &positions {
                    let offset = (row * sequence + position) * dimensions;
                    for (v, contribution) in
                        vector.iter_mut().zip(&values[offset..offset + dimensions])
                    {
                        *v += contribution;
                    }
                }
                for v in &mut vector {
                    *v /= positions.len() as f32;
                }
            }
            Pooling::Cls | Pooling::Last => {
                let position = if pooling == Pooling::Cls {
                    positions[0]
                } else {
                    positions[positions.len() - 1]
                };
                let offset = (row * sequence + position) * dimensions;
                vector.copy_from_slice(&values[offset..offset + dimensions]);
            }
            Pooling::Model => unreachable!(),
        }
        output.push(vector);
    }
    Ok(output)
}
