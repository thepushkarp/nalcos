use super::hub::{
    acquire_lock, checked_response, client, inspect_file, network_error, temporary_path,
};
use super::{Device, ResolveOptions};
use crate::error::{AppError, Result};
use crate::execution::Execution;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub device: Option<Device>,
    pub device_index: usize,
    pub threads: usize,
    pub batch_size: usize,
    pub calibrate: bool,
    pub ort_library: Option<PathBuf>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            device: None,
            device_index: 0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(4)),
            batch_size: 8,
            calibrate: true,
            ort_library: None,
        }
    }
}

impl RuntimeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.threads == 0
            || self.threads > 1024
            || self.batch_size == 0
            || self.batch_size > 1024
        {
            return Err(AppError::invalid(
                "Embedding threads and batch_size must be between 1 and 1024",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub device: Device,
    pub startup_ms: f64,
    pub query_ms: f64,
    pub batch_ms: f64,
    pub batch_size: usize,
}

/// Version of the numerical and performance gates used to select automatic devices.
/// Persist this with applied runtime settings so a stricter policy requires fresh setup.
pub const CALIBRATION_POLICY: &str =
    "calibration-policy-v2-parity9999-query3-doc3-10pct-32-batches";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CachedCalibration {
    pub selected_device: Device,
    pub measurements: Vec<Calibration>,
    pub reasons: Vec<String>,
}

fn calibration_path(model: &super::ResolvedModel, options: &RuntimeOptions) -> Result<PathBuf> {
    let inventory = devices()?;
    let library = if model.profile.backend == super::Backend::Onnx {
        let path = locate_ort_library(options.ort_library.as_deref())?;
        let metadata = fs::metadata(&path)?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos().to_string());
        Some((path, metadata.len(), modified))
    } else {
        None
    };
    let descriptor = serde_json::to_vec(&(
        CALIBRATION_POLICY,
        env!("CARGO_PKG_VERSION"),
        "llama-cpp-2-0.1.158;ort-rc10",
        std::env::consts::OS,
        std::env::consts::ARCH,
        &model.fingerprint,
        &model.profile.query_prefix,
        options.threads,
        options.batch_size,
        options.device_index,
        inventory,
        library,
        std::env::var("CUDA_VISIBLE_DEVICES").ok(),
    ))?;
    Ok(runtime_root()?
        .join("calibration")
        .join(format!("{:x}.json", Sha256::digest(descriptor))))
}

pub(crate) fn read_calibration(
    model: &super::ResolvedModel,
    options: &RuntimeOptions,
) -> Result<Option<CachedCalibration>> {
    let path = calibration_path(model, options)?;
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|e| {
            AppError::new(
                "calibration_invalid",
                format!("Cached device calibration is invalid: {e}"),
            )
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(AppError::new("calibration_unavailable", error.to_string())),
    }
}

pub(crate) fn save_calibration(
    model: &super::ResolvedModel,
    options: &RuntimeOptions,
    info: &RuntimeInfo,
) -> Result<()> {
    let path = calibration_path(model, options)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let cached = CachedCalibration {
        selected_device: info.selected_device,
        measurements: info.calibration.clone(),
        reasons: info.fallback_reasons.clone(),
    };
    super::hub::atomic_write(&path, &serde_json::to_vec(&cached)?)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub requested_device: Device,
    pub selected_device: Device,
    pub provider: String,
    pub runtime_version: String,
    pub device_name: String,
    pub fallback_reasons: Vec<String>,
    pub calibration: Vec<Calibration>,
    pub qualified: bool,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub name: String,
    pub device: Device,
    pub index: usize,
    pub memory_bytes: Option<usize>,
    pub compiled: bool,
    pub qualified: bool,
}

/// Enumerates actual compiled GGML devices; loading and inference are separate capability probes.
pub fn devices() -> Result<Vec<DeviceInfo>> {
    super::gguf::devices()
}

pub fn prepare_runtime(
    model: &super::ResolvedModel,
    runtime: &RuntimeOptions,
    options: ResolveOptions,
    execution: &Execution,
) -> Result<()> {
    execution.check()?;
    if model.profile.backend == super::Backend::Onnx {
        if runtime.ort_library.is_some() {
            locate_ort_library(runtime.ort_library.as_deref())?;
        } else {
            ensure_cpu_runtime(options, execution)?;
        }
    }
    Ok(())
}

pub(crate) const ORT_VERSION: &str = "1.23.2";

struct RuntimePack {
    archive: &'static str,
    sha256: &'static str,
    library: &'static str,
}

fn pack() -> Result<RuntimePack> {
    // 1.23.2 avoids the process-exit OrtEnv mutex crash in the 1.22 macOS runtime.
    // SHA256s are the official release asset digests; no mutable latest URL is used.
    let (archive, sha256, library) = match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => (
            "onnxruntime-osx-arm64-1.23.2.tgz",
            "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856",
            "libonnxruntime.1.23.2.dylib",
        ),
        ("macos", "x86_64") => (
            "onnxruntime-osx-x86_64-1.23.2.tgz",
            "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6",
            "libonnxruntime.1.23.2.dylib",
        ),
        ("linux", "x86_64") => (
            "onnxruntime-linux-x64-1.23.2.tgz",
            "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b",
            "libonnxruntime.so.1.23.2",
        ),
        ("linux", "aarch64") => (
            "onnxruntime-linux-aarch64-1.23.2.tgz",
            "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c",
            "libonnxruntime.so.1.23.2",
        ),
        _ => {
            return Err(AppError::new(
                "runtime_unsupported",
                "Managed ONNX Runtime packs support macOS and Linux on x86_64 and ARM64; supply NALCOS_ORT_LIBRARY for another platform",
            ));
        }
    };
    Ok(RuntimePack {
        archive,
        sha256,
        library,
    })
}

fn runtime_root() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("NALCOS_RUNTIME_CACHE") {
        return Ok(PathBuf::from(path));
    }
    let dirs = directories::ProjectDirs::from("dev", "nalcos", "nalcos").ok_or_else(|| {
        AppError::new(
            "cache_unavailable",
            "Cannot locate a user cache directory; set NALCOS_RUNTIME_CACHE",
        )
    })?;
    Ok(dirs.cache_dir().join("runtimes"))
}

fn pack_directory(pack: &RuntimePack) -> Result<PathBuf> {
    Ok(runtime_root()?.join(pack.archive.trim_end_matches(".tgz")))
}

pub(crate) fn locate_ort_library(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("NALCOS_ORT_LIBRARY").map(PathBuf::from))
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
    {
        if path.is_file() {
            return Ok(path);
        }
        return Err(AppError::new(
            "runtime_missing",
            format!("ONNX Runtime library {} does not exist", path.display()),
        ));
    }
    let pack = pack()?;
    // Release distributions may place their runtime next to the executable.
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        let bundled = parent.join("lib").join(pack.library);
        if bundled.is_file() {
            return Ok(bundled);
        }
    }
    let managed = pack_directory(&pack)?.join(pack.library);
    if managed.is_file() {
        return Ok(managed);
    }
    Err(AppError::new(
        "runtime_missing",
        format!(
            "ONNX Runtime {ORT_VERSION} is not installed; run init with downloads enabled, or set NALCOS_ORT_LIBRARY to a compatible library"
        ),
    ))
}

/// Install a pinned CPU runtime in the app cache only during explicit initialization/sync.
pub fn ensure_cpu_runtime(options: ResolveOptions, execution: &Execution) -> Result<PathBuf> {
    execution.check()?;
    if let Ok(path) = locate_ort_library(None) {
        return Ok(path);
    }
    if options.offline || !options.allow_download {
        return locate_ort_library(None);
    }
    // An invalid explicit override is an error, not permission to silently replace it.
    if std::env::var_os("NALCOS_ORT_LIBRARY").is_some()
        || std::env::var_os("ORT_DYLIB_PATH").is_some()
    {
        return locate_ort_library(None);
    }
    let pack = pack()?;
    let directory = pack_directory(&pack)?;
    let _lock = acquire_lock(&directory.with_extension("lock"), execution)?;
    let library = directory.join(pack.library);
    if library.is_file() {
        return Ok(library);
    }
    let archive_path = runtime_root()?.join(pack.archive);
    if !archive_path.is_file() || inspect_file(&archive_path, execution)?.sha256 != pack.sha256 {
        let temporary = temporary_path(&archive_path);
        let url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{}",
            pack.archive
        );
        let response = client(execution)?.get(url).send().map_err(network_error)?;
        let mut response = checked_response(response, "download ONNX Runtime")?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65_536];
        loop {
            execution.check()?;
            let count = response.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            file.write_all(&buffer[..count])?;
        }
        if format!("{:x}", hash.finalize()) != pack.sha256 {
            return Err(AppError::new(
                "runtime_checksum_mismatch",
                "ONNX Runtime archive does not match the pinned release checksum",
            ));
        }
        file.sync_all()?;
        fs::rename(temporary, &archive_path)?;
    }
    fs::create_dir_all(&directory)?;
    let decoder = flate2::read::GzDecoder::new(File::open(&archive_path)?);
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries()? {
        execution.check()?;
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let archive_path = entry.path()?;
        let path = archive_path
            .strip_prefix(".")
            .unwrap_or(archive_path.as_ref());
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let root = Path::new(pack.archive.trim_end_matches(".tgz"));
        let accepted = path == root.join("lib").join(pack.library)
            || path == root.join("LICENSE")
            || path == root.join("ThirdPartyNotices.txt");
        if !accepted {
            continue;
        }
        // Exact archive paths distinguish the loadable library from same-named dSYM files.
        // Only explicitly named regular files are extracted; symlinks are ignored.
        let target = directory.join(name);
        let temporary = temporary_path(&target);
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        std::io::copy(&mut entry, &mut output)?;
        output.sync_all()?;
        fs::rename(temporary, target)?;
    }
    if !library.is_file() {
        return Err(AppError::new(
            "runtime_invalid",
            "Verified runtime archive contained no expected shared library",
        ));
    }
    Ok(library)
}
