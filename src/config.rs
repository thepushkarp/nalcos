use crate::{
    embedding::{Device, ModelProfile, RuntimeOptions},
    error::{AppError, Result},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub data_dir: Option<PathBuf>,
    pub default_model: Option<String>,
    pub index: IndexConfig,
    pub search: SearchConfig,
    pub runtime: RuntimeConfig,
    pub profiles: BTreeMap<String, ModelProfile>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            data_dir: None,
            default_model: None,
            index: IndexConfig::default(),
            search: SearchConfig::default(),
            runtime: RuntimeConfig::default(),
            profiles: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct IndexConfig {
    pub exclude_paths: Vec<String>,
    pub max_blob_bytes: usize,
    pub max_patch_bytes: usize,
    pub max_document_bytes: usize,
    pub chunk_overlap_tokens: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            exclude_paths: vec![],
            max_blob_bytes: 1_048_576,
            max_patch_bytes: 2_097_152,
            max_document_bytes: 16_384,
            chunk_overlap_tokens: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    pub candidate_limit: usize,
    pub rrf_k: f64,
    pub lexical_weight: f64,
    pub semantic_weight: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            candidate_limit: 100,
            rrf_k: 60.0,
            lexical_weight: 1.0,
            semantic_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    pub query_device: String,
    pub index_device: String,
    pub threads: usize,
    pub batch_size: usize,
    pub ort_library: Option<PathBuf>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            query_device: "auto".into(),
            index_device: "auto".into(),
            threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(4)),
            batch_size: 8,
            ort_library: None,
        }
    }
}

impl Config {
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let default_path =
            directories::BaseDirs::new().map(|d| d.config_dir().join("nalcos/config.toml"));
        let path = explicit.map(Path::to_path_buf).or(default_path);
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let config = match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str::<Self>(&text).map_err(|e| {
                AppError::new("invalid_config", format!("Read {}: {e}", path.display()))
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && explicit.is_none() => {
                Self::default()
            }
            Err(e) => {
                return Err(AppError::new(
                    "invalid_config",
                    format!("Read {}: {e}", path.display()),
                ));
            }
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.version != 1 {
            return Err(AppError::new(
                "unsupported_schema",
                format!(
                    "Configuration version {} is unsupported; expected 1",
                    self.version
                ),
            ));
        }
        if self.runtime.threads == 0
            || self.runtime.threads > 1024
            || self.runtime.batch_size == 0
            || self.runtime.batch_size > 1024
        {
            return Err(AppError::new(
                "invalid_config",
                "threads and batch_size must be between 1 and 1024",
            ));
        }
        if self.index.max_document_bytes < 256 {
            return Err(AppError::new(
                "invalid_config",
                "index.max_document_bytes must be at least 256",
            ));
        }
        for (name, value) in [
            ("max_blob_bytes", self.index.max_blob_bytes),
            ("max_patch_bytes", self.index.max_patch_bytes),
            ("max_document_bytes", self.index.max_document_bytes),
        ] {
            if value == 0 {
                return Err(AppError::new(
                    "invalid_config",
                    format!("index.{name} must be positive"),
                ));
            }
        }
        if !(1..=10_000).contains(&self.search.candidate_limit)
            || !self.search.rrf_k.is_finite()
            || self.search.rrf_k <= 0.0
            || !self.search.lexical_weight.is_finite()
            || !self.search.semantic_weight.is_finite()
            || self.search.lexical_weight <= 0.0
            || self.search.semantic_weight <= 0.0
        {
            return Err(AppError::new(
                "invalid_config",
                "search weights and rrf_k must be finite and positive; candidate_limit must be 1..=10000",
            ));
        }
        parse_device(&self.runtime.query_device)?;
        parse_device(&self.runtime.index_device)?;
        Ok(())
    }

    pub fn index_path(&self, identity: &str) -> Result<PathBuf> {
        let base = std::env::var_os("NALCOS_DATA_DIR")
            .map(PathBuf::from)
            .or_else(|| self.data_dir.clone())
            .or_else(|| {
                directories::ProjectDirs::from("", "", "nalcos")
                    .map(|d| d.data_local_dir().to_path_buf())
            })
            .ok_or_else(|| {
                AppError::new(
                    "data_dir_unavailable",
                    "Set NALCOS_DATA_DIR or data_dir in the configuration",
                )
            })?;
        Ok(base.join("repos").join(identity).join("index.sqlite3"))
    }

    pub fn runtime_options(
        &self,
        indexing: bool,
        device_override: Option<&str>,
        calibrate: bool,
    ) -> Result<RuntimeOptions> {
        let configured = if indexing {
            &self.runtime.index_device
        } else {
            &self.runtime.query_device
        };
        let (device, device_index) = parse_device(device_override.unwrap_or(configured))?;
        Ok(RuntimeOptions {
            // Automatic workload settings defer to an explicit model profile.
            // `--device auto` remains an explicit request to calibrate instead.
            device: (device_override.is_some() || device != Device::Auto).then_some(device),
            device_index,
            threads: self.runtime.threads,
            batch_size: if indexing { self.runtime.batch_size } else { 1 },
            calibrate,
            ort_library: self.runtime.ort_library.clone(),
        })
    }
}

pub fn parse_device(value: &str) -> Result<(Device, usize)> {
    let (name, index) = match value.split_once(':') {
        Some((name, index)) => (
            name,
            index
                .parse::<usize>()
                .map_err(|_| AppError::invalid(format!("Invalid device index in {value}")))?,
        ),
        None => (value, 0),
    };
    let device = match name {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "cuda" => Device::Cuda,
        "vulkan" => Device::Vulkan,
        _ => {
            return Err(AppError::invalid(format!(
                "Unknown device {value}; use auto, cpu, metal, cuda[:N], or vulkan[:N]"
            )));
        }
    };
    if value.contains(':') && !matches!(device, Device::Cuda | Device::Vulkan) {
        return Err(AppError::invalid(format!(
            "Device {name} does not accept an index"
        )));
    }
    Ok((device, index))
}
