use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{path::PathBuf, time::Duration};

#[derive(Debug, Parser)]
#[command(version, about = "Find verifiable evidence in local Git history", long_about = None)]
pub struct Cli {
    #[arg(long, global = true, default_value = ".")]
    pub repo: PathBuf,
    #[arg(long, global = true, env = "NALCOS_CONFIG")]
    pub config: Option<PathBuf>,
    #[arg(long, global = true)]
    pub json: bool,
    #[arg(long, global = true, help = "Disallow network access and downloads")]
    pub offline: bool,
    #[arg(long, global = true, value_parser = parse_duration, help = "Time budget, e.g. 30s or 5m (GGUF checks between native batches)")]
    pub timeout: Option<Duration>,
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
    #[arg(
        long,
        global = true,
        help = "Inference device: auto, cpu, metal, cuda[:N], or vulkan[:N]"
    )]
    pub device: Option<String>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Install MiniLM INT8 on CPU (or the selected model) and build an initial index.
    Init(InitArgs),
    /// Reconcile history, resume indexing, or explicitly change embedding models.
    Sync(SyncArgs),
    /// Retrieve commits with source-verified evidence.
    Search(SearchArgs),
    /// Inspect a commit or a stable evidence identifier.
    Show(ShowArgs),
    /// Inspect readiness without changing the index.
    Status(StatusArgs),
}

#[derive(Debug, Clone, Default, Args)]
pub struct ScopeArgs {
    #[arg(long = "ref", conflicts_with_all = ["range", "all_refs"])]
    pub refs: Vec<String>,
    #[arg(long, conflicts_with_all = ["refs", "all_refs"])]
    pub range: Option<String>,
    #[arg(long, conflicts_with_all = ["refs", "range"])]
    pub all_refs: bool,
    #[arg(long)]
    pub first_parent: bool,
}

#[derive(Debug, Clone, Default, Args)]
pub struct ModelArgs {
    #[arg(
        long,
        help = "Hugging Face model ID, preset (e.g. minilm-int8), or profile:NAME"
    )]
    pub model: Option<String>,
    #[arg(long, help = "Resolve an explicit model revision during setup")]
    pub revision: Option<String>,
    #[arg(long, help = "Artifact filename within the model repository")]
    pub variant: Option<String>,
    #[arg(long, help = "Force a fresh, resumable embedding generation")]
    pub reembed: bool,
}

#[derive(Debug, Args)]
pub struct InitArgs {
    #[command(flatten)]
    pub scope: ScopeArgs,
    #[command(flatten)]
    pub model: ModelArgs,
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct SyncArgs {
    #[command(flatten)]
    pub scope: ScopeArgs,
    #[command(flatten)]
    pub model: ModelArgs,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(
        long,
        conflicts_with = "dry_run",
        help = "Reconcile in the foreground every two seconds"
    )]
    pub watch: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Hybrid,
    Lexical,
    Semantic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    Auto,
    Cached,
    Wait,
}

#[derive(Debug, Args)]
pub struct SearchArgs {
    pub query: String,
    #[command(flatten)]
    pub scope: ScopeArgs,
    #[arg(long, value_enum, default_value = "hybrid")]
    pub mode: SearchMode,
    #[arg(long, value_enum, default_value = "auto")]
    pub freshness: Freshness,
    #[arg(long = "path")]
    pub paths: Vec<String>,
    #[arg(long)]
    pub author: Option<String>,
    #[arg(long)]
    pub since: Option<String>,
    #[arg(long)]
    pub until: Option<String>,
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=100))]
    pub limit: u32,
    #[arg(long, default_value_t = 16_384)]
    pub max_bytes: usize,
}

#[derive(Debug, Args)]
#[command(group(clap::ArgGroup::new("source").required(true).args(["commit", "evidence"])))]
pub struct ShowArgs {
    pub commit: Option<String>,
    #[arg(long, conflicts_with_all = ["parent", "paths", "context"])]
    pub evidence: Option<String>,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub parent: u32,
    #[arg(long = "path")]
    pub paths: Vec<String>,
    #[arg(long, default_value_t = 3)]
    pub context: u32,
    #[arg(long, default_value_t = 16_384)]
    pub max_bytes: usize,
    #[arg(long)]
    pub full: bool,
}

#[derive(Debug, Args)]
pub struct StatusArgs {
    #[command(flatten)]
    pub scope: ScopeArgs,
    #[arg(
        long,
        help = "Probe the installed model using synthetic text; never download or repair"
    )]
    pub check: bool,
}

fn parse_duration(value: &str) -> Result<Duration, String> {
    let duration = if let Ok(seconds) = value.parse::<u64>() {
        Duration::from_secs(seconds)
    } else {
        humantime::parse_duration(value).map_err(|e| e.to_string())?
    };
    if duration.is_zero() {
        return Err("timeout must be greater than zero".into());
    }
    Ok(duration)
}
