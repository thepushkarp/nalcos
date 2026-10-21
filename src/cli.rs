use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{path::PathBuf, time::Duration};

#[derive(Debug, Parser)]
#[command(version, about = "Find verifiable evidence in local Git history", long_about = None)]
pub struct Cli {
    /// Any directory inside the Git checkout to use
    #[arg(long, global = true, default_value = ".")]
    pub repo: PathBuf,
    /// TOML configuration file
    #[arg(long, global = true, env = "NALCOS_CONFIG")]
    pub config: Option<PathBuf>,
    /// Print one JSON object on stdout; progress and diagnostics stay on stderr
    #[arg(long, global = true)]
    pub json: bool,
    #[arg(long, global = true, help = "Disallow network access and downloads")]
    pub offline: bool,
    #[arg(long, global = true, value_parser = parse_duration, help = "Time budget, e.g. 30s or 5m (GGUF checks between native batches)")]
    pub timeout: Option<Duration>,
    /// Report repository, index, and lock diagnostics; -vv adds search timings
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
    #[arg(
        long,
        global = true,
        help = "Inference device: auto, cpu, metal, coreml, cuda[:N], or vulkan[:N]"
    )]
    pub device: Option<String>,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Install CodeRank FP16 (or the selected model), select its fastest device, and index history.
    #[command(after_help = INIT_EXAMPLES)]
    Init(InitArgs),
    /// Reconcile history, resume indexing, or explicitly change embedding models.
    Sync(SyncArgs),
    /// Retrieve commits with source-verified evidence.
    #[command(after_help = SEARCH_EXAMPLES)]
    Search(SearchArgs),
    /// Inspect a commit or a stable evidence identifier.
    #[command(after_help = SHOW_EXAMPLES)]
    Show(ShowArgs),
    /// Inspect readiness without changing the index.
    Status(StatusArgs),
    /// Inspect recorded source omissions without changing the index.
    Omissions(OmissionsArgs),
}

#[derive(Debug, Clone, Default, Args)]
pub struct ScopeArgs {
    /// Use the ancestry of REF instead of HEAD; repeat for several refs
    #[arg(long = "ref", conflicts_with_all = ["range", "all_refs"])]
    pub refs: Vec<String>,
    /// Use commits reachable from B but not from A; only the two-dot A..B form
    #[arg(long, conflicts_with_all = ["refs", "all_refs"])]
    pub range: Option<String>,
    /// Use every local ref that resolves to a commit, plus worktree HEADs
    #[arg(long, conflicts_with_all = ["refs", "range"])]
    pub all_refs: bool,
    /// Follow only the first parent of merge commits
    #[arg(long)]
    pub first_parent: bool,
}

#[derive(Debug, Clone, Default, Args)]
pub struct ModelArgs {
    #[arg(
        long,
        help = "Hugging Face model ID, preset (e.g. coderank), or profile:NAME"
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
    /// Plan the work without creating an index, downloading, or probing an encoder
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct SyncArgs {
    #[command(flatten)]
    pub scope: ScopeArgs,
    #[command(flatten)]
    pub model: ModelArgs,
    /// Plan the work without creating an index, downloading, or probing an encoder
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
    /// Fuse lexical and semantic rankings; fall back to lexical with a warning
    Hybrid,
    /// Keyword matching only; needs no embedding model
    Lexical,
    /// Embedding similarity only; fail without usable embeddings
    Semantic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    /// Index new commits for up to two seconds, then search; writes the index
    Auto,
    /// Search the existing index without writing it
    Cached,
    /// Finish indexing the requested scope before searching; pair with --timeout
    Wait,
}

#[derive(Debug, Args)]
pub struct SearchArgs {
    /// Description of the change to find; lexical matching ORs its words
    pub query: String,
    #[command(flatten)]
    pub scope: ScopeArgs,
    /// Retrieval mode
    #[arg(long, value_enum, default_value = "hybrid")]
    pub mode: SearchMode,
    /// Index update policy before searching
    #[arg(long, value_enum, default_value = "auto")]
    pub freshness: Freshness,
    /// Restrict evidence to a Git pathspec; repeatable
    #[arg(long = "path")]
    pub paths: Vec<String>,
    /// Case-insensitive substring of the author's "Name <email>"
    #[arg(long)]
    pub author: Option<String>,
    /// Earliest committer date: YYYY-MM-DD (UTC) or an RFC 3339 timestamp
    #[arg(long)]
    pub since: Option<String>,
    /// Latest committer date, inclusive: YYYY-MM-DD (whole UTC day) or RFC 3339
    #[arg(long)]
    pub until: Option<String>,
    /// Maximum number of commits to return (1-100)
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=100))]
    pub limit: u32,
    /// Byte budget shared by all result excerpts
    #[arg(long, default_value_t = 16_384)]
    pub max_bytes: usize,
}

#[derive(Debug, Args)]
#[command(group(clap::ArgGroup::new("source").required(true).args(["commit", "evidence"])))]
pub struct ShowArgs {
    /// Commit to show as a patch against one parent
    pub commit: Option<String>,
    /// Exact evidence ID from search results (e1:...)
    #[arg(long, conflicts_with_all = ["parent", "paths", "context"])]
    pub evidence: Option<String>,
    /// Parent number to compare against, starting at 1
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub parent: u32,
    /// Restrict the patch to a Git pathspec; repeatable
    #[arg(long = "path")]
    pub paths: Vec<String>,
    /// Unchanged lines shown around each hunk
    #[arg(long, default_value_t = 3)]
    pub context: u32,
    /// Byte budget for the patch or evidence text
    #[arg(long, default_value_t = 16_384)]
    pub max_bytes: usize,
    /// Lift the byte budget; extraction safety limits still apply
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

#[derive(Debug, Args)]
pub struct OmissionsArgs {
    #[command(flatten)]
    pub scope: ScopeArgs,
    /// Maximum number of records to return (1-100)
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=100))]
    pub limit: u32,
    /// Records to skip; pass next_offset from the previous page
    #[arg(long, default_value_t = 0)]
    pub offset: u32,
}

const INIT_EXAMPLES: &str = "\
Examples:
  nalcos init
  nalcos init --dry-run
  nalcos init --model minilm";

const SEARCH_EXAMPLES: &str = "\
Examples:
  nalcos search \"stop retrying requests after cancellation\"
  nalcos search \"cache invalidation\" --mode lexical --freshness wait
  nalcos search \"retry backoff\" --range v1.8.0..HEAD --path 'src/net/**' --limit 5
  nalcos --json search \"flaky timeout\" --freshness cached --timeout 60s";

const SHOW_EXAMPLES: &str = "\
Examples:
  nalcos show HEAD~3
  nalcos show HEAD~3 --path 'src/net/**' --context 5
  nalcos show MERGE_SHA --parent 2
  nalcos --json show --evidence EVIDENCE_ID";

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
