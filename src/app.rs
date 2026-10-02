use crate::{
    cli::{Cli, Command, Freshness, ModelArgs, ScopeArgs, SearchArgs, SearchMode, ShowArgs},
    config::{Config, IndexConfig},
    embedding::{
        self, Backend, ChunkOptions, Encoder, InputKind, ModelProfile, ModelRequest,
        ResolveOptions, ResolvedModel,
    },
    error::{AppError, Result},
    execution::Execution,
    git::{
        DocumentRecord, EXTRACTION_VERSION, FilterOptions, IndexOptions, Repository, ScopeOptions,
        ScopeSnapshot, ShowOptions,
    },
    output::{self, truncate_utf8},
    store::{Candidate, CandidateFilter, Coverage, Generation, Store, VectorChunk},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    path::Path,
    time::{Duration, Instant},
};

const INDEX_POLICY_KEY: &str = "canonical_index_policy";
const EXTRACTION_VERSION_KEY: &str = "canonical_extraction_version";
const SOURCE_REFRESH_WARNING: &str =
    "Cached source extraction is outdated; run nalcos sync to refresh indexed patches";
const PROVIDER_PROBE: &str =
    "A small deterministic embedding identity probe for local history search.";

#[derive(Clone, Serialize, Deserialize)]
struct ModelSnapshot {
    resolved: ResolvedModel,
    selector: Option<String>,
    #[serde(default)]
    requested_profile: Option<ModelProfile>,
    #[serde(default)]
    config_default_model: Option<String>,
    #[serde(default)]
    config_profile: Option<ModelProfile>,
    chunk_overlap_tokens: usize,
    #[serde(default)]
    provider_probe: Option<Vec<f32>>,
    #[serde(default)]
    runtime: Option<Value>,
    #[serde(default)]
    runtime_policy: Option<Value>,
}

pub fn run(cli: &Cli, execution: &Execution) -> Result<Value> {
    let config = Config::load(cli.config.as_deref())?;
    let repo = Repository::discover(&cli.repo, execution)?;
    let index_path = config.index_path(&repo.identity)?;
    if cli.verbose > 0 {
        eprintln!(
            "Repository: {}\nIndex: {}",
            repo.work_dir.display(),
            index_path.display()
        );
    }
    let result = match &cli.command {
        Command::Init(args) => sync(
            cli,
            &config,
            &repo,
            &index_path,
            &args.scope,
            &args.model,
            args.dry_run,
            true,
            true,
            execution,
        )?,
        Command::Sync(args) => {
            let first = sync(
                cli,
                &config,
                &repo,
                &index_path,
                &args.scope,
                &args.model,
                args.dry_run,
                false,
                true,
                execution,
            )?;
            if !args.watch {
                return Ok(first);
            }
            output::emit(&first, cli.json)?;
            let mut signature = watch_signature(&first);
            loop {
                for _ in 0..20 {
                    execution.check()?;
                    std::thread::sleep(Duration::from_millis(100));
                }
                let next = sync(
                    cli,
                    &config,
                    &repo,
                    &index_path,
                    &args.scope,
                    &ModelArgs::default(),
                    false,
                    false,
                    false,
                    execution,
                )?;
                let next_signature = watch_signature(&next);
                if signature != next_signature
                    || next["commits_added"].as_u64().unwrap_or(0) != 0
                    || next["documents_embedded"].as_u64().unwrap_or(0) != 0
                {
                    output::emit(&next, cli.json)?;
                    signature = next_signature;
                }
            }
        }
        Command::Search(args) => search(cli, &config, &repo, &index_path, args, execution)?,
        Command::Show(args) => show(&repo, &index_path, args, execution)?,
        Command::Status(args) => status(
            cli,
            &config,
            &repo,
            &index_path,
            &args.scope,
            args.check,
            execution,
        )?,
    };
    Ok(result)
}

fn watch_signature(value: &Value) -> Value {
    json!([
        value["scope"],
        value["history_coverage"],
        value["embedding_coverage"],
        value["active_generation"],
        value["staging_generation"],
        value["warnings"]
    ])
}

fn scope_options(args: &ScopeArgs) -> ScopeOptions {
    ScopeOptions {
        refs: args.refs.clone(),
        range: args.range.clone(),
        all_refs: args.all_refs,
        first_parent: args.first_parent,
    }
}

fn index_options(config: &IndexConfig) -> IndexOptions {
    IndexOptions {
        exclude_paths: config.exclude_paths.clone(),
        max_blob_bytes: config.max_blob_bytes,
        max_patch_bytes: config.max_patch_bytes,
        max_document_bytes: config.max_document_bytes,
    }
}

fn repository_view(repo: &Repository) -> Value {
    json!({"id":repo.identity,"work_dir":repo.work_dir,"common_dir":repo.common_dir,"remote_freshness":"unchecked"})
}

fn scope_view(scope: &ScopeSnapshot) -> Result<Value> {
    let mut view = serde_json::to_value(scope)?;
    if let Some(object) = view.as_object_mut() {
        object.remove("oids");
        object.insert("commits".into(), json!(scope.oids.len()));
    }
    Ok(view)
}

fn coverage_views(coverage: &Coverage, semantic_available: bool) -> (Value, Value) {
    (
        json!({"total":coverage.total_commits,"indexed":coverage.indexed_commits,"complete":coverage.indexed_commits == coverage.total_commits,"omissions":coverage.omissions}),
        json!({"total":coverage.documents,"indexed":coverage.embedded_documents,"complete":semantic_available && coverage.embedded_documents == coverage.documents}),
    )
}

fn snapshot(generation: &Generation) -> Result<ModelSnapshot> {
    serde_json::from_value(generation.model.clone()).map_err(|e| {
        AppError::new(
            "invalid_index",
            format!("Read generation {} model snapshot: {e}", generation.id),
        )
    })
}

fn generation_view(generation: Option<&Generation>) -> Result<Value> {
    let Some(generation) = generation else {
        return Ok(Value::Null);
    };
    let saved = snapshot(generation)?;
    Ok(
        json!({"id":generation.id,"fingerprint":generation.fingerprint,"state":generation.state,"model":saved.resolved.profile.id,"revision":saved.resolved.profile.revision,"artifact":saved.resolved.profile.artifact,"dimensions":saved.resolved.profile.dimensions,"backend":saved.resolved.profile.backend,"query_prefix":saved.resolved.profile.query_prefix,"document_prefix":saved.resolved.profile.document_prefix}),
    )
}

fn selected_profile(
    args: &ModelArgs,
    config: &Config,
    desired: Option<&Generation>,
    apply_config: bool,
) -> Result<(Option<ModelProfile>, Option<String>, Option<ModelProfile>)> {
    let saved = desired.map(snapshot).transpose()?;
    // A persisted selection outlives its CLI invocation. Only a newly changed
    // configuration default supersedes it; unchanged defaults cannot undo a
    // completed switch or an interrupted staging generation on the next sync.
    let changed_default = apply_config
        && config.default_model.is_some()
        && saved
            .as_ref()
            .is_none_or(|s| s.config_default_model != config.default_model);
    let selector = args
        .model
        .clone()
        .or_else(|| {
            if changed_default {
                config.default_model.clone()
            } else {
                None
            }
        })
        .or_else(|| saved.as_ref().and_then(|s| s.selector.clone()));
    let mut profile = if args.model.is_some() || changed_default {
        profile_for_selector(selector.as_deref().expect("selected model"), config)?
    } else if apply_config
        && selector
            .as_deref()
            .is_some_and(|s| s.starts_with("profile:"))
    {
        profile_for_selector(selector.as_deref().expect("stored selector"), config)?
    } else if let Some(saved) = &saved {
        saved
            .requested_profile
            .clone()
            .unwrap_or_else(|| saved.resolved.profile.clone())
    } else {
        if args.revision.is_some() || args.variant.is_some() || args.reembed {
            return Err(AppError::new("model_required", "No active model to modify")
                .action("Pass --model HF_ID or --model profile:NAME"));
        }
        return Ok((None, selector, None));
    };
    if args.model.is_none()
        && let Some(saved) = &saved
        && selector == saved.selector
        && let Some(configured) = &saved.config_profile
        && let Some(previous) = &saved.requested_profile
    {
        // CLI artifact/revision overrides persist until those particular TOML
        // fields change. Editing a query prompt must not undo a pinned revision.
        if profile.revision == configured.revision {
            profile.revision = previous.revision.clone();
        }
        if profile.artifact == configured.artifact {
            profile.artifact = previous.artifact.clone();
        }
    }
    if let Some(revision) = &args.revision {
        profile.revision = revision.clone();
    }
    if let Some(artifact) = &args.variant {
        profile.artifact = artifact.clone();
    }
    profile.validate()?;
    if apply_config && config.index.chunk_overlap_tokens >= profile.max_tokens {
        return Err(AppError::new(
            "invalid_config",
            "index.chunk_overlap_tokens must be smaller than the model's token budget",
        ));
    }
    let requested = profile.clone();
    if args.model.is_none()
        && args.revision.is_none()
        && let Some(saved) = &saved
    {
        let previous = saved
            .requested_profile
            .as_ref()
            .unwrap_or(&saved.resolved.profile);
        if previous.id == profile.id && previous.revision == profile.revision {
            // Ordinary sync never follows a moving Hub ref, even when a profile's
            // query prompt or operational settings have changed in configuration.
            profile.revision = saved.resolved.profile.revision.clone();
        }
    }
    Ok((Some(profile), selector, Some(requested)))
}

fn profile_for_selector(selector: &str, config: &Config) -> Result<ModelProfile> {
    if let Some(name) = selector.strip_prefix("profile:") {
        return config.profiles.get(name).cloned().ok_or_else(|| {
            AppError::new(
                "invalid_config",
                format!("Model profile '{name}' is not defined in the selected configuration"),
            )
        });
    }
    embedding::profile(selector)
}

fn document_fingerprint(resolved: &ResolvedModel, overlap: usize) -> String {
    let mut hash = Sha256::new();
    hash.update(b"nalcos-document-format-v1\0");
    hash.update(resolved.fingerprint.as_bytes());
    hash.update(overlap.to_le_bytes());
    format!("{:x}", hash.finalize())
}

fn same_document_profile(a: &Option<ModelProfile>, b: &Option<ModelProfile>) -> bool {
    let canonical = |profile: &Option<ModelProfile>| {
        profile.as_ref().map(|profile| {
            let mut profile = profile.clone();
            profile.query_prefix.clear();
            profile.device = embedding::Device::Auto;
            profile.api_key_env = None;
            profile
        })
    };
    canonical(a) == canonical(b)
}

struct WriterLock(File);

impl WriterLock {
    fn acquire(
        index_path: &Path,
        wait: bool,
        verbose: bool,
        execution: &Execution,
    ) -> Result<Self> {
        let directory = index_path
            .parent()
            .ok_or_else(|| AppError::invalid("Index path has no parent"))?;
        std::fs::create_dir_all(directory)?;
        let path = directory.join("writer.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;
        let mut reported_contention = false;
        loop {
            execution.check()?;
            match fs2::FileExt::try_lock_exclusive(&file) {
                Ok(()) => return Ok(Self(file)),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if !wait {
                        return Err(AppError::new(
                            "index_busy",
                            "Another process is indexing this repository",
                        ));
                    }
                    if verbose && !reported_contention {
                        eprintln!("Waiting for index writer lock: {}", path.display());
                        reported_contention = true;
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(error) => {
                    return Err(AppError::new(
                        "index_lock_failed",
                        format!("Lock {}: {error}", path.display()),
                    ));
                }
            }
        }
    }
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        // Closing the file releases the OS lock even when an explicit unlock fails.
        if let Err(error) = fs2::FileExt::unlock(&self.0) {
            eprintln!("warning: release index writer lock: {error}");
        }
    }
}

fn ingest(
    repo: &Repository,
    store: &mut Store,
    oids: &[String],
    policy: &IndexConfig,
    rescan: bool,
    execution: &Execution,
) -> Result<usize> {
    let indexed = if rescan {
        HashSet::new()
    } else {
        store.indexed_oids()?
    };
    store.set_scope(oids)?;
    let retry: HashSet<_> = store
        .scoped_omissions()?
        .into_iter()
        .filter(|omission| {
            matches!(
                omission.reason.as_str(),
                "missing_object" | "missing_parent"
            )
        })
        .map(|omission| omission.commit_oid)
        .collect();
    let pending: Vec<_> = oids
        .iter()
        .filter(|oid| !indexed.contains(*oid) || retry.contains(*oid))
        .cloned()
        .collect();
    let options = index_options(policy);
    let mut completed = 0;
    let mut progress = Instant::now();
    for ids in pending.chunks(32) {
        execution.check()?;
        let commits = repo.load_commits(ids, execution)?;
        let extraction = repo.extract_documents(&commits, &options, execution)?;
        execution.check()?;
        store.ingest(&commits, &extraction.documents, &extraction.omissions)?;
        completed += commits.len();
        if progress.elapsed() >= Duration::from_secs(2) {
            eprintln!("Indexing history: {completed}/{} commits", pending.len());
            progress = Instant::now();
        }
    }
    Ok(completed)
}

fn saved_index_policy(store: &Store, fallback: &IndexConfig) -> Result<IndexConfig> {
    store
        .metadata(INDEX_POLICY_KEY)?
        .map(|value| serde_json::from_str(&value).map_err(AppError::from))
        .transpose()
        .map(|value| value.unwrap_or_else(|| fallback.clone()))
}

fn source_refresh_required(store: &Store) -> Result<bool> {
    Ok(store.metadata(EXTRACTION_VERSION_KEY)?.as_deref()
        != Some(EXTRACTION_VERSION.to_string().as_str()))
}

fn fill_embeddings(
    store: &mut Store,
    generation: &Generation,
    encoder: &mut Encoder,
    overlap: usize,
    scoped: bool,
    execution: &Execution,
) -> Result<usize> {
    let mut completed = 0;
    let mut progress = Instant::now();
    let max_tokens = snapshot(generation)?.resolved.profile.max_tokens;
    if overlap >= max_tokens {
        return Err(AppError::new(
            "invalid_config",
            "index.chunk_overlap_tokens must be smaller than the model's token budget",
        ));
    }
    loop {
        execution.check()?;
        let pending = if scoped {
            store.pending_documents_in_scope(generation.id, 16, execution)?
        } else {
            store.pending_documents(generation.id, 16, execution)?
        };
        if pending.is_empty() {
            break;
        }
        let mut documents = Vec::new();
        let mut texts = Vec::new();
        for document in pending {
            execution.check()?;
            if store.reuse_document(generation.id, &document.id)? {
                completed += 1;
                continue;
            }
            let chunks = encoder.chunk_documents(
                &document.text,
                ChunkOptions {
                    max_tokens,
                    overlap_tokens: overlap,
                },
            )?;
            texts.extend(chunks.iter().map(|chunk| chunk.text.clone()));
            documents.push((document.id, chunks));
        }
        let vectors = if texts.is_empty() {
            vec![]
        } else {
            encoder.encode(&texts, InputKind::Document, execution)?
        };
        if vectors.len() != texts.len() {
            return Err(AppError::new(
                "embedding_count_mismatch",
                "Encoder did not return one vector per input chunk",
            ));
        }
        let mut vectors = vectors.into_iter();
        for (id, chunks) in documents {
            let chunks: Vec<_> = chunks
                .into_iter()
                .map(|chunk| VectorChunk {
                    byte_start: chunk.byte_start,
                    byte_end: chunk.byte_end,
                    vector: vectors.next().expect("checked embedding count"),
                })
                .collect();
            execution.check()?;
            store.complete_document(generation.id, &id, &chunks)?;
            completed += 1;
        }
        if progress.elapsed() >= Duration::from_secs(2) {
            eprintln!("Indexing embeddings: {completed} documents completed");
            progress = Instant::now();
        }
    }
    Ok(completed)
}

fn check_provider_probe(
    saved: &ModelSnapshot,
    encoder: &mut Encoder,
    execution: &Execution,
) -> Result<Option<Vec<f32>>> {
    if !matches!(
        saved.resolved.profile.backend,
        Backend::OpenAi | Backend::Ollama
    ) {
        return Ok(None);
    }
    let vectors = encoder.encode(&[PROVIDER_PROBE.to_owned()], InputKind::Document, execution)?;
    let vector = vectors.into_iter().next().ok_or_else(|| {
        AppError::new(
            "embedding_count_mismatch",
            "Provider returned no identity probe vector",
        )
    })?;
    if let Some(reference) = &saved.provider_probe {
        let similarity: f32 = reference.iter().zip(&vector).map(|(a, b)| a * b).sum();
        if reference.len() != vector.len() || !similarity.is_finite() || similarity < 0.999 {
            return Err(AppError::new(
                "provider_changed",
                "The embedding provider no longer matches the active index's identity probe",
            )
            .action("Pin the provider deployment and run nalcos sync --reembed"));
        }
    }
    Ok(Some(vector))
}

#[allow(clippy::too_many_arguments)]
fn sync(
    cli: &Cli,
    config: &Config,
    repo: &Repository,
    path: &Path,
    scope_args: &ScopeArgs,
    model_args: &ModelArgs,
    dry_run: bool,
    initialization: bool,
    apply_config: bool,
    execution: &Execution,
) -> Result<Value> {
    let scope_options = scope_options(scope_args);
    let mut scope = repo.resolve_scope(&scope_options, execution)?;
    // Resolve writer intent under the lock so a queued sync cannot restore a
    // generation that another writer replaced while this process was waiting.
    let _lock = if dry_run {
        None
    } else {
        Some(WriterLock::acquire(path, true, cli.verbose > 0, execution)?)
    };
    // Inspect the saved selection without creating an index for invalid setup
    // requests. Keep the writer lock through validation and writable opening.
    let opened_store = if path.exists() {
        Some(Store::read_only(path)?)
    } else {
        None
    };
    let active = opened_store
        .as_ref()
        .map(Store::active_generation)
        .transpose()?
        .flatten();
    let staging = opened_store
        .as_ref()
        .map(Store::staging_generation)
        .transpose()?
        .flatten();
    let (profile, selector, requested_profile) = selected_profile(
        model_args,
        config,
        staging.as_ref().or(active.as_ref()),
        apply_config,
    )?;
    if initialization && profile.is_none() {
        return Err(AppError::new("model_required", "No default embedding model has passed release qualification").action("Run nalcos init --model HF_ID, configure a named profile, or use search --mode lexical"));
    }
    if dry_run {
        let source_refresh_required = opened_store
            .as_ref()
            .map(source_refresh_required)
            .transpose()?
            .unwrap_or(false);
        let warnings: Vec<_> = source_refresh_required
            .then_some(SOURCE_REFRESH_WARNING)
            .into_iter()
            .collect();
        let indexed = opened_store
            .as_ref()
            .map(Store::indexed_oids)
            .transpose()?
            .unwrap_or_default();
        let missing = scope
            .oids
            .iter()
            .filter(|oid| !indexed.contains(*oid))
            .count();
        let old_snapshot = active.as_ref().map(snapshot).transpose()?;
        let overlap_changed = old_snapshot
            .as_ref()
            .is_some_and(|s| s.chunk_overlap_tokens != config.index.chunk_overlap_tokens);
        let old_profile = old_snapshot.map(|s| s.resolved.profile);
        return Ok(
            json!({"schema_version":1,"command":if initialization {"init"} else {"sync"},"dry_run":true,"repository":repository_view(repo),"scope":scope_view(&scope)?,"commits_pending":missing,"source_refresh_required":source_refresh_required,"model":profile.as_ref().map(|p| &p.id),"model_resolution_required":profile.is_some(),"full_reembedding":model_args.reembed || overlap_changed || !same_document_profile(&profile, &old_profile),"downloads":"Only missing model/runtime assets would be downloaded; no network requests performed","writes_performed":false,"warnings":warnings}),
        );
    }
    // Readers use WAL snapshots while this writer resolves and prepares its encoder.
    let resolved = if !apply_config {
        // Watch iterations keep the pinned, already resolved contract. Loading an
        // encoder validates its assets if work arrives; idle polls need not hash
        // hundreds of megabytes of unchanged weights every two seconds.
        staging
            .as_ref()
            .or(active.as_ref())
            .map(snapshot)
            .transpose()?
            .map(|s| s.resolved)
    } else {
        profile
            .map(|profile| {
                embedding::resolve(
                    &ModelRequest {
                        model: None,
                        profile: Some(profile),
                    },
                    ResolveOptions {
                        allow_download: apply_config,
                        offline: cli.offline,
                    },
                    execution,
                )
            })
            .transpose()?
    };
    if apply_config && let Some(resolved) = &resolved {
        embedding::prepare_runtime(
            resolved,
            &config.runtime_options(true, cli.device.as_deref(), apply_config)?,
            ResolveOptions {
                allow_download: apply_config,
                offline: cli.offline,
            },
            execution,
        )?;
    }
    drop(opened_store);
    let mut store = Store::open(path)?;
    scope = repo.resolve_scope(&scope_options, execution)?;
    let previous_policy = saved_index_policy(&store, &config.index)?;
    let policy = if apply_config {
        config.index.clone()
    } else {
        previous_policy.clone()
    };
    let rescan = source_refresh_required(&store)? || (apply_config && previous_policy != policy);
    let retained: HashSet<String> = repo
        .resolve_scope(
            &ScopeOptions {
                all_refs: true,
                ..ScopeOptions::default()
            },
            execution,
        )?
        .oids
        .into_iter()
        .chain(scope.oids.iter().cloned())
        .collect();
    let ingest_oids = if rescan {
        let mut all: HashSet<_> = store.indexed_oids()?;
        all.retain(|oid| retained.contains(oid));
        all.extend(scope.oids.iter().cloned());
        let mut all: Vec<_> = all.into_iter().collect();
        all.sort();
        all
    } else {
        scope.oids.clone()
    };
    let commits_added = ingest(repo, &mut store, &ingest_oids, &policy, rescan, execution)?;
    // Copy identical content before discarding unreachable associations, preserving
    // vector reuse when a rebase changes object IDs but not formatted source text.
    if let Some(active) = store.active_generation()? {
        store.reuse_pending_documents(active.id, execution)?;
    }
    let commits_pruned = store.prune_commits_except(&retained)?;
    execution.check()?;
    store.set_metadata(INDEX_POLICY_KEY, &serde_json::to_string(&policy)?)?;
    // Ingestion commits in resumable batches. Publish the extraction version only
    // after every retained source has been refreshed, so interruption retries it.
    store.set_metadata(EXTRACTION_VERSION_KEY, &EXTRACTION_VERSION.to_string())?;
    store.set_scope(&scope.oids)?;
    if apply_config {
        store.optimize(execution)?;
    }
    let mut documents_embedded = 0;
    let mut runtime = Value::Null;
    if let Some(resolved) = resolved {
        let fingerprint = document_fingerprint(&resolved, policy.chunk_overlap_tokens);
        let mut saved = ModelSnapshot {
            resolved,
            config_profile: selector
                .as_deref()
                .and_then(|s| s.strip_prefix("profile:"))
                .and_then(|s| config.profiles.get(s))
                .cloned(),
            selector,
            requested_profile,
            config_default_model: config.default_model.clone(),
            chunk_overlap_tokens: policy.chunk_overlap_tokens,
            provider_probe: None,
            runtime: None,
            runtime_policy: None,
        };
        let old = store.active_generation()?;
        let generation = store.ensure_generation(
            &fingerprint,
            &serde_json::to_value(&saved)?,
            model_args.reembed,
        )?;
        let previous = snapshot(&generation)?;
        saved.provider_probe = previous.provider_probe;
        saved.runtime = previous.runtime;
        saved.runtime_policy = previous.runtime_policy;
        // Runtime setup is independent of document identity. Apply changed device
        // and workload settings even when every document already has a vector.
        let runtime_policy = json!({
            "calibration_policy": embedding::CALIBRATION_POLICY,
            "config": config.runtime,
            "profile_device": saved.resolved.profile.device,
            "device_override": cli.device,
            "ort_library_env": std::env::var_os("NALCOS_ORT_LIBRARY"),
            "ort_dylib_env": std::env::var_os("ORT_DYLIB_PATH"),
        });
        let runtime_changed =
            apply_config && saved.runtime_policy.as_ref() != Some(&runtime_policy);
        let pending = !store
            .pending_documents(generation.id, 1, execution)?
            .is_empty();
        let query_changed = old.as_ref().map(snapshot).transpose()?.is_some_and(|s| {
            s.resolved.profile.query_prefix != saved.resolved.profile.query_prefix
        });
        if pending
            || initialization
            || generation.state == "staging"
            || query_changed
            || runtime_changed
        {
            check_offline(&saved.resolved, cli.offline)?;
            let mut encoder = Encoder::load(
                &saved.resolved,
                config.runtime_options(true, cli.device.as_deref(), apply_config)?,
                execution,
            )?;
            saved.provider_probe = check_provider_probe(&saved, &mut encoder, execution)?;
            // Save the identity probe before committing any vectors so interrupted remote
            // migrations cannot resume against a changed provider under the same alias.
            if generation.state == "staging" {
                store.update_generation_model(generation.id, &serde_json::to_value(&saved)?)?;
            }
            documents_embedded = fill_embeddings(
                &mut store,
                &generation,
                &mut encoder,
                policy.chunk_overlap_tokens,
                false,
                execution,
            )?;
            runtime = serde_json::to_value(encoder.info())?;
            saved.runtime = Some(runtime.clone());
            drop(encoder);
            if apply_config {
                let mut query_encoder = Encoder::load(
                    &saved.resolved,
                    config.runtime_options(false, cli.device.as_deref(), true)?,
                    execution,
                )?;
                query_encoder.encode(
                    &["Find a previous code change".to_owned()],
                    InputKind::Query,
                    execution,
                )?;
                runtime = json!({"index":saved.runtime,"query":query_encoder.info()});
                saved.runtime_policy = Some(runtime_policy);
            }
        } else if let Some(info) = &saved.runtime {
            runtime = info.clone();
        }
        store.update_generation_model(generation.id, &serde_json::to_value(&saved)?)?;
        if generation.state != "active" {
            store.activate_generation(generation.id)?;
        }
        // A resumed staging watermark may precede commits ingested in this invocation.
        // After cutover, finish those arrivals using the same pinned encoder contract.
        if !store
            .pending_documents(generation.id, 1, execution)?
            .is_empty()
        {
            let active = store
                .active_generation()?
                .ok_or_else(semantic_unavailable)?;
            let mut encoder = Encoder::load(
                &saved.resolved,
                config.runtime_options(true, cli.device.as_deref(), false)?,
                execution,
            )?;
            check_provider_probe(&saved, &mut encoder, execution)?;
            documents_embedded += fill_embeddings(
                &mut store,
                &active,
                &mut encoder,
                policy.chunk_overlap_tokens,
                false,
                execution,
            )?;
        }
        store.cleanup_retired()?;
    }
    if apply_config && documents_embedded > 0 {
        store.optimize(execution)?;
    }
    let active = store.active_generation()?;
    let coverage = store.scope_coverage()?;
    let (history, embedding) = coverage_views(&coverage, active.is_some());
    let mut warnings = Vec::new();
    if !repo.validate_snapshot(&scope_options, &scope, execution)? {
        warnings.push(
            "Repository references changed during indexing; run sync again to cover the new scope"
                .to_owned(),
        );
    }
    if coverage.omissions > 0 {
        warnings.push(format!(
            "{} source omissions are recorded for this scope",
            coverage.omissions
        ));
    }
    if scope.shallow {
        warnings.push(
            "Repository is shallow; coverage beyond the local history boundary is unknown"
                .to_owned(),
        );
    }
    Ok(
        json!({"schema_version":1,"command":if initialization {"init"} else {"sync"},"dry_run":false,"repository":repository_view(repo),"scope":scope_view(&scope)?,"commits_added":commits_added,"commits_pruned":commits_pruned,"documents_embedded":documents_embedded,"source_refresh_required":false,"history_coverage":history,"embedding_coverage":embedding,"active_generation":generation_view(active.as_ref())?,"staging_generation":generation_view(store.staging_generation()?.as_ref())?,"runtime":runtime,"warnings":warnings}),
    )
}

fn check_offline(resolved: &ResolvedModel, offline: bool) -> Result<()> {
    if offline && matches!(resolved.profile.backend, Backend::OpenAi | Backend::Ollama) {
        return Err(AppError::new(
            "offline",
            "Embedding provider requests are disabled by --offline",
        ));
    }
    Ok(())
}

fn filters(args: &SearchArgs) -> Result<FilterOptions> {
    let since = args
        .since
        .as_deref()
        .map(|s| parse_date(s, false))
        .transpose()?;
    let until = args
        .until
        .as_deref()
        .map(|s| parse_date(s, true))
        .transpose()?;
    if since.zip(until).is_some_and(|(a, b)| a >= b) {
        return Err(AppError::invalid("--since must not be later than --until"));
    }
    Ok(FilterOptions {
        paths: args.paths.clone(),
        author: args.author.clone(),
        since,
        until,
    })
}

fn parse_date(value: &str, upper: bool) -> Result<i64> {
    if let Ok(date) = chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        let start = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| AppError::invalid("Invalid calendar date"))?
            .and_utc()
            .timestamp();
        return start
            .checked_add(if upper { 86_400 } else { 0 })
            .ok_or_else(|| AppError::invalid("Date is out of range"));
    }
    let value = chrono::DateTime::parse_from_rfc3339(value)
        .map_err(|_| {
            AppError::invalid(format!(
                "Invalid date '{value}'; use YYYY-MM-DD (UTC) or an RFC3339 timestamp"
            ))
        })?
        .timestamp();
    value
        .checked_add(i64::from(upper))
        .ok_or_else(|| AppError::invalid("Date is out of range"))
}

#[allow(clippy::too_many_arguments)]
fn refresh_search(
    cli: &Cli,
    config: &Config,
    repo: &Repository,
    path: &Path,
    eligible: &[String],
    mode: SearchMode,
    wait: bool,
    execution: &Execution,
) -> Result<()> {
    let _lock = WriterLock::acquire(path, wait, cli.verbose > 0, execution)?;
    let mut store = Store::open(path)?;
    let policy = saved_index_policy(&store, &config.index)?;
    // An empty index has no legacy sources. Existing sources are refreshed only
    // by explicit setup, never by the bounded incremental search path.
    if source_refresh_required(&store)? && store.indexed_oids()?.is_empty() {
        execution.check()?;
        store.set_metadata(EXTRACTION_VERSION_KEY, &EXTRACTION_VERSION.to_string())?;
    }
    if store.metadata(INDEX_POLICY_KEY)?.is_none() {
        store.set_metadata(INDEX_POLICY_KEY, &serde_json::to_string(&policy)?)?;
    }
    ingest(repo, &mut store, eligible, &policy, false, execution)?;
    if mode != SearchMode::Lexical
        && let Some(generation) = store.active_generation()?
        && !store
            .pending_documents_in_scope(generation.id, 1, execution)?
            .is_empty()
    {
        let saved = snapshot(&generation)?;
        check_offline(&saved.resolved, cli.offline)?;
        let mut encoder = Encoder::load(
            &saved.resolved,
            config.runtime_options(true, cli.device.as_deref(), false)?,
            execution,
        )?;
        check_provider_probe(&saved, &mut encoder, execution)?;
        fill_embeddings(
            &mut store,
            &generation,
            &mut encoder,
            saved.chunk_overlap_tokens,
            true,
            execution,
        )?;
    }
    Ok(())
}

fn fallback_allowed(error: &AppError) -> bool {
    !matches!(
        error.code.as_str(),
        "timeout"
            | "interrupted"
            | "invalid_input"
            | "invalid_config"
            | "invalid_index"
            | "index_error"
            | "explicit_device_unavailable"
    )
}

fn search(
    cli: &Cli,
    config: &Config,
    repo: &Repository,
    path: &Path,
    args: &SearchArgs,
    execution: &Execution,
) -> Result<Value> {
    if args.query.trim().is_empty() {
        return Err(AppError::invalid("Search query must not be empty"));
    }
    let filter_options = filters(args)?;
    let scope_options = scope_options(&args.scope);
    for attempt in 0..2 {
        execution.check()?;
        let mut checkpoint = Instant::now();
        let mut trace = |stage: &str| {
            if cli.verbose > 1 {
                eprintln!(
                    "Search attempt {} {stage}: {:.3} ms",
                    attempt + 1,
                    checkpoint.elapsed().as_secs_f64() * 1000.0
                );
            }
            checkpoint = Instant::now();
        };
        let scope = repo.resolve_scope(&scope_options, execution)?;
        trace("Git scope");
        let eligible = repo.filter_scope(&scope, &filter_options, execution)?;
        trace("history filters");
        let mut warnings = Vec::new();
        if scope.shallow {
            warnings.push(
                "Repository is shallow; coverage beyond the local history boundary is unknown"
                    .to_owned(),
            );
        }
        if args.freshness != Freshness::Cached {
            let indexing_execution = if args.freshness == Freshness::Auto {
                execution.bounded(Duration::from_secs(2))
            } else {
                execution.clone()
            };
            if let Err(error) = refresh_search(
                cli,
                config,
                repo,
                path,
                &eligible,
                args.mode,
                args.freshness == Freshness::Wait,
                &indexing_execution,
            ) {
                execution.check()?;
                if args.freshness == Freshness::Auto
                    && matches!(error.code.as_str(), "timeout" | "index_busy")
                {
                    warnings.push("Incremental indexing is incomplete; use --freshness wait or nalcos sync to finish".to_owned());
                } else if args.mode == SearchMode::Hybrid && fallback_allowed(&error) {
                    warnings.push(format!("Semantic indexing unavailable: {error}"));
                } else {
                    return Err(error);
                }
            }
        }
        if !path.exists() {
            if args.mode == SearchMode::Semantic {
                return Err(semantic_unavailable());
            }
            if args.mode == SearchMode::Hybrid {
                warnings.push("No active embedding generation; using lexical search".to_owned());
            }
            warnings
                .push("No index is available; run search with --freshness auto or wait".to_owned());
            let coverage = Coverage {
                total_commits: eligible.len(),
                ..Coverage::default()
            };
            let (history, embedding) = coverage_views(&coverage, false);
            if !repo.validate_snapshot(&scope_options, &scope, execution)? {
                if attempt == 0 {
                    continue;
                } else {
                    return Err(repository_changed());
                }
            }
            return Ok(
                json!({"schema_version":1,"command":"search","repository":repository_view(repo),"scope":scope_view(&scope)?,"mode_requested":args.mode,"mode_used":"lexical","source_refresh_required":false,"history_coverage":history,"embedding_coverage":embedding,"results":[],"active_generation":null,"runtime":null,"output_truncated":false,"warnings":warnings}),
            );
        }
        let mut store = Store::read_only(path)?;
        trace("refresh and index open");
        store.set_scope(&eligible)?;
        store.begin_snapshot()?;
        trace("index scope");
        let source_refresh_required = source_refresh_required(&store)?;
        if source_refresh_required {
            warnings.push(SOURCE_REFRESH_WARNING.to_owned());
        }
        let active = store.active_generation()?;
        let mut runtime = Value::Null;
        let mut used = args.mode;
        let query_vector = if args.mode == SearchMode::Lexical {
            None
        } else {
            let encoded = (|| -> Result<Vec<f32>> {
                let generation = active.as_ref().ok_or_else(semantic_unavailable)?;
                let saved = snapshot(generation)?;
                check_offline(&saved.resolved, cli.offline)?;
                let mut encoder = Encoder::load(
                    &saved.resolved,
                    config.runtime_options(false, cli.device.as_deref(), false)?,
                    execution,
                )?;
                check_provider_probe(&saved, &mut encoder, execution)?;
                let vectors = encoder.encode(
                    std::slice::from_ref(&args.query),
                    InputKind::Query,
                    execution,
                )?;
                runtime = serde_json::to_value(encoder.info())?;
                vectors.into_iter().next().ok_or_else(|| {
                    AppError::new(
                        "embedding_count_mismatch",
                        "Encoder returned no query vector",
                    )
                })
            })();
            match encoded {
                Ok(vector) => Some(vector),
                Err(error) if args.mode == SearchMode::Hybrid && fallback_allowed(&error) => {
                    warnings.push(format!(
                        "Using lexical search because semantic search is unavailable: {error}"
                    ));
                    used = SearchMode::Lexical;
                    None
                }
                Err(error) => return Err(error),
            }
        };
        trace("query encoder");
        let mut candidate_limit = config.search.candidate_limit.max(args.limit as usize);
        let mut missing_sources = HashSet::new();
        let mut candidate_filter = CandidateFilter::default();
        if !args.paths.is_empty() {
            candidate_filter.allowed_documents = Some(repo.matching_document_ids(
                &store.document_paths_in_scope(execution)?,
                &args.paths,
                execution,
            )?);
        }
        let ranked = loop {
            execution.check()?;
            let lexical = if used != SearchMode::Semantic {
                store.lexical_candidates(
                    &args.query,
                    candidate_limit,
                    &candidate_filter,
                    execution,
                )?
            } else {
                vec![]
            };
            trace("lexical retrieval");
            let semantic = if let Some(vector) = &query_vector {
                store.semantic_candidates(
                    vector,
                    active.as_ref().expect("query generation").id,
                    candidate_limit,
                    &candidate_filter,
                    execution,
                )?
            } else {
                vec![]
            };
            trace("semantic retrieval");
            let channels_exhausted =
                lexical.len() < candidate_limit && semantic.len() < candidate_limit;
            let documents: Vec<_> = lexical
                .iter()
                .chain(semantic.iter())
                .map(|candidate| candidate.document.clone())
                .collect();
            let verification = repo.validate_documents(&documents, execution)?;
            trace("source verification");
            if !verification.missing_ids.is_empty() {
                missing_sources.extend(verification.missing_ids.iter().cloned());
                candidate_filter
                    .excluded_documents
                    .extend(verification.missing_ids);
                // Refill before commit aggregation/fusion. Another source from the
                // same commit can remain valid after a parent or blob disappears.
                continue;
            }
            let filtered = rank_candidates(lexical, semantic, used, config);
            if filtered.len() >= args.limit as usize
                || channels_exhausted
                || candidate_limit >= eligible.len().max(1)
            {
                break filtered;
            }
            candidate_limit = candidate_limit.saturating_mul(2).min(eligible.len().max(1));
        };
        if !missing_sources.is_empty() {
            warnings.push(format!(
                "Omitted {} evidence records whose Git source could not be verified",
                missing_sources.len()
            ));
        }
        let coverage = store.scope_coverage()?;
        trace("ranking and coverage");
        let (history, embedding) = coverage_views(&coverage, active.is_some());
        if coverage.indexed_commits < coverage.total_commits {
            warnings.push("History coverage is partial".to_owned());
        }
        if used != SearchMode::Lexical && coverage.embedded_documents < coverage.documents {
            warnings.push("Embedding coverage is partial".to_owned());
        }
        if coverage.omissions > 0 {
            warnings.push(format!(
                "{} source omissions are recorded for this scope",
                coverage.omissions
            ));
        }
        let mut budget = args.max_bytes;
        let mut truncated = false;
        let mut results = Vec::new();
        for hit in ranked.into_iter().take(args.limit as usize) {
            let commit = store.get_commit(&hit.oid)?.ok_or_else(|| {
                AppError::new(
                    "invalid_index",
                    format!("Missing commit metadata for {}", hit.oid),
                )
            })?;
            let evidence: Vec<_> = hit
                .evidence
                .into_iter()
                .take(2)
                .map(|candidate| {
                    evidence_view(
                        &candidate.document,
                        Some((candidate.byte_start, candidate.byte_end)),
                        &mut budget,
                        &mut truncated,
                    )
                })
                .collect::<Result<_>>()?;
            let mut commit = serde_json::to_value(commit)?;
            if let Some(message) = commit["message"].as_str() {
                let (message, clipped) = truncate_utf8(message, 4096);
                let message = message.to_owned();
                truncated |= clipped;
                commit["message"] = json!(message);
            }
            results.push(json!({"commit":commit,"score":hit.score,"evidence":evidence}));
        }
        let generation = generation_view(active.as_ref())?;
        store.end_snapshot()?;
        execution.check()?;
        trace("result formatting");
        let scope_unchanged = repo.validate_snapshot(&scope_options, &scope, execution)?;
        trace("scope recheck");
        if !scope_unchanged {
            if attempt == 0 {
                continue;
            } else {
                return Err(repository_changed());
            }
        }
        return Ok(
            json!({"schema_version":1,"command":"search","repository":repository_view(repo),"scope":scope_view(&scope)?,"filters":{"paths":args.paths,"author":args.author,"since":args.since,"until":args.until},"mode_requested":args.mode,"mode_used":used,"freshness":args.freshness,"source_refresh_required":source_refresh_required,"history_coverage":history,"embedding_coverage":embedding,"active_generation":generation,"runtime":runtime,"results":results,"output_truncated":truncated,"warnings":warnings}),
        );
    }
    Err(repository_changed())
}

fn semantic_unavailable() -> AppError {
    AppError::new(
        "semantic_unavailable",
        "No active embedding generation is available",
    )
    .action("Run nalcos init --model HF_ID or use --mode lexical")
}

fn repository_changed() -> AppError {
    AppError::new(
        "repository_changed",
        "Selected Git references changed repeatedly during search",
    )
    .action("Retry after repository updates finish")
}

struct RankedHit {
    oid: String,
    score: f64,
    evidence: Vec<Candidate>,
}

fn rank_candidates(
    lexical: Vec<Candidate>,
    semantic: Vec<Candidate>,
    mode: SearchMode,
    config: &Config,
) -> Vec<RankedHit> {
    let mut hits: HashMap<String, RankedHit> = HashMap::new();
    for (channel, weight) in [
        (lexical, config.search.lexical_weight),
        (semantic, config.search.semantic_weight),
    ] {
        for (position, candidate) in channel.into_iter().enumerate() {
            let oid = candidate.document.commit_oid.clone();
            let score = if mode == SearchMode::Hybrid {
                weight / (config.search.rrf_k + position as f64 + 1.0)
            } else {
                candidate.score
            };
            let hit = hits.entry(oid.clone()).or_insert_with(|| RankedHit {
                oid,
                score: 0.0,
                evidence: vec![],
            });
            hit.score += score;
            if !hit
                .evidence
                .iter()
                .any(|e| e.document.id == candidate.document.id)
            {
                hit.evidence.push(candidate);
            }
        }
    }
    let mut hits: Vec<_> = hits.into_values().collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.oid.cmp(&b.oid)));
    hits
}

fn evidence_view(
    document: &DocumentRecord,
    range: Option<(usize, usize)>,
    budget: &mut usize,
    truncated: &mut bool,
) -> Result<Value> {
    let (start, end) = range.unwrap_or((0, document.text.len()));
    let text = document.text.get(start..end).ok_or_else(|| {
        AppError::new(
            "invalid_index",
            format!("Invalid source range for evidence {}", document.id),
        )
    })?;
    let (excerpt, clipped) = truncate_utf8(text, *budget);
    *budget = budget.saturating_sub(excerpt.len());
    *truncated |= clipped;
    let mut value = serde_json::to_value(document)?;
    let object = value.as_object_mut().expect("document object");
    object.remove("text");
    object.insert("excerpt".into(), json!(excerpt));
    object.insert("excerpt_byte_start".into(), json!(start));
    object.insert("excerpt_byte_end".into(), json!(start + excerpt.len()));
    object.insert("excerpt_truncated".into(), json!(clipped));
    object.insert("verified".into(), json!(true));
    Ok(value)
}

fn show(repo: &Repository, path: &Path, args: &ShowArgs, execution: &Execution) -> Result<Value> {
    if let Some(id) = &args.evidence {
        if !path.exists() {
            return Err(AppError::new(
                "evidence_not_found",
                "No repository index contains that evidence identifier",
            ));
        }
        let store = Store::read_only(path)?;
        let document = store.get_document(id)?.ok_or_else(|| {
            AppError::new(
                "evidence_not_found",
                format!("Evidence '{id}' does not exist in this repository index"),
            )
        })?;
        let verification = repo.validate_documents(std::slice::from_ref(&document), execution)?;
        if !verification.valid_ids.contains(id) {
            return Err(AppError::new(
                "source_unavailable",
                format!("Git source for evidence '{id}' is no longer available"),
            ));
        }
        let commit = repo
            .load_commits(std::slice::from_ref(&document.commit_oid), execution)?
            .into_iter()
            .next()
            .ok_or_else(|| AppError::new("source_unavailable", "Commit source is unavailable"))?;
        let mut budget = if args.full {
            usize::MAX
        } else {
            args.max_bytes
        };
        let mut truncated = false;
        let evidence = evidence_view(&document, None, &mut budget, &mut truncated)?;
        return Ok(
            json!({"schema_version":1,"command":"show","repository":repository_view(repo),"commit":commit,"evidence":evidence,"output_truncated":truncated}),
        );
    }
    let options = ShowOptions {
        parent: args.parent as usize,
        paths: args.paths.clone(),
        context: args.context as usize,
        max_bytes: (!args.full).then_some(args.max_bytes),
    };
    let result = repo.show(
        args.commit.as_deref().expect("clap validates source"),
        &options,
        execution,
    )?;
    Ok(
        json!({"schema_version":1,"command":"show","repository":repository_view(repo),"commit":result.commit,"parent_oid":result.parent_oid,"patch":result.patch,"omissions":result.omissions,"output_truncated":result.truncated}),
    )
}

fn status(
    cli: &Cli,
    config: &Config,
    repo: &Repository,
    path: &Path,
    args: &ScopeArgs,
    check: bool,
    execution: &Execution,
) -> Result<Value> {
    let scope = repo.resolve_scope(&scope_options(args), execution)?;
    let (coverage, active, staging, counts, source_refresh_required) = if path.exists() {
        let mut store = Store::read_only(path)?;
        store.set_scope(&scope.oids)?;
        store.begin_snapshot()?;
        (
            store.scope_coverage()?,
            store.active_generation()?,
            store.staging_generation()?,
            serde_json::to_value(store.counts()?)?,
            source_refresh_required(&store)?,
        )
    } else {
        (
            Coverage {
                total_commits: scope.oids.len(),
                ..Coverage::default()
            },
            None,
            None,
            Value::Null,
            false,
        )
    };
    let (history, embedding) = coverage_views(&coverage, active.is_some());
    let readiness = if !path.exists() {
        "uninitialized"
    } else if source_refresh_required {
        "source_refresh_required"
    } else if active.is_none() {
        "lexical_only"
    } else if coverage.indexed_commits < coverage.total_commits
        || coverage.embedded_documents < coverage.documents
    {
        "partial"
    } else {
        "ready"
    };
    let mut probe = Value::Null;
    if check {
        let generation = active.as_ref().ok_or_else(semantic_unavailable)?;
        let saved = snapshot(generation)?;
        check_offline(&saved.resolved, cli.offline)?;
        let mut encoder = Encoder::load(
            &saved.resolved,
            config.runtime_options(false, cli.device.as_deref(), false)?,
            execution,
        )?;
        check_provider_probe(&saved, &mut encoder, execution)?;
        let vectors = encoder.encode(
            &["Find a previous code change".to_owned()],
            InputKind::Query,
            execution,
        )?;
        probe = json!({"passed":true,"dimensions":vectors.first().map(Vec::len),"runtime":encoder.info(),"synthetic_input":true});
    }
    let desired = selected_profile(
        &ModelArgs::default(),
        config,
        staging.as_ref().or(active.as_ref()),
        true,
    )?
    .2;
    let mut warnings = Vec::new();
    if source_refresh_required {
        warnings.push(SOURCE_REFRESH_WARNING.to_owned());
    }
    if coverage.omissions > 0 {
        warnings.push(format!(
            "{} source omissions are recorded for this scope",
            coverage.omissions
        ));
    }
    if scope.shallow {
        warnings.push(
            "Repository is shallow; coverage beyond the local history boundary is unknown"
                .to_owned(),
        );
    }
    Ok(
        json!({"schema_version":1,"command":"status","repository":repository_view(repo),"scope":scope_view(&scope)?,"readiness":readiness,"source_refresh_required":source_refresh_required,"index_path":path,"storage_bytes":if path.exists() {std::fs::metadata(path)?.len()} else {0},"counts":counts,"history_coverage":history,"embedding_coverage":embedding,"active_generation":generation_view(active.as_ref())?,"staging_generation":generation_view(staging.as_ref())?,"desired_profile":desired,"effective_config":config,"runtime":active.as_ref().map(snapshot).transpose()?.and_then(|s| s.runtime),"check":probe,"warnings":warnings,"release_qualification":"pending"}),
    )
}
