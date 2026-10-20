use super::profile::*;
use crate::error::{AppError, Result};
use crate::execution::Execution;
use fs2::FileExt;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Resolve/install is an explicit operation. Encoder loading never calls this function.
pub fn resolve(
    request: &ModelRequest,
    options: ResolveOptions,
    execution: &Execution,
) -> Result<ResolvedModel> {
    execution.check()?;
    if request.model.is_some() && request.profile.is_some() {
        return Err(AppError::invalid(
            "Choose either a model id or a custom model profile",
        ));
    }
    let mut selected = if let Some(value) = &request.profile {
        value.clone()
    } else if let Some(id) = &request.model {
        profile(id)?
    } else {
        return Err(AppError::new(
            "model_required",
            "No embedding model was supplied to the resolver; run nalcos init or select an explicit model profile",
        ));
    };
    selected.validate()?;
    let mut files = BTreeMap::new();
    let (artifact_path, tokenizer_path) = match selected.backend {
        Backend::Onnx | Backend::Gguf => {
            let cache = shared_cache()?;
            selected.revision = resolve_revision(&cache, &selected, options, execution)?;
            let mut filenames = vec![selected.artifact.clone()];
            if let Some(tokenizer) = &selected.tokenizer {
                filenames.push(tokenizer.clone());
            }
            filenames.extend(selected.extra_files.iter().cloned());
            filenames.sort();
            filenames.dedup();
            for filename in filenames {
                let path = resolve_file(&cache, &selected, &filename, options, execution)?;
                let resolved = inspect_file(&path, execution)?;
                files.insert(filename, resolved);
            }
            let artifact = files.get(&selected.artifact).map(|f| f.path.clone());
            let tokenizer = selected
                .tokenizer
                .as_ref()
                .and_then(|n| files.get(n))
                .map(|f| f.path.clone());
            (artifact, tokenizer)
        }
        Backend::OpenAi | Backend::Ollama => {
            if options.offline {
                return Err(AppError::new(
                    "offline_provider",
                    "Remote embedding providers are unavailable in offline mode",
                ));
            }
            let path = PathBuf::from(selected.tokenizer.as_deref().unwrap_or_default());
            if !path.is_absolute() {
                return Err(AppError::invalid(
                    "A provider tokenizer must be an absolute local tokenizer.json path",
                ));
            }
            files.insert("tokenizer.json".into(), inspect_file(&path, execution)?);
            (None, Some(path))
        }
    };
    let fingerprint = fingerprint(&selected, &files)?;
    Ok(ResolvedModel {
        profile: selected,
        fingerprint,
        artifact_path,
        tokenizer_path,
        files,
    })
}

pub(crate) fn fingerprint(
    profile: &ModelProfile,
    files: &BTreeMap<String, ResolvedFile>,
) -> Result<String> {
    let content: BTreeMap<_, _> = files
        .iter()
        .map(|(name, file)| (name, (&file.sha256, file.bytes)))
        .collect();
    let contract = if profile.backend == Backend::Gguf {
        "nalcos-embedding-contract-v2-l2-gguf-parse-special"
    } else {
        "nalcos-embedding-contract-v1-l2"
    };
    let semantic = serde_json::to_vec(&(contract, profile.semantic_profile(), content))?;
    Ok(format!("{:x}", Sha256::digest(semantic)))
}

pub(crate) fn validate_resolved(model: &ResolvedModel, execution: &Execution) -> Result<()> {
    model.profile.validate()?;
    if model.fingerprint != fingerprint(&model.profile, &model.files)? {
        return Err(AppError::new(
            "encoder_contract_changed",
            "Resolved embedding profile changed; run init or sync with the new profile to build a separate generation",
        ).action("Run nalcos sync to build and atomically activate embeddings under the current encoder contract"));
    }
    for file in model.files.values() {
        execution.check()?;
        let metadata = fs::metadata(&file.path).map_err(|e| AppError::new("model_not_cached", format!("Cached model file {} is unavailable: {e}; run init or sync explicitly to restore it", file.path.display())))?;
        if metadata.len() != file.bytes {
            return Err(AppError::new(
                "model_corrupt",
                format!(
                    "Cached model file {} changed size; resolve the profile again before using it",
                    file.path.display()
                ),
            ));
        }
        let (modified_ns, file_id) = metadata_identity(&metadata);
        // Tokenizers are cheap to hash and can be user-managed. Large HF blobs are immutable
        // by convention; any file replacement or timestamp change invalidates the fast path.
        if (file.bytes < 8 * 1024 * 1024
            || modified_ns.is_none()
            || modified_ns != file.modified_ns
            || file_id != file.file_id)
            && inspect_file(&file.path, execution)?.sha256 != file.sha256
        {
            return Err(AppError::new(
                "model_corrupt",
                format!(
                    "Cached file {} no longer matches its resolved SHA256; run init or sync to resolve a new profile",
                    file.path.display()
                ),
            ));
        }
    }
    // Paths are derived from the hashed content map rather than accepted independently.
    if matches!(model.profile.backend, Backend::Onnx | Backend::Gguf)
        && model.artifact_path.as_ref() != model.files.get(&model.profile.artifact).map(|f| &f.path)
    {
        return Err(AppError::new(
            "profile_changed",
            "Model artifact path does not match the resolved file map",
        ));
    }
    let tokenizer_key = if matches!(model.profile.backend, Backend::OpenAi | Backend::Ollama) {
        Some("tokenizer.json")
    } else {
        model.profile.tokenizer.as_deref()
    };
    if model.tokenizer_path.as_ref()
        != tokenizer_key
            .and_then(|key| model.files.get(key))
            .map(|f| &f.path)
    {
        return Err(AppError::new(
            "profile_changed",
            "Tokenizer path does not match the resolved file map",
        ));
    }
    Ok(())
}

pub(crate) fn shared_cache() -> Result<hf_hub::Cache> {
    if let Some(path) =
        std::env::var_os("HF_HUB_CACHE").or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
    {
        return Ok(hf_hub::Cache::new(PathBuf::from(path)));
    }
    Ok(home_cache())
}

fn home_cache() -> hf_hub::Cache {
    if std::env::var_os("HF_HOME").is_none()
        && let Some(path) = std::env::var_os("XDG_CACHE_HOME")
    {
        return hf_hub::Cache::new(PathBuf::from(path).join("huggingface").join("hub"));
    }
    hf_hub::Cache::from_env()
}

fn repository_root(cache: &hf_hub::Cache, id: &str) -> PathBuf {
    cache
        .path()
        .join(format!("models--{}", id.replace('/', "--")))
}

fn immutable_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

fn resolve_revision(
    cache: &hf_hub::Cache,
    profile: &ModelProfile,
    options: ResolveOptions,
    execution: &Execution,
) -> Result<String> {
    if immutable_revision(&profile.revision) {
        return Ok(profile.revision.clone());
    }
    validate_relative_path(&profile.revision)?;
    let reference = repository_root(cache, &profile.id)
        .join("refs")
        .join(&profile.revision);
    if options.offline || !options.allow_download {
        let value = fs::read_to_string(&reference).map_err(|_| {
            AppError::new(
                "model_not_cached",
                format!(
                    "Revision '{}' of {} has no cached commit; run init with downloads enabled",
                    profile.revision, profile.id
                ),
            )
        })?;
        let revision = value.trim();
        if !immutable_revision(revision) {
            return Err(AppError::new(
                "invalid_revision",
                "Cached Hugging Face reference is not an immutable commit",
            ));
        }
        return Ok(revision.into());
    }
    let mut url = hub_endpoint()?;
    url.path_segments_mut()
        .map_err(|_| AppError::invalid("Invalid Hugging Face endpoint"))?
        .extend(["api", "models"])
        .extend(profile.id.split('/'))
        .extend(["revision", &profile.revision]);
    let operation = format!("Resolve model revision {}@{}", profile.id, profile.revision);
    eprintln!("{operation}");
    let response = authenticated(client(execution)?.get(url))
        .send()
        .map_err(network_error)
        .map_err(|error| download_context(error, &operation))?;
    let response = checked_response(response, "read revision metadata")
        .map_err(|error| download_context(error, &operation))?;
    let metadata: serde_json::Value = response
        .json()
        .map_err(network_error)
        .map_err(|error| download_context(error, &operation))?;
    let revision = metadata
        .get("sha")
        .and_then(|v| v.as_str())
        .filter(|v| immutable_revision(v))
        .ok_or_else(|| {
            AppError::new(
                "invalid_revision",
                "Hugging Face returned no immutable model revision",
            )
        })?;
    execution.check()?;
    // hf-hub's cache layout is shared with Python; no second model directory is created.
    if let Some(parent) = reference.parent() {
        fs::create_dir_all(parent)?;
    }
    atomic_write(&reference, revision.as_bytes())?;
    Ok(revision.into())
}

fn resolve_file(
    cache: &hf_hub::Cache,
    profile: &ModelProfile,
    filename: &str,
    options: ResolveOptions,
    execution: &Execution,
) -> Result<PathBuf> {
    let repository = repository_root(cache, &profile.id);
    let snapshot = repository
        .join("snapshots")
        .join(&profile.revision)
        .join(filename);
    if snapshot.is_file() {
        return Ok(snapshot);
    }
    if options.offline || !options.allow_download {
        return Err(AppError::new(
            "model_not_cached",
            format!(
                "{}@{} is missing {filename}; run init or sync with downloads enabled",
                profile.id, profile.revision
            ),
        ));
    }
    let lock_path = cache
        .path()
        .join(".locks")
        .join(format!("models--{}", profile.id.replace('/', "--")))
        .join("nalcos-resolve.lock");
    let _lock = acquire_lock(&lock_path, execution)?;
    if snapshot.is_file() {
        return Ok(snapshot);
    }
    let mut url = hub_endpoint()?;
    url.path_segments_mut()
        .map_err(|_| AppError::invalid("Invalid Hugging Face endpoint"))?
        .extend(profile.id.split('/'))
        .extend(["resolve", &profile.revision])
        .extend(filename.split('/'));
    let metadata_client = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(request_timeout(execution))
        .build()
        .map_err(network_error)?;
    let artifact = format!("{}@{} / {filename}", profile.id, profile.revision);
    eprintln!("Checking model artifact: {artifact}");
    let metadata = authenticated(metadata_client.head(url.clone()))
        .send()
        .map_err(network_error)
        .map_err(|error| download_context(error, &format!("Read metadata for {artifact}")))?;
    if !metadata.status().is_success() && !metadata.status().is_redirection() {
        return Err(download_context(
            AppError::new(
                "model_download_failed",
                format!("Artifact metadata returned HTTP {}", metadata.status()),
            ),
            &artifact,
        ));
    }
    let etag = metadata
        .headers()
        .get("x-linked-etag")
        .or_else(|| metadata.headers().get("etag"))
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim_matches('"').to_owned())
        .filter(|v| matches!(v.len(), 40 | 64) && v.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| {
            download_context(
                AppError::new(
                    "model_download_failed",
                    "Hugging Face did not return a content-addressed artifact ETag",
                ),
                &artifact,
            )
        })?;
    let blob = repository.join("blobs").join(&etag);
    if !blob.is_file() {
        fs::create_dir_all(
            blob.parent()
                .ok_or_else(|| AppError::invalid("Invalid model cache path"))?,
        )?;
        download_artifact(
            &blob,
            &artifact,
            (etag.len() == 64).then_some(etag.as_str()),
            || Ok(authenticated(client(execution)?.get(url.clone()))),
            execution,
        )
        .map_err(|mut error| {
            if error.code == "download_checksum_mismatch" {
                error.code = "model_checksum_mismatch".into();
            } else if error.code == "download_failed" {
                error.code = "model_download_failed".into();
            }
            error
        })?;
    }
    fs::create_dir_all(
        snapshot
            .parent()
            .ok_or_else(|| AppError::invalid("Invalid snapshot path"))?,
    )?;
    #[cfg(unix)]
    {
        let temporary = temporary_path(&snapshot);
        std::os::unix::fs::symlink(&blob, &temporary)?;
        fs::rename(temporary, &snapshot)?;
    }
    #[cfg(not(unix))]
    fs::hard_link(&blob, &snapshot)?;
    Ok(snapshot)
}

pub(crate) fn inspect_file(path: &Path, execution: &Execution) -> Result<ResolvedFile> {
    let mut file = File::open(path).map_err(|e| {
        AppError::new(
            "model_not_cached",
            format!("Cannot open {}: {e}", path.display()),
        )
    })?;
    let metadata = file.metadata()?;
    let bytes = metadata.len();
    let (modified_ns, file_id) = metadata_identity(&metadata);
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 131_072];
    loop {
        execution.check()?;
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(ResolvedFile {
        path: path.into(),
        bytes,
        sha256: format!("{:x}", hash.finalize()),
        modified_ns,
        file_id,
    })
}

fn metadata_identity(metadata: &fs::Metadata) -> (Option<u64>, Option<(u64, u64)>) {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| u64::try_from(d.as_nanos()).ok());
    #[cfg(unix)]
    let identity = {
        use std::os::unix::fs::MetadataExt;
        Some((metadata.dev(), metadata.ino()))
    };
    #[cfg(not(unix))]
    let identity = None;
    (modified, identity)
}

fn hub_endpoint() -> Result<reqwest::Url> {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let url =
        reqwest::Url::parse(&endpoint).map_err(|_| AppError::invalid("Invalid HF_ENDPOINT"))?;
    if !matches!(url.scheme(), "https" | "http")
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(AppError::invalid(
            "HF_ENDPOINT must be an HTTP(S) URL without credentials",
        ));
    }
    Ok(url)
}

fn authenticated(request: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
    let token = std::env::var("HF_TOKEN").ok().or_else(|| {
        if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN")
            .is_ok_and(|v| matches!(v.to_ascii_uppercase().as_str(), "1" | "TRUE" | "YES" | "ON"))
        {
            None
        } else {
            if let Some(path) = std::env::var_os("HF_TOKEN_PATH") {
                fs::read_to_string(path)
                    .ok()
                    .map(|token| token.trim().to_owned())
                    .filter(|token| !token.is_empty())
            } else {
                home_cache().token()
            }
        }
    });
    match token {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

pub(crate) fn request_timeout(execution: &Execution) -> Duration {
    execution
        .remaining()
        .unwrap_or(Duration::from_secs(180))
        .min(Duration::from_secs(180))
}

pub(crate) fn client(execution: &Execution) -> Result<Client> {
    execution.check()?;
    Client::builder()
        // Artifact bodies can take minutes on slow connections. Only the user's
        // remaining command budget limits their duration; connection setup is bounded.
        .timeout(execution.remaining())
        .connect_timeout(Duration::from_secs(15))
        .user_agent(concat!("nalcos/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(network_error)
}

pub(crate) fn network_error(error: reqwest::Error) -> AppError {
    let code = if error.is_timeout() {
        "timeout"
    } else {
        "provider_unavailable"
    };
    AppError::new(code, error.without_url().to_string())
}

pub(crate) fn checked_response(
    response: reqwest::blocking::Response,
    operation: &str,
) -> Result<reqwest::blocking::Response> {
    if response.status().is_success() {
        Ok(response)
    } else {
        Err(AppError::new(
            "provider_http_error",
            format!("Cannot {operation}: HTTP {}", response.status()),
        ))
    }
}

fn download_context(mut error: AppError, operation: &str) -> AppError {
    error.message = format!("{operation}: {}", error.message);
    error.action = Some("Retry the same nalcos init or nalcos sync command with downloads enabled; completed downloads remain cached. Increase --timeout if the connection needs more time".into());
    error
}

/// Caller holds the target's installer lock. Only this named partial belongs to
/// the downloader; failed/crashed transfers restart rather than append unverified bytes.
pub(crate) fn download_artifact(
    target: &Path,
    artifact: &str,
    sha256: Option<&str>,
    request: impl Fn() -> Result<reqwest::blocking::RequestBuilder>,
    execution: &Execution,
) -> Result<()> {
    let temporary = target.with_extension("nalcos-download.part");
    if discard_partial(&temporary).map_err(|error| download_context(error, artifact))? {
        eprintln!("Restarting {artifact}; discarded its incomplete download");
    }
    let mut attempt = 1;
    loop {
        execution
            .check()
            .map_err(|error| download_context(error, artifact))?;
        eprintln!("Downloading {artifact} (attempt {attempt}/3)");
        let result = request().and_then(|request| {
            download_attempt(request, &temporary, target, artifact, sha256, execution)
        });
        let Err(mut error) = result else {
            return Ok(());
        };
        match discard_partial(&temporary) {
            Ok(true) => eprintln!("Discarded incomplete download: {artifact}"),
            Ok(false) => {}
            Err(cleanup) => {
                error.message = format!("{}; {}", error.message, cleanup.message);
                return Err(download_context(error, artifact));
            }
        }
        if error.code != "download_incomplete" || attempt == 3 {
            if error.code == "download_incomplete" {
                error.code = "download_failed".into();
            }
            return Err(download_context(error, artifact));
        }
        eprintln!("Retrying {artifact} from byte 0: {}", error.message);
        for _ in 0..attempt * 10 {
            execution
                .check()
                .map_err(|error| download_context(error, artifact))?;
            std::thread::sleep(Duration::from_millis(50));
        }
        attempt += 1;
    }
}

fn download_attempt(
    request: reqwest::blocking::RequestBuilder,
    temporary: &Path,
    target: &Path,
    artifact: &str,
    sha256: Option<&str>,
    execution: &Execution,
) -> Result<()> {
    let response = request.send();
    execution.check()?;
    let mut response = response.map_err(|error| {
        let retryable = error.is_connect() || error.is_timeout() || error.is_request();
        let mut error = network_error(error);
        error.code = if retryable {
            "download_incomplete"
        } else {
            "download_failed"
        }
        .into();
        error.message = format!("Open artifact response: {}", error.message);
        error
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(AppError::new(
            if status.is_server_error() || status.as_u16() == 429 {
                "download_incomplete"
            } else {
                "download_failed"
            },
            format!("Open artifact response: HTTP {status}"),
        ));
    }
    let total = response.content_length();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary)
        .map_err(|error| {
            AppError::new(
                "io_error",
                format!("Create partial download {}: {error}", temporary.display()),
            )
        })?;
    let mut hash = Sha256::new();
    let mut received = 0u64;
    let mut progress = Instant::now();
    transfer_progress(artifact, received, total);
    let mut buffer = [0u8; 65_536];
    loop {
        execution.check()?;
        let read = response.read(&mut buffer);
        execution.check()?;
        let count = read.map_err(|error| {
            let kind = error.kind();
            let detail = error.into_inner().map_or_else(
                || format!("response I/O error ({kind:?})"),
                |inner| match inner.downcast::<reqwest::Error>() {
                    Ok(error) => error.without_url().to_string(),
                    Err(_) => format!("response I/O error ({kind:?})"),
                },
            );
            AppError::new(
                "download_incomplete",
                format!("Read response after {received} bytes: {detail}"),
            )
        })?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|error| AppError::new("io_error", format!("Write artifact bytes: {error}")))?;
        hash.update(&buffer[..count]);
        received += count as u64;
        if progress.elapsed() >= Duration::from_secs(2) {
            transfer_progress(artifact, received, total);
            progress = Instant::now();
        }
    }
    if total.is_some_and(|total| received != total) {
        return Err(AppError::new(
            "download_incomplete",
            format!(
                "Response ended after {received} bytes; expected {}",
                total.unwrap()
            ),
        ));
    }
    eprintln!("Verifying {artifact} ({received} bytes)");
    if sha256.is_some_and(|expected| format!("{:x}", hash.finalize()) != expected) {
        return Err(AppError::new(
            "download_checksum_mismatch",
            "Artifact did not match its expected SHA256",
        ));
    }
    output
        .sync_all()
        .map_err(|error| AppError::new("io_error", format!("Flush artifact bytes: {error}")))?;
    drop(output);
    execution.check()?;
    fs::rename(temporary, target).map_err(|error| {
        AppError::new(
            "io_error",
            format!("Publish verified artifact {}: {error}", target.display()),
        )
    })?;
    eprintln!("Downloaded {artifact} ({received} bytes)");
    Ok(())
}

fn transfer_progress(artifact: &str, received: u64, total: Option<u64>) {
    if let Some(total) = total {
        eprintln!("Downloading {artifact}: {received}/{total} bytes");
    } else {
        eprintln!("Downloading {artifact}: {received} bytes");
    }
}

fn discard_partial(path: &Path) -> Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(AppError::new(
            "io_error",
            format!("Discard partial download {}: {error}", path.display()),
        )),
    }
}

pub(crate) fn acquire_lock(path: &Path, execution: &Execution) -> Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    loop {
        execution.check()?;
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(file),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(25))
            }
            Err(e) => return Err(e.into()),
        }
    }
}

pub(crate) fn temporary_path(path: &Path) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    path.with_extension(format!("nalcos-{}-{nanos}.part", std::process::id()))
}

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = temporary_path(path);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, atomic::AtomicBool};

    #[test]
    fn artifact_body_uses_the_remaining_command_budget() {
        for timeout in [None, Some(Duration::from_millis(100))] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0; 1];
                stream.read_exact(&mut request).unwrap();
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\na",
                    )
                    .unwrap();
                std::thread::sleep(Duration::from_millis(250));
                // A deadline may close the client before the last byte is sent.
                if let Err(error) = stream.write_all(b"b") {
                    assert!(matches!(
                        error.kind(),
                        std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                    ));
                }
            });
            let execution = Execution::new(timeout, Arc::new(AtomicBool::new(false)));
            let directory = tempfile::tempdir().unwrap();
            let target = directory.path().join("artifact");
            let result = download_artifact(
                &target,
                "slow fixture artifact",
                Some(&format!("{:x}", Sha256::digest(b"ab"))),
                || Ok(client(&execution)?.get(format!("http://{address}"))),
                &execution,
            );
            if timeout.is_some() {
                assert_eq!(result.unwrap_err().code, "timeout");
                assert!(!target.exists());
            } else {
                result.unwrap();
                assert_eq!(fs::read(&target).unwrap(), b"ab");
            }
            assert!(!target.with_extension("nalcos-download.part").exists());
            server.join().unwrap();
        }
    }

    #[test]
    fn truncated_download_restarts_and_invalid_checksum_never_publishes() {
        use std::io::{BufRead, BufReader};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let contents = b"verified fixture artifact";
        let server = std::thread::spawn(move || {
            for body in [&contents[..5], contents.as_slice(), contents.as_slice()] {
                let (mut stream, _) = listener.accept().unwrap();
                {
                    let mut request = BufReader::new(&mut stream);
                    let mut line = String::new();
                    while request.read_line(&mut line).unwrap() != 0 {
                        if line.ends_with("\r\n\r\n") {
                            break;
                        }
                    }
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    contents.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("artifact");
        fs::write(
            target.with_extension("nalcos-download.part"),
            "stale partial",
        )
        .unwrap();
        let execution = Execution::unlimited();
        let request = || Ok(client(&execution)?.get(format!("http://{address}")));
        download_artifact(
            &target,
            "fixture artifact",
            Some(&format!("{:x}", Sha256::digest(contents))),
            request,
            &execution,
        )
        .unwrap();
        assert_eq!(fs::read(&target).unwrap(), contents);
        assert!(!target.with_extension("nalcos-download.part").exists());

        let invalid = directory.path().join("invalid");
        let error = download_artifact(
            &invalid,
            "checksum fixture",
            Some(&"0".repeat(64)),
            request,
            &execution,
        )
        .unwrap_err();
        assert_eq!(error.code, "download_checksum_mismatch");
        assert!(error.message.contains("checksum fixture"));
        assert!(error.action.unwrap().contains("Retry"));
        assert!(!invalid.exists());
        assert!(!invalid.with_extension("nalcos-download.part").exists());
        server.join().unwrap();
    }
}
