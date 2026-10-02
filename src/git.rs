//! Read-only Git plumbing. Object identity and live ref reachability are separate:
//! an indexed object is never, by itself, evidence that it belongs to a query.

use crate::error::{AppError, Result};
use crate::execution::Execution;
use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use similar::{Algorithm, TextDiff};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

const PIPE_LIMIT: usize = 256 * 1024 * 1024;
const COMMIT_LIMIT: usize = 4 * 1024 * 1024;
/// Canonical extraction semantics persisted by the index lifecycle.
pub const EXTRACTION_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Repository {
    pub work_dir: PathBuf,
    pub common_dir: PathBuf,
    pub identity: String,
    pub object_format: String,
    pub bare: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeOptions {
    pub refs: Vec<String>,
    pub range: Option<String>,
    pub all_refs: bool,
    pub first_parent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedRoot {
    pub name: String,
    pub oid: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeSnapshot {
    pub signature: String,
    pub roots: Vec<ResolvedRoot>,
    pub excluded_roots: Vec<ResolvedRoot>,
    #[serde(skip)]
    pub oids: Vec<String>,
    pub eligible_count: usize,
    pub shallow: bool,
    pub shallow_signature: String,
    pub first_parent: bool,
}

#[derive(Debug, Clone, Default)]
pub struct FilterOptions {
    pub paths: Vec<String>,
    /// Case-insensitive literal substring of author name or email.
    pub author: Option<String>,
    /// Committer timestamps: inclusive lower bound, exclusive upper bound.
    pub since: Option<i64>,
    pub until: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitPath {
    pub display: String,
    pub bytes_hex: String,
}

impl GitPath {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        // Control characters and invalid UTF-8 are escaped for terminal/JSON display;
        // only bytes_hex participates in path identity and source lookups.
        let display = match std::str::from_utf8(bytes) {
            Ok(s) => s
                .chars()
                .flat_map(|c| {
                    if c.is_control() || c == '\\' {
                        c.escape_default().collect::<Vec<_>>()
                    } else {
                        vec![c]
                    }
                })
                .collect(),
            Err(_) => bytes
                .iter()
                .map(|b| {
                    if (0x20..=0x7e).contains(b) && *b != b'\\' {
                        (*b as char).to_string()
                    } else {
                        format!("\\x{b:02x}")
                    }
                })
                .collect(),
        };
        Self {
            display,
            bytes_hex: hex(bytes),
        }
    }

    pub fn bytes(&self) -> Result<Vec<u8>> {
        unhex(&self.bytes_hex)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRecord {
    pub oid: String,
    pub tree_oid: String,
    pub parents: Vec<String>,
    pub author_name: String,
    pub author_email: String,
    pub authored_at: i64,
    pub committed_at: i64,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    Message,
    Diff,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: String,
    pub commit_oid: String,
    pub parent_oid: Option<String>,
    pub parent_index: usize,
    pub kind: DocumentKind,
    pub old_path: Option<GitPath>,
    pub new_path: Option<GitPath>,
    pub old_blob: Option<String>,
    pub new_blob: Option<String>,
    pub old_start: u32,
    pub old_lines: u32,
    pub new_start: u32,
    pub new_lines: u32,
    pub text: String,
    pub content_hash: String,
    pub truncated: bool,
}

/// Source coordinates needed to restrict retrieval without loading document text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPaths {
    pub id: String,
    pub commit_oid: String,
    pub parent_index: usize,
    pub kind: DocumentKind,
    pub old_path: Option<GitPath>,
    pub new_path: Option<GitPath>,
}

impl From<&DocumentRecord> for DocumentPaths {
    fn from(document: &DocumentRecord) -> Self {
        Self {
            id: document.id.clone(),
            commit_oid: document.commit_oid.clone(),
            parent_index: document.parent_index,
            kind: document.kind.clone(),
            old_path: document.old_path.clone(),
            new_path: document.new_path.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionOmission {
    pub commit_oid: String,
    pub path: Option<GitPath>,
    pub reason: String,
    pub detail: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub documents: Vec<DocumentRecord>,
    pub omissions: Vec<ExtractionOmission>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexOptions {
    pub exclude_paths: Vec<String>,
    pub max_blob_bytes: usize,
    pub max_patch_bytes: usize,
    pub max_document_bytes: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            exclude_paths: vec![],
            max_blob_bytes: 256 * 1024,
            max_patch_bytes: 256 * 1024,
            max_document_bytes: 16 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShowOptions {
    pub parent: usize,
    pub paths: Vec<String>,
    pub context: usize,
    pub max_bytes: Option<usize>,
}

impl Default for ShowOptions {
    fn default() -> Self {
        Self {
            parent: 1,
            paths: vec![],
            context: 3,
            max_bytes: Some(64 * 1024),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowResult {
    pub commit: CommitRecord,
    pub parent_oid: Option<String>,
    pub documents: Vec<DocumentRecord>,
    pub omissions: Vec<ExtractionOmission>,
    pub patch: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub document_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    pub valid_ids: Vec<String>,
    pub missing_ids: Vec<String>,
    pub errors: Vec<ValidationIssue>,
}

#[derive(Debug, Clone)]
struct RawChange {
    commit_oid: String,
    old_mode: String,
    new_mode: String,
    old_blob: Option<String>,
    new_blob: Option<String>,
    old_path: Option<GitPath>,
    new_path: Option<GitPath>,
}

#[derive(Debug)]
struct GitOutput {
    success: bool,
    code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl Repository {
    pub fn discover(path: &Path, execution: &Execution) -> Result<Self> {
        let cwd = std::fs::canonicalize(path).map_err(|e| {
            AppError::invalid(format!(
                "Cannot discover repository at {}: {e}",
                path.display()
            ))
        })?;
        if !cwd.is_dir() {
            return Err(AppError::invalid("Repository path must be a directory"));
        }
        // The option probe is intentional: silently ignoring an unsupported environment
        // variable would allow partial-clone reads to fetch from a remote.
        let capability = git_output(&cwd, &["--version"], None, execution, 1024)?;
        if !capability.success {
            return Err(AppError::new(
                "unsupported_git",
                format!(
                    "Git failed the --no-lazy-fetch capability check; Git 2.45 or later is required: {}",
                    String::from_utf8_lossy(&capability.stderr)
                        .lines()
                        .next()
                        .unwrap_or("capability probe failed")
                ),
            )
            .action("Install Git 2.45 or later and ensure it is first on PATH"));
        }
        let bare = text_output(git(
            &cwd,
            &["rev-parse", "--is-bare-repository"],
            None,
            execution,
            1024,
        )?)? == "true";
        let common = git(
            &cwd,
            &["rev-parse", "--path-format=absolute", "--git-common-dir"],
            None,
            execution,
            16384,
        )?;
        let common_dir = std::fs::canonicalize(path_output(common)?)?;
        let work_dir = if bare {
            cwd.clone()
        } else {
            std::fs::canonicalize(path_output(git(
                &cwd,
                &["rev-parse", "--show-toplevel"],
                None,
                execution,
                16384,
            )?)?)?
        };
        let object_format = text_output(git(
            &cwd,
            &["rev-parse", "--show-object-format"],
            None,
            execution,
            1024,
        )?)?;
        if !matches!(object_format.as_str(), "sha1" | "sha256") {
            return Err(AppError::new(
                "unsupported_repository",
                format!("Unsupported Git object format: {object_format}"),
            ));
        }
        reject_grafts(&common_dir)?;
        let identity = digest(common_dir.as_os_str().as_bytes());
        Ok(Self {
            work_dir,
            common_dir,
            identity,
            object_format,
            bare,
        })
    }

    pub fn resolve_scope(
        &self,
        options: &ScopeOptions,
        execution: &Execution,
    ) -> Result<ScopeSnapshot> {
        let (roots, excluded_roots, shallow_bytes) = self.resolve_roots(options, execution)?;
        let signature = scope_signature(
            &roots,
            &excluded_roots,
            &shallow_bytes,
            options.first_parent,
        )?;
        let mut oids = Vec::new();
        if !roots.is_empty() {
            let mut args = vec!["rev-list".to_string(), "--topo-order".into()];
            if options.first_parent {
                args.push("--first-parent".into());
            }
            args.push("--stdin".into());
            let input = roots
                .iter()
                .map(|r| format!("{}\n", r.oid))
                .chain(excluded_roots.iter().map(|r| format!("^{}\n", r.oid)))
                .collect::<String>();
            let output = self.run(&args, Some(input.into_bytes()), execution, PIPE_LIMIT)?;
            for line in output.split(|b| *b == b'\n').filter(|s| !s.is_empty()) {
                oids.push(parse_oid(line)?);
            }
        }
        Ok(ScopeSnapshot {
            signature,
            roots,
            excluded_roots,
            eligible_count: oids.len(),
            oids,
            shallow: !shallow_bytes.is_empty(),
            shallow_signature: digest(&shallow_bytes),
            first_parent: options.first_parent,
        })
    }

    pub fn validate_snapshot(
        &self,
        options: &ScopeOptions,
        snapshot: &ScopeSnapshot,
        execution: &Execution,
    ) -> Result<bool> {
        let (roots, excluded, shallow) = self.resolve_roots(options, execution)?;
        Ok(
            scope_signature(&roots, &excluded, &shallow, options.first_parent)?
                == snapshot.signature,
        )
    }

    fn resolve_roots(
        &self,
        options: &ScopeOptions,
        execution: &Execution,
    ) -> Result<(Vec<ResolvedRoot>, Vec<ResolvedRoot>, Vec<u8>)> {
        reject_grafts(&self.common_dir)?;
        if usize::from(!options.refs.is_empty())
            + usize::from(options.range.is_some())
            + usize::from(options.all_refs)
            > 1
        {
            return Err(AppError::invalid(
                "Choose --ref, --range, or --all-refs, not more than one",
            ));
        }
        let mut roots = Vec::new();
        let mut excluded = Vec::new();
        if let Some(range) = &options.range {
            let (a, b) = range
                .split_once("..")
                .ok_or_else(|| AppError::invalid("A range must have the form A..B"))?;
            if a.is_empty()
                || b.is_empty()
                || a.contains("..")
                || b.contains("..")
                || b.starts_with('.')
            {
                return Err(AppError::invalid(
                    "A range must have two endpoints and exactly two dots: A..B",
                ));
            }
            roots.push(ResolvedRoot {
                name: b.into(),
                oid: self.resolve_commit(b, execution)?,
            });
            excluded.push(ResolvedRoot {
                name: a.into(),
                oid: self.resolve_commit(a, execution)?,
            });
        } else if options.all_refs {
            let data = self.run(&["for-each-ref", "--format=%(refname)%00%(objectname)%00%(objecttype)%00%(*objectname)%00%(*objecttype)"], None, execution, PIPE_LIMIT)?;
            for line in data.split(|b| *b == b'\n').filter(|s| !s.is_empty()) {
                let fields: Vec<_> = line.split(|b| *b == 0).collect();
                if fields.len() != 5 {
                    return Err(malformed("for-each-ref record"));
                }
                let oid = match fields[2] {
                    b"commit" => Some(parse_oid(fields[1])?),
                    b"tag" if fields[4] == b"commit" => Some(parse_oid(fields[3])?),
                    b"tag" => self.try_resolve_commit(&parse_oid(fields[1])?, execution)?,
                    _ => None,
                };
                if let Some(oid) = oid {
                    roots.push(ResolvedRoot {
                        name: String::from_utf8_lossy(fields[0]).into_owned(),
                        oid,
                    });
                }
            }
            let worktrees = self.run(
                &["worktree", "list", "--porcelain", "-z"],
                None,
                execution,
                PIPE_LIMIT,
            )?;
            let mut name = String::new();
            for field in worktrees.split(|b| *b == 0) {
                if let Some(path) = field.strip_prefix(b"worktree ") {
                    name = format!("worktree:{}:HEAD", hex(path));
                }
                if let Some(oid) = field.strip_prefix(b"HEAD ")
                    && !oid.iter().all(|b| *b == b'0')
                {
                    roots.push(ResolvedRoot {
                        name: name.clone(),
                        oid: parse_oid(oid)?,
                    });
                }
            }
        } else if !options.refs.is_empty() {
            for name in &options.refs {
                roots.push(ResolvedRoot {
                    name: name.clone(),
                    oid: self.resolve_commit(name, execution)?,
                });
            }
        } else if let Some(oid) = self.try_resolve_commit("HEAD", execution)? {
            roots.push(ResolvedRoot {
                name: "HEAD".into(),
                oid,
            });
        } else {
            let symbolic = git_output(
                &self.work_dir,
                &["symbolic-ref", "--quiet", "HEAD"],
                None,
                execution,
                16384,
            )?;
            if !symbolic.success {
                return Err(AppError::new(
                    "missing_object",
                    "HEAD does not resolve to an available commit",
                ));
            }
            let name = text_output(symbolic.stdout)?;
            let exists = git_output(
                &self.work_dir,
                &["show-ref", "--verify", "--quiet", &name],
                None,
                execution,
                16384,
            )?;
            if exists.success || !exists.stderr.is_empty() {
                return Err(AppError::new(
                    "missing_object",
                    "HEAD references a commit that Git cannot read",
                ));
            }
        }
        roots.sort_by(|a, b| a.name.cmp(&b.name).then(a.oid.cmp(&b.oid)));
        roots.dedup();
        let shallow_path = path_output(self.run(
            &[
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "shallow",
            ],
            None,
            execution,
            16384,
        )?)?;
        let shallow = match std::fs::read(shallow_path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => vec![],
            Err(e) => return Err(e.into()),
        };
        Ok((roots, excluded, shallow))
    }

    pub fn resolve_commit(&self, revision: &str, execution: &Execution) -> Result<String> {
        self.try_resolve_commit(revision, execution)?
            .ok_or_else(|| {
                AppError::invalid(format!(
                    "Revision {revision:?} does not resolve to an available commit"
                ))
            })
    }

    fn try_resolve_commit(&self, revision: &str, execution: &Execution) -> Result<Option<String>> {
        reject_grafts(&self.common_dir)?;
        if revision.is_empty() || revision.contains('\0') {
            return Err(AppError::invalid(
                "Invalid empty or NUL-containing revision",
            ));
        }
        let rev = format!("{revision}^{{commit}}");
        let out = git_output(
            &self.work_dir,
            &["rev-parse", "--verify", "--quiet", "--end-of-options", &rev],
            None,
            execution,
            16384,
        )?;
        if out.success {
            Ok(Some(parse_oid(trim_ascii(&out.stdout))?))
        } else if out.code == Some(1) {
            Ok(None)
        } else {
            Err(git_error("resolve revision", &out.stderr))
        }
    }

    pub fn load_commits(
        &self,
        oids: &[String],
        execution: &Execution,
    ) -> Result<Vec<CommitRecord>> {
        let mut result = Vec::with_capacity(oids.len());
        for batch in oids.chunks(128) {
            let objects = self.read_objects(batch, COMMIT_LIMIT, execution)?;
            for oid in batch {
                let (kind, bytes) = objects.get(oid).ok_or_else(|| missing(oid))?;
                if kind != "commit" {
                    return Err(AppError::invalid(format!(
                        "Object {oid} is {kind}, not a commit"
                    )));
                }
                result.push(parse_commit(oid, bytes)?);
            }
        }
        Ok(result)
    }

    pub fn filter_scope(
        &self,
        snapshot: &ScopeSnapshot,
        filters: &FilterOptions,
        execution: &Execution,
    ) -> Result<Vec<String>> {
        if filters.paths.is_empty()
            && filters.author.is_none()
            && filters.since.is_none()
            && filters.until.is_none()
        {
            return Ok(snapshot.oids.clone());
        }
        let mut eligible = Vec::new();
        for batch in snapshot.oids.chunks(128) {
            let records = self.load_commits(batch, execution)?;
            let author = filters.author.as_ref().map(|s| s.to_lowercase());
            let selected: Vec<_> = records
                .into_iter()
                .filter(|c| {
                    author.as_ref().is_none_or(|a| {
                        format!("{} <{}>", c.author_name, c.author_email)
                            .to_lowercase()
                            .contains(a)
                    }) && filters.since.is_none_or(|s| c.committed_at >= s)
                        && filters.until.is_none_or(|u| c.committed_at < u)
                })
                .collect();
            if filters.paths.is_empty() {
                eligible.extend(selected.into_iter().map(|c| c.oid));
            } else {
                let changes = self.raw_changes(&selected, 1, &filters.paths, execution)?;
                let matching: HashSet<_> = changes.into_iter().map(|c| c.commit_oid).collect();
                eligible.extend(
                    selected
                        .into_iter()
                        .filter(|c| matching.contains(&c.oid))
                        .map(|c| c.oid),
                );
            }
        }
        Ok(eligible)
    }

    /// Apply Git pathspecs to both sides of each canonical change before ranking.
    /// Commit messages remain eligible when their first-parent changes touch a path.
    pub fn matching_document_ids(
        &self,
        documents: &[DocumentPaths],
        paths: &[String],
        execution: &Execution,
    ) -> Result<HashSet<String>> {
        execution.check()?;
        if paths.is_empty() {
            return Ok(documents
                .iter()
                .map(|document| document.id.clone())
                .collect());
        }
        let mut groups = BTreeMap::<usize, HashMap<&str, Vec<&DocumentPaths>>>::new();
        for document in documents {
            execution.check()?;
            let parent = if document.kind == DocumentKind::Message {
                1
            } else {
                document.parent_index
            };
            groups
                .entry(parent)
                .or_default()
                .entry(&document.commit_oid)
                .or_default()
                .push(document);
        }
        let mut allowed = HashSet::new();
        for (parent, by_commit) in groups {
            let mut oids: Vec<_> = by_commit.keys().map(|oid| (*oid).to_owned()).collect();
            oids.sort();
            for batch in oids.chunks(128) {
                execution.check()?;
                let commits = self.load_commits(batch, execution)?;
                let changes = self.raw_changes(&commits, parent, paths, execution)?;
                let matching_commits: HashSet<_> = changes
                    .iter()
                    .map(|change| change.commit_oid.as_str())
                    .collect();
                let matching_paths: HashSet<_> = changes
                    .iter()
                    .map(|change| {
                        (
                            change.commit_oid.as_str(),
                            change.old_path.as_ref().map(|path| path.bytes_hex.as_str()),
                            change.new_path.as_ref().map(|path| path.bytes_hex.as_str()),
                        )
                    })
                    .collect();
                for oid in batch {
                    for document in &by_commit[oid.as_str()] {
                        execution.check()?;
                        let matches = if document.kind == DocumentKind::Message {
                            matching_commits.contains(oid.as_str())
                        } else {
                            matching_paths.contains(&(
                                oid.as_str(),
                                document
                                    .old_path
                                    .as_ref()
                                    .map(|path| path.bytes_hex.as_str()),
                                document
                                    .new_path
                                    .as_ref()
                                    .map(|path| path.bytes_hex.as_str()),
                            ))
                        };
                        if matches {
                            allowed.insert(document.id.clone());
                        }
                    }
                }
            }
        }
        Ok(allowed)
    }

    pub fn extract_documents(
        &self,
        commits: &[CommitRecord],
        options: &IndexOptions,
        execution: &Execution,
    ) -> Result<ExtractionResult> {
        self.extract(commits, options, 1, &[], 3, true, execution)
    }

    #[allow(clippy::too_many_arguments)]
    fn extract(
        &self,
        commits: &[CommitRecord],
        options: &IndexOptions,
        parent: usize,
        paths: &[String],
        context: usize,
        include_messages: bool,
        execution: &Execution,
    ) -> Result<ExtractionResult> {
        if options.max_document_bytes < 256
            || options.max_patch_bytes == 0
            || options.max_blob_bytes == 0
        {
            return Err(AppError::invalid(
                "Extraction limits require positive blob/patch limits and at least 256 document bytes",
            ));
        }
        let excludes = exclusion_globs(&options.exclude_paths)?;
        let mut result = ExtractionResult::default();
        for batch in commits.chunks(4) {
            execution.check()?;
            let by_oid: HashMap<_, _> = batch.iter().map(|c| (c.oid.as_str(), c)).collect();
            let parent_oids: Vec<_> = batch
                .iter()
                .map(|c| selected_parent(c, parent))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let parent_info = self.object_info(&parent_oids, execution)?;
            let mut comparable = Vec::new();
            for commit in batch {
                let available = selected_parent(commit, parent)?.as_ref().is_none_or(
                    |oid| matches!(parent_info.get(oid), Some(Some((kind,_))) if kind == "commit"),
                );
                if available {
                    comparable.push(commit.clone());
                } else {
                    result.omissions.push(ExtractionOmission { commit_oid: commit.oid.clone(), path: None,
                    reason: "missing_object".into(), detail: "Parent commit is unavailable locally; first-parent evidence could not be extracted".into() });
                }
            }
            let mut changes = self.raw_changes(&comparable, parent, paths, execution)?;
            let mut wanted = BTreeSet::new();
            let mut source_bytes: HashMap<String, usize> = HashMap::new();
            let all_blobs: Vec<_> = changes
                .iter()
                .flat_map(|c| c.old_blob.iter().chain(c.new_blob.iter()))
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let info = self.object_info(&all_blobs, execution)?;
            let mut readable = HashSet::new();
            for (i, change) in changes.iter().enumerate() {
                let path = change.new_path.as_ref().or(change.old_path.as_ref());
                let is_excluded = excluded(change, &excludes)?;
                if !is_excluded {
                    let commit = by_oid[change.commit_oid.as_str()];
                    result.documents.push(make_document(
                        commit,
                        selected_parent(commit, parent)?,
                        parent,
                        DocumentKind::Diff,
                        Some(change),
                        (0, 0, 0, 0),
                        file_prefix(commit, change),
                        false,
                    )?);
                }
                let reason = if is_excluded {
                    Some((
                        "excluded",
                        "Path is excluded by index configuration".to_string(),
                    ))
                } else if change.old_mode == "160000" || change.new_mode == "160000" {
                    Some((
                        "submodule",
                        "Submodule content belongs to its own repository".into(),
                    ))
                } else {
                    let mut bytes = 0usize;
                    let mut failure = None;
                    for oid in change.old_blob.iter().chain(change.new_blob.iter()) {
                        match info.get(oid) {
                            Some(Some((kind, size)))
                                if kind == "blob" && *size <= options.max_blob_bytes =>
                            {
                                bytes = bytes.saturating_add(*size);
                            }
                            Some(Some((_, size))) => {
                                failure = Some((
                                    "blob_limit",
                                    format!(
                                        "Blob {oid} has {size} bytes; limit is {}",
                                        options.max_blob_bytes
                                    ),
                                ))
                            }
                            _ => {
                                failure = Some((
                                    "missing_object",
                                    format!(
                                        "Blob {oid} is unavailable locally; no fetch was attempted"
                                    ),
                                ))
                            }
                        }
                    }
                    let used = source_bytes.entry(change.commit_oid.clone()).or_default();
                    let budget = options
                        .max_patch_bytes
                        .saturating_mul(16)
                        .max(options.max_blob_bytes.saturating_mul(2));
                    if failure.is_none() && used.saturating_add(bytes) > budget {
                        failure = Some((
                            "source_budget",
                            format!("Commit exceeded its {budget}-byte source extraction budget"),
                        ));
                    }
                    if failure.is_none() {
                        *used = used.saturating_add(bytes);
                    }
                    failure
                };
                if let Some((reason, detail)) = reason {
                    result.omissions.push(ExtractionOmission {
                        commit_oid: change.commit_oid.clone(),
                        path: path.cloned(),
                        reason: reason.into(),
                        detail,
                    });
                } else {
                    readable.insert(i);
                    wanted.extend(
                        change
                            .old_blob
                            .iter()
                            .chain(change.new_blob.iter())
                            .cloned(),
                    );
                }
            }
            let blobs = self.read_objects(
                &wanted.into_iter().collect::<Vec<_>>(),
                options.max_blob_bytes,
                execution,
            )?;
            let mut patch_bytes: HashMap<String, usize> = HashMap::new();
            if include_messages {
                for commit in batch {
                    let (text, truncated) =
                        truncate_utf8(&commit.message, options.max_document_bytes);
                    result.documents.push(make_document(
                        commit,
                        None,
                        0,
                        DocumentKind::Message,
                        None,
                        (0, 0, 0, 0),
                        text,
                        truncated,
                    )?);
                    if truncated {
                        result.omissions.push(ExtractionOmission {
                            commit_oid: commit.oid.clone(),
                            path: None,
                            reason: "message_limit".into(),
                            detail: "Commit message exceeded the document byte limit".into(),
                        });
                    }
                }
            }
            for (i, change) in changes.drain(..).enumerate() {
                execution.check()?;
                if !readable.contains(&i) {
                    continue;
                }
                let commit = by_oid[change.commit_oid.as_str()];
                let old = blob_text(change.old_blob.as_ref(), &blobs);
                let new = blob_text(change.new_blob.as_ref(), &blobs);
                let (old, new) = match (old, new) {
                    (Some(old), Some(new)) => (old, new),
                    _ => {
                        result.omissions.push(ExtractionOmission {
                            commit_oid: commit.oid.clone(),
                            path: change.new_path.clone().or(change.old_path.clone()),
                            reason: "binary".into(),
                            detail: "Binary or non-UTF-8 blob content was not indexed".into(),
                        });
                        continue;
                    }
                };
                let parent_oid = selected_parent(commit, parent)?;
                let mut diff_config = TextDiff::configure();
                diff_config.algorithm(Algorithm::Patience);
                if let Some(remaining) = execution.remaining() {
                    diff_config.timeout(remaining);
                }
                let diff = diff_config.diff_lines(old, new);
                execution.check()?;
                let hunks = source_hunks(&diff, context)?;
                let prefix = file_prefix(commit, &change);
                for hunk in hunks {
                    let used = patch_bytes.entry(commit.oid.clone()).or_default();
                    if used.saturating_add(hunk.text.len()) > options.max_patch_bytes {
                        result.omissions.push(ExtractionOmission {
                            commit_oid: commit.oid.clone(),
                            path: change.new_path.clone().or(change.old_path.clone()),
                            reason: "patch_limit".into(),
                            detail: "Commit patch exceeded its configured byte limit".into(),
                        });
                        break;
                    }
                    *used = used.saturating_add(hunk.text.len());
                    for chunk in split_hunk(
                        &hunk,
                        options.max_document_bytes.saturating_sub(prefix.len()),
                    )? {
                        if chunk.text.len().saturating_add(prefix.len())
                            > options.max_document_bytes
                        {
                            result.omissions.push(ExtractionOmission {
                                commit_oid: commit.oid.clone(),
                                path: change.new_path.clone().or(change.old_path.clone()),
                                reason: "document_limit".into(),
                                detail: "A changed line exceeded the document byte limit".into(),
                            });
                            continue;
                        }
                        result.documents.push(make_document(
                            commit,
                            parent_oid.clone(),
                            parent,
                            DocumentKind::Diff,
                            Some(&change),
                            (
                                chunk.old_start,
                                chunk.old_lines,
                                chunk.new_start,
                                chunk.new_lines,
                            ),
                            format!("{prefix}{}", chunk.text),
                            false,
                        )?);
                    }
                }
            }
        }
        Ok(result)
    }

    pub fn show(
        &self,
        revision: &str,
        options: &ShowOptions,
        execution: &Execution,
    ) -> Result<ShowResult> {
        if options.max_bytes == Some(0) {
            return Err(AppError::invalid("Show output limits must be positive"));
        }
        let oid = self.resolve_commit(revision, execution)?;
        let commit = self.load_commits(&[oid], execution)?.remove(0);
        let parent_oid = selected_parent(&commit, options.parent)?;
        let extraction = self.extract(
            std::slice::from_ref(&commit),
            &IndexOptions {
                exclude_paths: vec![],
                max_blob_bytes: options.max_bytes.unwrap_or(PIPE_LIMIT).max(1024 * 1024),
                max_patch_bytes: options.max_bytes.unwrap_or(PIPE_LIMIT),
                max_document_bytes: options.max_bytes.unwrap_or(PIPE_LIMIT).max(256),
            },
            options.parent,
            &options.paths,
            options.context,
            false,
            execution,
        )?;
        let complete = extraction
            .documents
            .iter()
            .map(|d| d.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let (patch, clipped) = truncate_utf8(&complete, options.max_bytes.unwrap_or(usize::MAX));
        Ok(ShowResult {
            commit,
            parent_oid,
            patch,
            truncated: clipped || !extraction.omissions.is_empty(),
            documents: extraction.documents,
            omissions: extraction.omissions,
        })
    }

    pub fn validate_documents(
        &self,
        documents: &[DocumentRecord],
        execution: &Execution,
    ) -> Result<ValidationReport> {
        let mut result = ValidationReport::default();
        let oids: Vec<_> = documents
            .iter()
            .map(|d| d.commit_oid.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let info = self.object_info(&oids, execution)?;
        let present: Vec<_> = oids
            .into_iter()
            .filter(|oid| matches!(info.get(oid), Some(Some((kind,_))) if kind == "commit"))
            .collect();
        let commits = self.load_commits(&present, execution)?;
        let by_oid: HashMap<_, _> = commits.iter().map(|c| (c.oid.as_str(), c)).collect();
        let mut changes = Vec::new();
        for parent in documents
            .iter()
            .filter(|d| d.kind == DocumentKind::Diff)
            .map(|d| d.parent_index)
            .collect::<BTreeSet<_>>()
        {
            let candidates: Vec<_> = commits
                .iter()
                .filter(|c| {
                    documents.iter().any(|d| {
                        d.commit_oid == c.oid
                            && d.parent_index == parent
                            && d.kind == DocumentKind::Diff
                    })
                })
                .cloned()
                .collect();
            let parent_oids = candidates
                .iter()
                .filter_map(|c| selected_parent(c, parent).ok().flatten())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let parent_info = self.object_info(&parent_oids, execution)?;
            let selected = candidates
                .into_iter()
                .filter(|c| match selected_parent(c, parent) {
                    Ok(None) => true,
                    Ok(Some(oid)) => {
                        matches!(parent_info.get(&oid), Some(Some((kind, _))) if kind == "commit")
                    }
                    Err(_) => false,
                })
                .collect::<Vec<_>>();
            changes.extend(
                self.raw_changes(&selected, parent, &[], execution)?
                    .into_iter()
                    .map(|c| (parent, c)),
            );
        }
        let blobs: Vec<_> = documents
            .iter()
            .filter(|d| d.kind == DocumentKind::Diff && (d.old_lines != 0 || d.new_lines != 0))
            .flat_map(|d| d.old_blob.iter().chain(d.new_blob.iter()))
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let blob_info = self.object_info(&blobs, execution)?;
        let readable: Vec<_> = blobs.into_iter().filter(|oid| matches!(blob_info.get(oid), Some(Some((kind,size))) if kind == "blob" && *size <= PIPE_LIMIT)).collect();
        let blob_data = self.read_objects(&readable, PIPE_LIMIT, execution)?;
        for doc in documents {
            execution.check()?;
            let issue = match by_oid.get(doc.commit_oid.as_str()) {
                None => Some("commit_missing".to_string()),
                Some(_) if digest(doc.text.as_bytes()) != doc.content_hash => {
                    Some("cached_content_hash_mismatch".into())
                }
                Some(commit) if doc.kind == DocumentKind::Message => {
                    let valid = if doc.truncated {
                        commit.message.starts_with(&doc.text)
                    } else {
                        commit.message == doc.text
                    };
                    (!valid).then(|| "commit_message_mismatch".into())
                }
                Some(commit) => {
                    let parent = selected_parent(commit, doc.parent_index).ok();
                    let source = changes.iter().find(|(p, c)| {
                        *p == doc.parent_index
                            && c.commit_oid == doc.commit_oid
                            && c.old_path == doc.old_path
                            && c.new_path == doc.new_path
                            && c.old_blob == doc.old_blob
                            && c.new_blob == doc.new_blob
                    });
                    match source.filter(|_| parent.as_ref() == Some(&doc.parent_oid)) {
                        None => Some("change_not_in_commit".into()),
                        Some((_, source))
                            if (doc.old_start, doc.old_lines, doc.new_start, doc.new_lines)
                                == (0, 0, 0, 0) =>
                        {
                            (doc.text != file_prefix(commit, source))
                                .then(|| "path_metadata_mismatch".into())
                        }
                        Some((_, source)) => match (
                            blob_text(doc.old_blob.as_ref(), &blob_data),
                            blob_text(doc.new_blob.as_ref(), &blob_data),
                        ) {
                            (Some(old), Some(new))
                                if verify_hunk_document(doc, commit, source, old, new) =>
                            {
                                None
                            }
                            (Some(_), Some(_)) => Some("evidence_does_not_match_source".into()),
                            _ => Some("evidence_object_missing".into()),
                        },
                    }
                }
            };
            if let Some(reason) = issue {
                result.missing_ids.push(doc.id.clone());
                result.errors.push(ValidationIssue {
                    document_id: doc.id.clone(),
                    reason,
                });
            } else {
                result.valid_ids.push(doc.id.clone());
            }
        }
        Ok(result)
    }

    fn raw_changes(
        &self,
        commits: &[CommitRecord],
        parent: usize,
        paths: &[String],
        execution: &Execution,
    ) -> Result<Vec<RawChange>> {
        if commits.is_empty() {
            return Ok(vec![]);
        }
        let mut input = String::new();
        for commit in commits {
            input.push_str(&commit.oid);
            if let Some(oid) = selected_parent(commit, parent)? {
                input.push(' ');
                input.push_str(&oid);
            }
            input.push('\n');
        }
        let mut args: Vec<String> = [
            "diff-tree",
            "--stdin",
            "--raw",
            "-z",
            "-r",
            "--root",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--no-abbrev",
            "-M50%",
            "-l200",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        args.push("--".into());
        let mut changes = parse_raw_changes(&self.run(
            &args,
            Some(input.clone().into_bytes()),
            execution,
            PIPE_LIMIT,
        )?)?;
        if !paths.is_empty() {
            // Git pathspecs can hide one side before rename detection. Preserve the
            // unfiltered change and use Git's filtered paths only for selection.
            args.extend(paths.iter().cloned());
            let matching = parse_raw_changes(&self.run(
                &args,
                Some(input.into_bytes()),
                execution,
                PIPE_LIMIT,
            )?)?
            .into_iter()
            .flat_map(|c| {
                c.old_path
                    .into_iter()
                    .chain(c.new_path)
                    .map(move |p| (c.commit_oid.clone(), p.bytes_hex))
            })
            .collect::<HashSet<_>>();
            changes.retain(|c| {
                c.old_path
                    .iter()
                    .chain(c.new_path.iter())
                    .any(|p| matching.contains(&(c.commit_oid.clone(), p.bytes_hex.clone())))
            });
        }
        Ok(changes)
    }

    fn object_info(
        &self,
        oids: &[String],
        execution: &Execution,
    ) -> Result<HashMap<String, Option<(String, usize)>>> {
        if oids.is_empty() {
            return Ok(HashMap::new());
        }
        for oid in oids {
            parse_oid(oid.as_bytes())?;
        }
        let input = format!("{}\n", oids.join("\n"));
        let output = self.run(
            &[
                "cat-file",
                "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            ],
            Some(input.into_bytes()),
            execution,
            PIPE_LIMIT,
        )?;
        let mut result = HashMap::new();
        for line in output.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let fields: Vec<_> = line.split(|b| *b == b' ').collect();
            if fields.len() == 2 && fields[1] == b"missing" {
                result.insert(parse_oid(fields[0])?, None);
            } else if fields.len() == 3 {
                result.insert(
                    parse_oid(fields[0])?,
                    Some((
                        ascii(fields[1])?,
                        ascii(fields[2])?
                            .parse()
                            .map_err(|_| malformed("object size"))?,
                    )),
                );
            } else {
                return Err(malformed("cat-file object header"));
            }
        }
        Ok(result)
    }

    fn read_objects(
        &self,
        oids: &[String],
        max_size: usize,
        execution: &Execution,
    ) -> Result<HashMap<String, (String, Vec<u8>)>> {
        if oids.is_empty() {
            return Ok(HashMap::new());
        }
        let info = self.object_info(oids, execution)?;
        for oid in oids {
            match info.get(oid) {
                Some(Some((_, size))) if *size <= max_size => {}
                Some(Some((_, size))) => {
                    return Err(AppError::new(
                        "source_limit",
                        format!("Object {oid} has {size} bytes; read limit is {max_size}"),
                    ));
                }
                _ => return Err(missing(oid)),
            }
        }
        let input = format!("{}\n", oids.join("\n"));
        let output = self.run(
            &["cat-file", "--batch"],
            Some(input.into_bytes()),
            execution,
            PIPE_LIMIT,
        )?;
        parse_objects(&output)
    }

    fn run<S: AsRef<OsStr>>(
        &self,
        args: &[S],
        input: Option<Vec<u8>>,
        execution: &Execution,
        limit: usize,
    ) -> Result<Vec<u8>> {
        git(&self.work_dir, args, input, execution, limit)
    }
}

fn git<S: AsRef<OsStr>>(
    cwd: &Path,
    args: &[S],
    input: Option<Vec<u8>>,
    execution: &Execution,
    limit: usize,
) -> Result<Vec<u8>> {
    let out = git_output(cwd, args, input, execution, limit)?;
    if out.success {
        Ok(out.stdout)
    } else {
        Err(git_error("read repository", &out.stderr))
    }
}

fn git_output<S: AsRef<OsStr>>(
    cwd: &Path,
    args: &[S],
    input: Option<Vec<u8>>,
    execution: &Execution,
    limit: usize,
) -> Result<GitOutput> {
    execution.check()?;
    let mut command = Command::new("git");
    command
        .current_dir(cwd)
        .args([
            "--no-pager",
            "--no-optional-locks",
            "--no-replace-objects",
            "--no-lazy-fetch",
        ])
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_NO_LAZY_FETCH", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for key in [
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_SHALLOW_FILE",
        "GIT_GRAFT_FILE",
    ] {
        command.env_remove(key);
    }
    let mut child = command.spawn().map_err(|e| {
        AppError::new(
            "git_unavailable",
            format!("Cannot run Git at {}: {e}", cwd.display()),
        )
    })?;
    let overflow = Arc::new(AtomicBool::new(false));
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| malformed("Git stdout pipe"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| malformed("Git stderr pipe"))?;
    let flag = overflow.clone();
    let reader = thread::spawn(move || read_pipe(stdout, limit, flag));
    let err_flag = overflow.clone();
    let errors = thread::spawn(move || read_pipe(stderr, 1024 * 1024, err_flag));
    let writer = input.map(|input| {
        let mut stdin = child.stdin.take().expect("piped stdin");
        thread::spawn(move || stdin.write_all(&input))
    });
    let mut cancelled = None;
    let status = loop {
        if let Err(error) = execution.check() {
            cancelled = Some(error);
            let _ = child.kill();
            break child.wait();
        }
        if overflow.load(Ordering::Relaxed) {
            cancelled = Some(AppError::new(
                "source_limit",
                format!("Git output exceeded its {limit}-byte read budget"),
            ));
            let _ = child.kill();
            break child.wait();
        }
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => thread::sleep(Duration::from_millis(5)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(e);
            }
        }
    };
    let stdout = reader
        .join()
        .map_err(|_| malformed("Git stdout reader"))??;
    let stderr = errors
        .join()
        .map_err(|_| malformed("Git stderr reader"))??;
    let write_result = writer
        .map(|w| w.join().map_err(|_| malformed("Git stdin writer")))
        .transpose()?;
    if let Some(error) = cancelled {
        return Err(error);
    }
    let status = status?;
    if overflow.load(Ordering::Relaxed) {
        return Err(AppError::new(
            "source_limit",
            format!("Git output exceeded its {limit}-byte read budget"),
        ));
    }
    if status.success()
        && let Some(write_result) = write_result
    {
        write_result?;
    }
    Ok(GitOutput {
        success: status.success(),
        code: status.code(),
        stdout,
        stderr,
    })
}

fn read_pipe(
    mut pipe: impl Read,
    limit: usize,
    overflow: Arc<AtomicBool>,
) -> std::io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0u8; 8192];
    loop {
        let count = pipe.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        let keep = count.min(limit.saturating_sub(output.len()));
        output.extend_from_slice(&buffer[..keep]);
        if keep != count {
            overflow.store(true, Ordering::Relaxed);
        }
    }
    Ok(output)
}

fn parse_objects(output: &[u8]) -> Result<HashMap<String, (String, Vec<u8>)>> {
    let mut position = 0;
    let mut objects = HashMap::new();
    while position < output.len() {
        let end = output[position..]
            .iter()
            .position(|b| *b == b'\n')
            .map(|p| p + position)
            .ok_or_else(|| malformed("object header terminator"))?;
        let header: Vec<_> = output[position..end].split(|b| *b == b' ').collect();
        if header.len() == 2 && header[1] == b"missing" {
            return Err(missing(&ascii(header[0])?));
        }
        if header.len() != 3 {
            return Err(malformed("object header"));
        }
        let oid = parse_oid(header[0])?;
        let kind = ascii(header[1])?;
        let size: usize = ascii(header[2])?
            .parse()
            .map_err(|_| malformed("object size"))?;
        position = end + 1;
        let content_end = position
            .checked_add(size)
            .filter(|end| *end < output.len())
            .ok_or_else(|| malformed("truncated object content"))?;
        if output[content_end] != b'\n' {
            return Err(malformed("object content terminator"));
        }
        objects.insert(oid, (kind, output[position..content_end].to_vec()));
        position = content_end + 1;
    }
    Ok(objects)
}

fn parse_commit(oid: &str, bytes: &[u8]) -> Result<CommitRecord> {
    let separator = bytes
        .windows(2)
        .position(|w| w == b"\n\n")
        .ok_or_else(|| malformed("commit header"))?;
    let mut tree = None;
    let mut parents = Vec::new();
    let mut author = None;
    let mut committer = None;
    for line in bytes[..separator].split(|b| *b == b'\n') {
        if let Some(value) = line.strip_prefix(b"tree ") {
            tree = Some(parse_oid(value)?);
        } else if let Some(value) = line.strip_prefix(b"parent ") {
            parents.push(parse_oid(value)?);
        } else if let Some(value) = line.strip_prefix(b"author ") {
            author = Some(parse_actor(value)?);
        } else if let Some(value) = line.strip_prefix(b"committer ") {
            committer = Some(parse_actor(value)?);
        }
    }
    let (author_name, author_email, authored_at) =
        author.ok_or_else(|| malformed("commit author"))?;
    let (_, _, committed_at) = committer.ok_or_else(|| malformed("commit committer"))?;
    Ok(CommitRecord {
        oid: oid.into(),
        tree_oid: tree.ok_or_else(|| malformed("commit tree"))?,
        parents,
        author_name,
        author_email,
        authored_at,
        committed_at,
        message: String::from_utf8_lossy(&bytes[separator + 2..])
            .trim_end_matches('\n')
            .into(),
    })
}

fn parse_actor(bytes: &[u8]) -> Result<(String, String, i64)> {
    let text = String::from_utf8_lossy(bytes);
    let (identity_and_time, _) = text
        .rsplit_once(' ')
        .ok_or_else(|| malformed("actor timezone"))?;
    let (identity, timestamp) = identity_and_time
        .rsplit_once(' ')
        .ok_or_else(|| malformed("actor time"))?;
    let timestamp = timestamp
        .parse()
        .map_err(|_| malformed("actor timestamp"))?;
    let (name, email) = identity
        .rsplit_once(" <")
        .ok_or_else(|| malformed("actor identity"))?;
    Ok((
        name.into(),
        email
            .strip_suffix('>')
            .ok_or_else(|| malformed("actor email"))?
            .into(),
        timestamp,
    ))
}

fn parse_raw_changes(output: &[u8]) -> Result<Vec<RawChange>> {
    if !output.is_empty() && output.last() != Some(&0) {
        return Err(malformed("raw diff terminator"));
    }
    let mut fields = output.split(|b| *b == 0);
    let mut commit = None;
    let mut result = Vec::new();
    while let Some(field) = fields.next() {
        if field.is_empty() {
            continue;
        }
        if !field.starts_with(b":") {
            commit = Some(parse_oid(field)?);
            continue;
        }
        let header: Vec<_> = field[1..].split(|b| *b == b' ').collect();
        if header.len() != 5 {
            return Err(malformed("raw diff header"));
        }
        let status = *header[4]
            .first()
            .ok_or_else(|| malformed("raw diff status"))?;
        let first = fields.next().ok_or_else(|| malformed("raw diff path"))?;
        let second = if status == b'R' || status == b'C' {
            fields
                .next()
                .ok_or_else(|| malformed("raw rename destination"))?
        } else {
            first
        };
        if first.is_empty() || second.is_empty() {
            return Err(malformed("empty raw diff path"));
        }
        let oid = |b: &[u8]| -> Result<Option<String>> {
            if b.iter().all(|x| *x == b'0') {
                Ok(None)
            } else {
                Ok(Some(parse_oid(b)?))
            }
        };
        result.push(RawChange {
            commit_oid: commit.clone().ok_or_else(|| malformed("raw diff commit"))?,
            old_mode: ascii(header[0])?,
            new_mode: ascii(header[1])?,
            old_blob: oid(header[2])?,
            new_blob: oid(header[3])?,
            old_path: (status != b'A').then(|| GitPath::from_bytes(first)),
            new_path: (status != b'D').then(|| GitPath::from_bytes(second)),
        });
    }
    Ok(result)
}

fn selected_parent(commit: &CommitRecord, index: usize) -> Result<Option<String>> {
    if index == 0 || index > commit.parents.len().max(1) {
        return Err(AppError::invalid(format!(
            "Commit {} has {} parents; parent {index} is invalid",
            commit.oid,
            commit.parents.len()
        )));
    }
    Ok(commit.parents.get(index - 1).cloned())
}

#[derive(Debug)]
struct Hunk {
    old_start: u32,
    old_lines: u32,
    new_start: u32,
    new_lines: u32,
    text: String,
}

fn source_hunks<'a>(diff: &'a TextDiff<'a, 'a, 'a, str>, context: usize) -> Result<Vec<Hunk>> {
    let mut unified = diff.unified_diff();
    unified.context_radius(context);
    let mut hunks = split_hunks(&unified.to_string())?;
    for (hunk, source) in hunks.iter_mut().zip(unified.iter_hunks()) {
        // A deletion consumes no new lines and an insertion consumes no old
        // lines. Their unused indices can be stale after diff compaction; use
        // the indices of actual source lines to make the evidence coordinates.
        let mut ranges = [None, None];
        for change in source.iter_changes() {
            for (index, range) in [change.old_index(), change.new_index()]
                .into_iter()
                .zip(&mut ranges)
            {
                if let Some(index) = index {
                    let (_, count) = range.get_or_insert((index as u32 + 1, 0));
                    *count += 1;
                }
            }
        }
        let (old_start, old_lines) = ranges[0].unwrap_or((hunk.old_start, hunk.old_lines));
        let (new_start, new_lines) = ranges[1].unwrap_or((hunk.new_start, hunk.new_lines));
        if (old_start, old_lines, new_start, new_lines)
            != (
                hunk.old_start,
                hunk.old_lines,
                hunk.new_start,
                hunk.new_lines,
            )
        {
            let (_, body) = hunk
                .text
                .split_once('\n')
                .ok_or_else(|| malformed("diff hunk header"))?;
            hunk.text = format!("@@ -{old_start},{old_lines} +{new_start},{new_lines} @@\n{body}");
            hunk.old_start = old_start;
            hunk.old_lines = old_lines;
            hunk.new_start = new_start;
            hunk.new_lines = new_lines;
        }
    }
    Ok(hunks)
}

fn split_hunks(patch: &str) -> Result<Vec<Hunk>> {
    let mut hunks = Vec::new();
    for line in patch.split_inclusive('\n') {
        if line.starts_with("@@ ") {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() < 4 {
                return Err(malformed("diff hunk header"));
            }
            let (old_start, old_lines) = parse_range(fields[1], '-')?;
            let (new_start, new_lines) = parse_range(fields[2], '+')?;
            hunks.push(Hunk {
                old_start,
                old_lines,
                new_start,
                new_lines,
                text: line.into(),
            });
        } else if let Some(last) = hunks.last_mut() {
            last.text.push_str(line);
        } else if !line.is_empty() {
            return Err(malformed("unified diff before first hunk"));
        }
    }
    Ok(hunks)
}

fn parse_range(value: &str, sign: char) -> Result<(u32, u32)> {
    let value = value
        .strip_prefix(sign)
        .ok_or_else(|| malformed("hunk range sign"))?;
    let (start, count) = value.split_once(',').unwrap_or((value, "1"));
    Ok((
        start.parse().map_err(|_| malformed("hunk range start"))?,
        count.parse().map_err(|_| malformed("hunk range count"))?,
    ))
}

fn split_hunk(hunk: &Hunk, limit: usize) -> Result<Vec<Hunk>> {
    if hunk.text.len() <= limit {
        return Ok(vec![Hunk {
            old_start: hunk.old_start,
            old_lines: hunk.old_lines,
            new_start: hunk.new_start,
            new_lines: hunk.new_lines,
            text: hunk.text.clone(),
        }]);
    }
    let mut old = if hunk.old_lines == 0 {
        hunk.old_start
    } else {
        hunk.old_start.saturating_sub(1)
    };
    let mut new = if hunk.new_lines == 0 {
        hunk.new_start
    } else {
        hunk.new_start.saturating_sub(1)
    };
    let mut start_old = old;
    let mut start_new = new;
    let mut body = String::new();
    let mut changed = false;
    let mut chunks = Vec::new();
    for line in hunk.text.split_inclusive('\n').skip(1) {
        if !body.is_empty()
            && line.as_bytes().first() != Some(&b'\\')
            && body.len().saturating_add(line.len()).saturating_add(80) > limit
        {
            if changed {
                chunks.push(hunk_from_body(
                    start_old,
                    old,
                    start_new,
                    new,
                    std::mem::take(&mut body),
                ));
            }
            body.clear();
            changed = false;
            start_old = old;
            start_new = new;
        }
        match line.as_bytes().first() {
            Some(b' ') => {
                old += 1;
                new += 1;
            }
            Some(b'-') => {
                old += 1;
                changed = true;
            }
            Some(b'+') => {
                new += 1;
                changed = true;
            }
            Some(b'\\') => {}
            _ => return Err(malformed("hunk line prefix")),
        }
        body.push_str(line);
    }
    if changed {
        chunks.push(hunk_from_body(start_old, old, start_new, new, body));
    }
    Ok(chunks)
}

fn hunk_from_body(
    old_begin: u32,
    old_end: u32,
    new_begin: u32,
    new_end: u32,
    body: String,
) -> Hunk {
    let old_lines = old_end - old_begin;
    let new_lines = new_end - new_begin;
    let old_start = old_begin + u32::from(old_lines > 0);
    let new_start = new_begin + u32::from(new_lines > 0);
    Hunk {
        old_start,
        old_lines,
        new_start,
        new_lines,
        text: format!("@@ -{old_start},{old_lines} +{new_start},{new_lines} @@\n{body}"),
    }
}

fn file_prefix(commit: &CommitRecord, change: &RawChange) -> String {
    let (title, _) = truncate_utf8(commit.message.lines().next().unwrap_or(""), 512);
    let modes = match (change.old_mode.as_str(), change.new_mode.as_str()) {
        ("000000", new) => format!("new file mode {new}\n"),
        (old, "000000") => format!("deleted file mode {old}\n"),
        (old, new) if old != new => format!("old mode {old}\nnew mode {new}\n"),
        _ => String::new(),
    };
    format!(
        "{}\n{modes}--- {}\n+++ {}\n",
        title,
        change.old_path.as_ref().map_or("/dev/null", |p| &p.display),
        change.new_path.as_ref().map_or("/dev/null", |p| &p.display)
    )
}

#[allow(clippy::too_many_arguments)]
fn make_document(
    commit: &CommitRecord,
    parent_oid: Option<String>,
    parent_index: usize,
    kind: DocumentKind,
    change: Option<&RawChange>,
    range: (u32, u32, u32, u32),
    text: String,
    truncated: bool,
) -> Result<DocumentRecord> {
    let mut doc = DocumentRecord {
        id: String::new(),
        commit_oid: commit.oid.clone(),
        parent_oid,
        parent_index,
        kind,
        old_path: change.and_then(|c| c.old_path.clone()),
        new_path: change.and_then(|c| c.new_path.clone()),
        old_blob: change.and_then(|c| c.old_blob.clone()),
        new_blob: change.and_then(|c| c.new_blob.clone()),
        old_start: range.0,
        old_lines: range.1,
        new_start: range.2,
        new_lines: range.3,
        content_hash: digest(text.as_bytes()),
        text,
        truncated,
    };
    doc.id = format!("e1:{}", digest(&serde_json::to_vec(&doc)?));
    Ok(doc)
}

fn verify_hunk_document(
    doc: &DocumentRecord,
    commit: &CommitRecord,
    change: &RawChange,
    old: &str,
    new: &str,
) -> bool {
    let prefix = file_prefix(commit, change);
    let Some(patch) = doc.text.strip_prefix(&prefix) else {
        return false;
    };
    let Ok(hunks) = split_hunks(patch) else {
        return false;
    };
    if hunks.len() != 1 {
        return false;
    }
    let hunk = &hunks[0];
    if (
        hunk.old_start,
        hunk.old_lines,
        hunk.new_start,
        hunk.new_lines,
    ) != (doc.old_start, doc.old_lines, doc.new_start, doc.new_lines)
    {
        return false;
    }
    let old_lines: Vec<_> = old.split_inclusive('\n').collect();
    let new_lines: Vec<_> = new.split_inclusive('\n').collect();
    let mut oi = if doc.old_lines == 0 {
        doc.old_start
    } else {
        doc.old_start.saturating_sub(1)
    } as usize;
    let mut ni = if doc.new_lines == 0 {
        doc.new_start
    } else {
        doc.new_start.saturating_sub(1)
    } as usize;
    if oi > old_lines.len() || ni > new_lines.len() {
        return false;
    }
    let (old_begin, new_begin) = (oi, ni);
    let lines: Vec<_> = hunk.text.split_inclusive('\n').skip(1).collect();
    for (i, line) in lines.iter().enumerate() {
        let Some((&tag, content)) = line.as_bytes().split_first() else {
            return false;
        };
        if tag == b'\\' {
            continue;
        }
        let Ok(mut expected) = std::str::from_utf8(content) else {
            return false;
        };
        if lines
            .get(i + 1)
            .is_some_and(|s| s.starts_with("\\ No newline"))
        {
            expected = expected.strip_suffix('\n').unwrap_or(expected);
        }
        if tag == b' ' || tag == b'-' {
            if old_lines.get(oi).copied() != Some(expected) {
                return false;
            }
            oi += 1;
        }
        if tag == b' ' || tag == b'+' {
            if new_lines.get(ni).copied() != Some(expected) {
                return false;
            }
            ni += 1;
        }
        if !matches!(tag, b' ' | b'-' | b'+') {
            return false;
        }
    }
    oi - old_begin == doc.old_lines as usize && ni - new_begin == doc.new_lines as usize
}

fn blob_text<'a>(
    oid: Option<&String>,
    blobs: &'a HashMap<String, (String, Vec<u8>)>,
) -> Option<&'a str> {
    let Some(oid) = oid else {
        return Some("");
    };
    let (kind, bytes) = blobs.get(oid)?;
    if kind != "blob" || bytes.contains(&0) {
        return None;
    }
    std::str::from_utf8(bytes).ok()
}

fn exclusion_globs(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(
            Glob::new(pattern).map_err(|e| {
                AppError::invalid(format!("Invalid exclusion glob {pattern:?}: {e}"))
            })?,
        );
    }
    builder
        .build()
        .map_err(|e| AppError::invalid(format!("Invalid exclusions: {e}")))
}

fn excluded(change: &RawChange, patterns: &GlobSet) -> Result<bool> {
    for path in change.old_path.iter().chain(change.new_path.iter()) {
        if patterns.is_match(Path::new(OsStr::from_bytes(&path.bytes()?))) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn scope_signature(
    roots: &[ResolvedRoot],
    excluded: &[ResolvedRoot],
    shallow: &[u8],
    first_parent: bool,
) -> Result<String> {
    Ok(digest(&serde_json::to_vec(&(
        roots,
        excluded,
        hex(shallow),
        first_parent,
    ))?))
}

fn reject_grafts(common_dir: &Path) -> Result<()> {
    match std::fs::read(common_dir.join("info/grafts")) {
        Ok(bytes)
            if bytes.split(|b| *b == b'\n').any(|line| {
                let line = trim_ascii(line);
                !line.is_empty() && !line.starts_with(b"#")
            }) =>
        {
            Err(AppError::new(
                "unsupported_repository",
                "Legacy Git grafts change object ancestry; remove or migrate them before indexing",
            ))
        }
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn unhex(value: &str) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2) {
        return Err(malformed("hex path length"));
    }
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| {
            let a = (p[0] as char).to_digit(16);
            let b = (p[1] as char).to_digit(16);
            match (a, b) {
                (Some(a), Some(b)) => Ok((a * 16 + b) as u8),
                _ => Err(malformed("hex path byte")),
            }
        })
        .collect()
}
fn parse_oid(bytes: &[u8]) -> Result<String> {
    if !matches!(bytes.len(), 40 | 64) || !bytes.iter().all(u8::is_ascii_hexdigit) {
        return Err(malformed("object ID"));
    }
    Ok(ascii(bytes)?.to_lowercase())
}
fn ascii(bytes: &[u8]) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| malformed("ASCII Git metadata"))
}
fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(start, |i| i + 1);
    &bytes[start..end]
}
fn path_output(mut bytes: Vec<u8>) -> Result<PathBuf> {
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.is_empty() {
        return Err(malformed("repository path"));
    }
    Ok(PathBuf::from(OsString::from_vec(bytes)))
}
fn text_output(bytes: Vec<u8>) -> Result<String> {
    ascii(trim_ascii(&bytes))
}
fn truncate_utf8(text: &str, limit: usize) -> (String, bool) {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].into(), end != text.len())
}
fn malformed(what: &str) -> AppError {
    AppError::new(
        "invalid_git_data",
        format!("Malformed or truncated {what} from Git"),
    )
}
fn missing(oid: &str) -> AppError {
    AppError::new(
        "missing_object",
        format!("Git object {oid} is unavailable locally; no fetch was attempted"),
    )
}
fn git_error(operation: &str, stderr: &[u8]) -> AppError {
    AppError::new(
        "git_error",
        format!(
            "Git could not {operation}: {}",
            String::from_utf8_lossy(stderr).trim()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture(tempfile::TempDir);

    impl Fixture {
        fn new() -> Self {
            let fixture = Self(tempfile::tempdir().unwrap());
            fixture.git(&["init", "--quiet", "--initial-branch=main"], None);
            fixture
        }

        fn git(&self, args: &[&str], input: Option<&[u8]>) -> String {
            fixture_git(self.0.path(), args, input)
        }

        fn commit(&self, files: &[(&[u8], &str)], parents: &[&str], message: &str) -> String {
            let mut entries = Vec::new();
            for (name, contents) in files {
                let blob = self.git(&["hash-object", "-w", "--stdin"], Some(contents.as_bytes()));
                entries.extend_from_slice(format!("100644 blob {blob}\t").as_bytes());
                entries.extend_from_slice(name);
                entries.push(0);
            }
            let tree = self.git(&["mktree", "-z"], Some(&entries));
            let mut args = vec!["commit-tree", tree.as_str()];
            for parent in parents {
                args.extend(["-p", parent]);
            }
            self.git(&args, Some(message.as_bytes()))
        }
    }

    fn fixture_git(path: &Path, args: &[&str], input: Option<&[u8]>) -> String {
        let output = fixture_git_output(path, args, input);
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    fn fixture_git_output(
        path: &Path,
        args: &[&str],
        input: Option<&[u8]>,
    ) -> std::process::Output {
        let mut command = Command::new("git");
        command
            .current_dir(path)
            .args(args)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_NO_LAZY_FETCH", "0")
            .env("GIT_AUTHOR_NAME", "NaLCoS Test")
            .env("GIT_AUTHOR_EMAIL", "test@example.invalid")
            .env("GIT_COMMITTER_NAME", "NaLCoS Test")
            .env("GIT_COMMITTER_EMAIL", "test@example.invalid")
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for name in [
            "GIT_DIR",
            "GIT_COMMON_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_NAMESPACE",
            "GIT_SHALLOW_FILE",
            "GIT_GRAFT_FILE",
        ] {
            command.env_remove(name);
        }
        let mut child = command.spawn().unwrap();
        if let Some(input) = input {
            child.stdin.take().unwrap().write_all(input).unwrap();
        }
        child.wait_with_output().unwrap()
    }

    #[test]
    fn document_pathspecs_keep_matching_rename_evidence_and_exclude_other_changed_files() {
        let fixture = Fixture::new();
        let old = b"old\nname\xff".as_slice();
        let new = b"new\tname".as_slice();
        let base = fixture.commit(
            &[(old, "original\n"), (b"unrelated", "before\n")],
            &[],
            "base",
        );
        let tip = fixture.commit(
            &[(new, "original\n"), (b"unrelated", "needle\n")],
            &[&base],
            "rename and update another file",
        );
        let execution = Execution::unlimited();
        let repository = Repository::discover(fixture.0.path(), &execution).unwrap();
        let records = repository.load_commits(&[tip], &execution).unwrap();
        let extracted = repository
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();
        let paths: Vec<_> = extracted
            .documents
            .iter()
            .map(DocumentPaths::from)
            .collect();
        let expected: HashSet<_> = paths
            .iter()
            .filter(|document| {
                document.kind == DocumentKind::Message
                    || document
                        .new_path
                        .as_ref()
                        .is_some_and(|path| path.bytes().unwrap() == new)
            })
            .map(|document| document.id.clone())
            .collect();
        assert_eq!(expected.len(), 2);
        for pathspecs in [
            vec!["old*".into()],
            vec![":(literal)new\tname".into()],
            vec!["*".into(), ":(exclude)unrelated".into()],
        ] {
            assert_eq!(
                repository
                    .matching_document_ids(&paths, &pathspecs, &execution)
                    .unwrap(),
                expected
            );
        }
        let only_other = repository
            .matching_document_ids(&paths, &["unrelated".into()], &execution)
            .unwrap();
        assert!(
            paths
                .iter()
                .filter(|document| only_other.contains(&document.id))
                .all(|document| {
                    document.kind == DocumentKind::Message
                        || document.new_path.as_ref().unwrap().bytes().unwrap() == b"unrelated"
                })
        );
        assert!(
            extracted
                .documents
                .iter()
                .any(|document| only_other.contains(&document.id)
                    && document.text.contains("+needle"))
        );
        assert_eq!(
            repository
                .matching_document_ids(&paths, &[], &execution)
                .unwrap()
                .len(),
            paths.len()
        );
        assert!(
            repository
                .matching_document_ids(&paths, &["absent".into()], &execution)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn merge_parents_rename_filters_and_cached_evidence_use_exact_source_trees() {
        let fixture = Fixture::new();
        let old = b"old\nname\xff".as_slice();
        let new = b"new\tname".as_slice();
        let base = fixture.commit(&[(old, "original\n")], &[], "base");
        let left = fixture.commit(&[(new, "original\n")], &[&base], "rename");
        let right = fixture.commit(
            &[(old, "original\n"), (b"side", "feature\n")],
            &[&base],
            "side",
        );
        let merge = fixture.commit(
            &[(new, "original\n"), (b"side", "feature\n")],
            &[&left, &right],
            "merge",
        );
        fixture.git(&["update-ref", "refs/heads/main", &merge], None);
        let execution = Execution::unlimited();
        let repository = Repository::discover(fixture.0.path(), &execution).unwrap();
        let scope = repository
            .resolve_scope(&ScopeOptions::default(), &execution)
            .unwrap();
        assert_eq!(scope.oids.len(), 4);
        let first_parent = repository
            .resolve_scope(
                &ScopeOptions {
                    first_parent: true,
                    ..Default::default()
                },
                &execution,
            )
            .unwrap();
        assert_eq!(first_parent.oids, [&merge, &left, &base].map(String::from));
        let records = repository
            .load_commits(std::slice::from_ref(&merge), &execution)
            .unwrap();
        assert_eq!(records[0].parents, [&left, &right].map(String::from));
        let extracted = repository
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();
        assert!(extracted.omissions.is_empty());
        assert!(
            extracted
                .documents
                .iter()
                .filter(|d| d.kind == DocumentKind::Diff)
                .all(|d| d.new_path.as_ref().unwrap().bytes().unwrap() == b"side")
        );
        let verified = repository
            .validate_documents(&extracted.documents, &execution)
            .unwrap();
        assert_eq!(verified.valid_ids.len(), extracted.documents.len());

        let shown = repository
            .show(
                &merge,
                &ShowOptions {
                    parent: 2,
                    paths: vec!["old*".into()],
                    ..Default::default()
                },
                &execution,
            )
            .unwrap();
        assert_eq!(shown.parent_oid.as_deref(), Some(right.as_str()));
        assert_eq!(shown.documents.len(), 1);
        let rename = &shown.documents[0];
        assert_eq!(rename.old_path.as_ref().unwrap().bytes().unwrap(), old);
        assert_eq!(rename.new_path.as_ref().unwrap().bytes().unwrap(), new);
        assert_eq!(
            repository
                .matching_document_ids(&[DocumentPaths::from(rename)], &["old*".into()], &execution)
                .unwrap(),
            HashSet::from([rename.id.clone()])
        );
        assert_eq!(
            repository
                .validate_documents(&shown.documents, &execution)
                .unwrap()
                .valid_ids,
            vec![rename.id.clone()]
        );
        let matching = repository
            .filter_scope(
                &scope,
                &FilterOptions {
                    paths: vec!["old*".into()],
                    ..Default::default()
                },
                &execution,
            )
            .unwrap();
        assert!(matching.contains(&left));
        assert!(!matching.contains(&merge));

        let mut forged = rename.clone();
        forged.old_path = Some(GitPath::from_bytes(b"nonexistent"));
        assert_eq!(
            repository
                .validate_documents(&[forged.clone()], &execution)
                .unwrap()
                .missing_ids,
            vec![forged.id.clone()]
        );
        forged.parent_index = 99;
        assert_eq!(
            repository
                .validate_documents(&[forged.clone()], &execution)
                .unwrap()
                .missing_ids,
            vec![forged.id]
        );

        let executable_tree = fixture.git(
            &["mktree", "-z"],
            Some(
                fixture
                    .git(&["ls-tree", "-z", &merge], None)
                    .replace("100644", "100755")
                    .as_bytes(),
            ),
        );
        let executable = fixture.git(
            &["commit-tree", &executable_tree, "-p", &merge],
            Some(b"make executable"),
        );
        let mode_change = repository
            .show(&executable, &ShowOptions::default(), &execution)
            .unwrap();
        assert!(
            mode_change
                .patch
                .contains("old mode 100644\nnew mode 100755\n")
        );
        assert_eq!(
            repository
                .validate_documents(&mode_change.documents, &execution)
                .unwrap()
                .valid_ids
                .len(),
            mode_change.documents.len()
        );

        let all_refs = ScopeOptions {
            all_refs: true,
            ..Default::default()
        };
        fixture.git(
            &["update-ref", "refs/archive/saved-change", &executable],
            None,
        );
        let with_custom_ref = repository.resolve_scope(&all_refs, &execution).unwrap();
        assert!(with_custom_ref.oids.contains(&executable));
        assert!(
            with_custom_ref
                .roots
                .iter()
                .any(|root| root.name == "refs/archive/saved-change" && root.oid == executable)
        );
        fixture.git(&["update-ref", "-d", "refs/archive/saved-change"], None);
        assert!(
            !repository
                .validate_snapshot(&all_refs, &with_custom_ref, &execution)
                .unwrap()
        );
        assert!(
            !repository
                .resolve_scope(&all_refs, &execution)
                .unwrap()
                .oids
                .contains(&executable)
        );

        let alternate = fixture.commit(&[(b"unrelated", "replacement\n")], &[], "replacement");
        fixture.git(
            &["update-ref", &format!("refs/replace/{base}"), &alternate],
            None,
        );
        assert_eq!(
            repository
                .resolve_scope(&ScopeOptions::default(), &execution)
                .unwrap()
                .oids,
            scope.oids
        );
        assert_eq!(
            repository.load_commits(&[base], &execution).unwrap()[0].message,
            "base"
        );
    }

    #[test]
    fn shallow_boundaries_omit_missing_parents_and_invalidate_when_deepened() {
        let fixture = Fixture::new();
        let base = fixture.commit(&[(b"file", "old\n")], &[], "base");
        let tip = fixture.commit(&[(b"file", "new\n")], &[&base], "tip");
        fixture.git(&["update-ref", "refs/heads/main", &tip], None);
        let execution = Execution::unlimited();
        let source = Repository::discover(fixture.0.path(), &execution).unwrap();
        let records = source
            .load_commits(std::slice::from_ref(&tip), &execution)
            .unwrap();
        let complete = source
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();
        let destination = tempfile::tempdir().unwrap();
        fixture.git(
            &[
                "clone",
                "--quiet",
                "--no-checkout",
                "--depth=1",
                &format!("file://{}", fixture.0.path().display()),
                destination.path().to_str().unwrap(),
            ],
            None,
        );
        let shallow = Repository::discover(destination.path(), &execution).unwrap();
        let scope = shallow
            .resolve_scope(&ScopeOptions::default(), &execution)
            .unwrap();
        assert!(scope.shallow);
        assert_eq!(scope.oids.as_slice(), std::slice::from_ref(&tip));
        let extracted = shallow
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();
        assert_eq!(extracted.documents.len(), 1);
        assert_eq!(extracted.documents[0].kind, DocumentKind::Message);
        assert_eq!(extracted.omissions[0].reason, "missing_object");
        let verified = shallow
            .validate_documents(&complete.documents, &execution)
            .unwrap();
        assert_eq!(verified.valid_ids, vec![extracted.documents[0].id.clone()]);
        assert!(!verified.missing_ids.is_empty());

        fixture_git(
            destination.path(),
            &["fetch", "--quiet", "--unshallow"],
            None,
        );
        assert!(
            !shallow
                .validate_snapshot(&ScopeOptions::default(), &scope, &execution)
                .unwrap()
        );
        let extracted = shallow
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();
        assert!(extracted.omissions.is_empty());
        assert_eq!(extracted.documents.len(), complete.documents.len());
        std::fs::write(shallow.common_dir.join("info/grafts"), format!("{tip}\n")).unwrap();
        assert_eq!(
            shallow
                .validate_snapshot(&ScopeOptions::default(), &scope, &execution)
                .unwrap_err()
                .code,
            "unsupported_repository"
        );
    }

    #[test]
    fn missing_promisor_blob_reports_incomplete_evidence_without_attempting_fetch() {
        use std::os::unix::fs::PermissionsExt;

        let fixture = Fixture::new();
        let tip = fixture.commit(&[(b"file", "promised content\n")], &[], "base");
        fixture.git(&["update-ref", "refs/heads/main", &tip], None);
        let blob = fixture.git(&["rev-parse", &format!("{tip}:file")], None);
        let execution = Execution::unlimited();
        let repository = Repository::discover(fixture.0.path(), &execution).unwrap();
        let records = repository.load_commits(&[tip], &execution).unwrap();
        let complete = repository
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();

        let upload_pack = fixture.0.path().join("reject-upload-pack");
        let attempted = fixture.0.path().join("reject-upload-pack.marker");
        std::fs::write(&upload_pack, "#!/bin/sh\n: > \"$0.marker\"\nexit 1\n").unwrap();
        std::fs::set_permissions(&upload_pack, std::fs::Permissions::from_mode(0o755)).unwrap();
        fixture.git(&["config", "remote.origin.promisor", "true"], None);
        fixture.git(
            &[
                "config",
                "remote.origin.url",
                fixture.0.path().to_str().unwrap(),
            ],
            None,
        );
        fixture.git(
            &[
                "config",
                "remote.origin.uploadpack",
                &format!("'{}'", upload_pack.to_str().unwrap().replace('\'', "'\\''")),
            ],
            None,
        );
        std::fs::rename(
            repository
                .common_dir
                .join("objects")
                .join(&blob[..2])
                .join(&blob[2..]),
            fixture.0.path().join("saved-blob"),
        )
        .unwrap();

        assert_eq!(
            repository
                .read_objects(std::slice::from_ref(&blob), 1024, &execution)
                .unwrap_err()
                .code,
            "missing_object"
        );
        let incomplete = repository
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();
        assert!(
            incomplete
                .omissions
                .iter()
                .any(|item| item.reason == "missing_object")
        );
        assert!(
            !repository
                .validate_documents(&complete.documents, &execution)
                .unwrap()
                .missing_ids
                .is_empty()
        );
        assert!(
            !attempted.exists(),
            "guarded source reads attempted a fetch"
        );

        // The rejecting local upload-pack proves this fixture detects an attempted
        // lazy fetch without using a network service or completing object hydration.
        let unguarded = fixture_git_output(
            fixture.0.path(),
            &["-c", "protocol.file.allow=always", "cat-file", "-e", &blob],
            None,
        );
        assert!(!unguarded.status.success());
        assert!(attempted.exists(), "the fetch detector was not exercised");
    }

    #[test]
    fn raw_diff_preserves_rename_paths_with_newlines_and_invalid_utf8() {
        let oid = "a".repeat(40);
        let old = "b".repeat(40);
        let new = "c".repeat(40);
        let mut raw = format!("{oid}\0:100644 100644 {old} {new} R090\0").into_bytes();
        raw.extend_from_slice(b"old\nname\xff\0new\tname\0");
        let changes = parse_raw_changes(&raw).unwrap();
        assert_eq!(
            changes[0].old_path.as_ref().unwrap().bytes().unwrap(),
            b"old\nname\xff"
        );
        assert_eq!(
            changes[0].new_path.as_ref().unwrap().bytes().unwrap(),
            b"new\tname"
        );
    }

    #[test]
    fn batch_objects_use_lengths_not_line_or_nul_delimiters() {
        let oid = "a".repeat(40);
        let content = b"a\n\0b";
        let mut output = format!("{oid} blob {}\n", content.len()).into_bytes();
        output.extend_from_slice(content);
        output.push(b'\n');
        assert_eq!(parse_objects(&output).unwrap()[&oid].1, content);
        output.pop();
        assert!(parse_objects(&output).is_err());
    }

    #[test]
    fn compacted_deletion_uses_consumed_lines_for_verified_hunk_ranges() {
        let fixture = Fixture::new();
        let old = "a\n\nb\n";
        let new = "\nx\n\ny\n\nz\n\nw\n\nv\na\na\n";
        let base = fixture.commit(&[(b"file", old)], &[], "base");
        let tip = fixture.commit(&[(b"file", new)], &[&base], "change");
        let execution = Execution::unlimited();
        let repository = Repository::discover(fixture.0.path(), &execution).unwrap();
        let records = repository.load_commits(&[tip], &execution).unwrap();
        let extracted = repository
            .extract_documents(&records, &IndexOptions::default(), &execution)
            .unwrap();
        let hunks: Vec<_> = extracted
            .documents
            .iter()
            .filter(|doc| doc.old_lines != 0 || doc.new_lines != 0)
            .collect();
        assert_eq!(hunks.len(), 1);
        assert_eq!(
            (
                hunks[0].old_start,
                hunks[0].old_lines,
                hunks[0].new_start,
                hunks[0].new_lines
            ),
            (1, 3, 1, 12)
        );
        assert!(hunks[0].text.contains("@@ -1,3 +1,12 @@\n"));
        let verification = repository
            .validate_documents(&extracted.documents, &execution)
            .unwrap();
        assert!(verification.errors.is_empty(), "{:?}", verification.errors);
        assert_eq!(verification.valid_ids.len(), extracted.documents.len());
    }

    #[test]
    fn generated_chunks_verify_against_both_source_sides() {
        let commit = CommitRecord {
            oid: "a".repeat(40),
            tree_oid: "b".repeat(40),
            parents: vec!["c".repeat(40)],
            author_name: "A".into(),
            author_email: "a@b".into(),
            authored_at: 0,
            committed_at: 0,
            message: "Change behavior".into(),
        };
        let change = RawChange {
            commit_oid: commit.oid.clone(),
            old_mode: "100644".into(),
            new_mode: "100644".into(),
            old_blob: Some("d".repeat(40)),
            new_blob: Some("e".repeat(40)),
            old_path: Some(GitPath::from_bytes(b"a.rs")),
            new_path: Some(GitPath::from_bytes(b"a.rs")),
        };
        let old = "a\nb\nc\nd\ne\nf\ng\nh\nlast";
        let new = "a\nB\nc\nd\nE\nf\ng\nh\nLAST";
        let patch = TextDiff::configure()
            .algorithm(Algorithm::Patience)
            .diff_lines(old, new)
            .unified_diff()
            .context_radius(3)
            .to_string();
        for hunk in split_hunks(&patch).unwrap() {
            for chunk in split_hunk(&hunk, 95).unwrap() {
                let doc = make_document(
                    &commit,
                    commit.parents.first().cloned(),
                    1,
                    DocumentKind::Diff,
                    Some(&change),
                    (
                        chunk.old_start,
                        chunk.old_lines,
                        chunk.new_start,
                        chunk.new_lines,
                    ),
                    format!("{}{}", file_prefix(&commit, &change), chunk.text),
                    false,
                )
                .unwrap();
                assert!(
                    verify_hunk_document(&doc, &commit, &change, old, new),
                    "{}",
                    doc.text
                );
                if doc.old_lines > 0 {
                    assert!(!verify_hunk_document(&doc, &commit, &change, "wrong", new));
                }
                if doc.new_lines > 0 {
                    assert!(!verify_hunk_document(&doc, &commit, &change, old, "wrong"));
                }
            }
        }
    }

    #[test]
    fn source_paths_are_distinct_even_when_display_could_be_similar() {
        let bad = GitPath::from_bytes(b"a\xff");
        let literal = GitPath::from_bytes(b"a\\xff");
        assert_ne!(bad.bytes_hex, literal.bytes_hex);
        assert_ne!(bad.display, literal.display);
    }
}
