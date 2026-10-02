//! Durable source documents and independently replaceable embedding generations.
//!
//! Git reachability is deliberately connection-local: every query must install its
//! freshly resolved scope before reading candidates. A cached row is never proof
//! that a commit still belongs to the requested history.

use crate::error::{AppError, Result};
use crate::execution::Execution;
use crate::git::{CommitRecord, DocumentKind, DocumentPaths, DocumentRecord, ExtractionOmission};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    pub id: i64,
    pub fingerprint: String,
    pub model: Value,
    pub dimensions: Option<usize>,
    pub watermark: i64,
    pub state: String,
    pub created_at: i64,
}

#[derive(Debug, Clone)]
pub struct VectorChunk {
    pub byte_start: usize,
    pub byte_end: usize,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub document: DocumentRecord,
    /// Larger is better within a retrieval channel, never a probability.
    pub score: f64,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Debug, Clone, Default)]
pub struct CandidateFilter {
    pub excluded_documents: HashSet<String>,
    /// None admits all documents; an empty set deliberately admits none.
    pub allowed_documents: Option<HashSet<String>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct StoreCounts {
    pub commits: usize,
    pub documents: usize,
    pub omissions: usize,
    pub vectors: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Coverage {
    pub total_commits: usize,
    pub indexed_commits: usize,
    pub documents: usize,
    pub embedded_documents: usize,
    pub omissions: usize,
}

pub struct Store {
    conn: Connection,
    read_only: bool,
    scope_installed: bool,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                AppError::new(
                    "index_error",
                    format!("Create index directory {}: {e}", parent.display()),
                )
            })?;
        }
        let conn = Connection::open(path).map_err(|e| {
            AppError::new("index_error", format!("Open index {}: {e}", path.display()))
        })?;
        let mut store = Self {
            conn,
            read_only: false,
            scope_installed: false,
        };
        store.configure()?;
        store.initialize()?;
        Ok(store)
    }

    pub fn read_only(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let conn =
            Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(|e| {
                AppError::new("index_error", format!("Read index {}: {e}", path.display()))
            })?;
        let mut store = Self {
            conn,
            read_only: true,
            scope_installed: false,
        };
        store.configure()?;
        store.verify_schema()?;
        Ok(store)
    }

    fn configure(&mut self) -> Result<()> {
        self.conn.busy_timeout(Duration::from_millis(250))?;
        self.conn.execute_batch(
            "PRAGMA foreign_keys=ON; PRAGMA temp_store=MEMORY;
             CREATE TEMP TABLE query_scope(oid TEXT PRIMARY KEY, position INTEGER NOT NULL);",
        )?;
        Ok(())
    }

    fn initialize(&mut self) -> Result<()> {
        let version: i64 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version != 0 {
            self.verify_schema()?;
            self.conn.execute_batch("PRAGMA synchronous=NORMAL")?;
            return Ok(());
        }
        self.conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
             BEGIN IMMEDIATE;
             CREATE TABLE commits(oid TEXT PRIMARY KEY, record TEXT NOT NULL);
             CREATE TABLE documents(
               rowid INTEGER PRIMARY KEY AUTOINCREMENT,
               id TEXT NOT NULL UNIQUE,
               commit_oid TEXT NOT NULL REFERENCES commits(oid),
               content_hash TEXT NOT NULL,
               text TEXT NOT NULL,
               record TEXT NOT NULL
             );
             CREATE INDEX documents_commit ON documents(commit_oid);
             CREATE INDEX documents_content ON documents(content_hash);
             CREATE INDEX documents_identity_commit ON documents(id,commit_oid);
             CREATE VIRTUAL TABLE document_fts USING fts5(
               text, content='documents', content_rowid='rowid', tokenize='unicode61'
             );
             CREATE TRIGGER documents_insert AFTER INSERT ON documents BEGIN
               INSERT INTO document_fts(rowid,text) VALUES(new.rowid,new.text);
             END;
             CREATE TRIGGER documents_delete AFTER DELETE ON documents BEGIN
               INSERT INTO document_fts(document_fts,rowid,text) VALUES('delete',old.rowid,old.text);
             END;
             CREATE TABLE omissions(
               commit_oid TEXT NOT NULL REFERENCES commits(oid),
               record TEXT NOT NULL,
               UNIQUE(commit_oid,record)
             );
             CREATE INDEX omissions_commit ON omissions(commit_oid);
             CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE generations(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               fingerprint TEXT NOT NULL,
               model TEXT NOT NULL,
               dimensions INTEGER,
               watermark INTEGER NOT NULL,
               state TEXT NOT NULL CHECK(state IN ('active','staging','retired')),
               created_at INTEGER NOT NULL,
               allow_prior_reuse INTEGER NOT NULL
             );
             CREATE UNIQUE INDEX one_active ON generations(state) WHERE state='active';
             CREATE UNIQUE INDEX one_staging ON generations(state) WHERE state='staging';
             CREATE INDEX generations_fingerprint ON generations(fingerprint);
             CREATE TABLE embedded_documents(
               generation_id INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
               document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
               PRIMARY KEY(generation_id,document_id)
             );
             CREATE TABLE embeddings(
               generation_id INTEGER NOT NULL,
               document_id TEXT NOT NULL,
               chunk_index INTEGER NOT NULL,
               byte_start INTEGER NOT NULL,
               byte_end INTEGER NOT NULL,
               dimensions INTEGER NOT NULL,
               vector BLOB NOT NULL,
               PRIMARY KEY(generation_id,document_id,chunk_index),
               FOREIGN KEY(generation_id,document_id)
                 REFERENCES embedded_documents(generation_id,document_id) ON DELETE CASCADE
             );
             PRAGMA user_version=1;
             COMMIT;",
        )?;
        Ok(())
    }

    /// Add optional lookup indexes and refresh statistics after explicit setup.
    /// Covering metadata lookups avoid reading source JSON during vector scans.
    pub fn optimize(&self, execution: &Execution) -> Result<()> {
        self.require_writable()?;
        self.with_execution(execution, || {
            // Check all tables, allow ANALYZE, and use SQLite's temporary analysis
            // limit. Cached readers must never perform this persistent maintenance.
            self.conn.execute_batch(
                "CREATE INDEX IF NOT EXISTS documents_identity_commit ON documents(id,commit_oid);
                 PRAGMA optimize=0x10012;",
            )?;
            Ok(())
        })
    }

    fn verify_schema(&self) -> Result<()> {
        let version: i64 = self
            .conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version != SCHEMA_VERSION {
            return Err(AppError::new(
                "unsupported_schema",
                format!(
                    "Index schema {version} is not supported; this build requires {SCHEMA_VERSION}"
                ),
            ));
        }
        Ok(())
    }

    fn require_writable(&self) -> Result<()> {
        if self.read_only {
            return Err(AppError::new(
                "index_read_only",
                "This index connection is read-only",
            ));
        }
        if !self.conn.is_autocommit() {
            return Err(AppError::new(
                "index_busy",
                "Finish the query snapshot before writing the index",
            ));
        }
        Ok(())
    }

    fn require_scope(&self) -> Result<()> {
        if !self.scope_installed {
            return Err(AppError::new(
                "scope_required",
                "Resolve and install Git scope before querying the index",
            ));
        }
        Ok(())
    }

    fn with_execution<T>(
        &self,
        execution: &Execution,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        execution.check()?;
        let sql_execution = execution.clone();
        self.conn
            .progress_handler(1_000, Some(move || sql_execution.check().is_err()))?;
        let result = operation();
        let clear_result = self.conn.progress_handler(0, None::<fn() -> bool>);
        // SQLite reports an interrupted statement generically; preserve the actual
        // deadline/cancellation reason exposed by the CLI's shared execution token.
        execution.check()?;
        clear_result?;
        result
    }

    pub fn indexed_oids(&self) -> Result<HashSet<String>> {
        let mut statement = self.conn.prepare("SELECT oid FROM commits")?;
        let rows = statement.query_map([], |r| r.get(0))?;
        Ok(rows.collect::<std::result::Result<HashSet<String>, _>>()?)
    }

    /// Remove only derived cache rows outside the caller's resolved retention set.
    /// The caller must include all local ref/worktree roots and explicitly requested
    /// historical scope; query scope is intentionally left unchanged by maintenance.
    pub fn prune_commits_except(&mut self, retained: &HashSet<String>) -> Result<usize> {
        self.require_writable()?;
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS retained_commits(oid TEXT PRIMARY KEY);
             DELETE FROM retained_commits;",
        )?;
        {
            let mut insert = tx.prepare("INSERT INTO retained_commits(oid) VALUES(?1)")?;
            for oid in retained {
                insert.execute([oid])?;
            }
        }
        tx.execute(
            "DELETE FROM documents WHERE NOT EXISTS(
               SELECT 1 FROM retained_commits r WHERE r.oid=documents.commit_oid
             )",
            [],
        )?;
        tx.execute(
            "DELETE FROM omissions WHERE NOT EXISTS(
               SELECT 1 FROM retained_commits r WHERE r.oid=omissions.commit_oid
             )",
            [],
        )?;
        let removed = tx.execute(
            "DELETE FROM commits WHERE NOT EXISTS(
               SELECT 1 FROM retained_commits r WHERE r.oid=commits.oid
             )",
            [],
        )?;
        tx.commit()?;
        Ok(removed)
    }

    /// Supplied commits and their complete extractions commit together. Unchanged
    /// document IDs retain vectors; policy changes remove no-longer-included sources.
    pub fn ingest(
        &mut self,
        commits: &[CommitRecord],
        documents: &[DocumentRecord],
        omissions: &[ExtractionOmission],
    ) -> Result<()> {
        self.require_writable()?;
        let mut retained = HashMap::<&str, HashSet<&str>>::new();
        for commit in commits {
            retained.insert(commit.oid.as_str(), HashSet::new());
        }
        for document in documents {
            let Some(ids) = retained.get_mut(document.commit_oid.as_str()) else {
                return Err(AppError::new(
                    "invalid_input",
                    format!(
                        "Document {} has no supplied commit {}",
                        document.id, document.commit_oid
                    ),
                ));
            };
            ids.insert(document.id.as_str());
        }
        for omission in omissions {
            if !retained.contains_key(omission.commit_oid.as_str()) {
                return Err(AppError::new(
                    "invalid_input",
                    format!("Omission has no supplied commit {}", omission.commit_oid),
                ));
            }
        }
        let tx = self.conn.transaction()?;
        for commit in commits {
            let record = serde_json::to_string(commit)?;
            tx.execute(
                "INSERT OR IGNORE INTO commits(oid,record) VALUES(?1,?2)",
                params![commit.oid, record],
            )?;
            let existing: String = tx.query_row(
                "SELECT record FROM commits WHERE oid=?1",
                [&commit.oid],
                |r| r.get(0),
            )?;
            if existing != record {
                return Err(AppError::new(
                    "source_conflict",
                    format!(
                        "Cached commit {} differs from its immutable source",
                        commit.oid
                    ),
                ));
            }
            let old_ids = {
                let mut statement = tx.prepare("SELECT id FROM documents WHERE commit_oid=?1")?;
                statement
                    .query_map([&commit.oid], |r| r.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?
            };
            for old_id in old_ids {
                if !retained[commit.oid.as_str()].contains(old_id.as_str()) {
                    tx.execute("DELETE FROM documents WHERE id=?1", [&old_id])?;
                }
            }
            tx.execute("DELETE FROM omissions WHERE commit_oid=?1", [&commit.oid])?;
        }
        for document in documents {
            let record = serde_json::to_string(document)?;
            tx.execute(
                "INSERT OR IGNORE INTO documents(id,commit_oid,content_hash,text,record) VALUES(?1,?2,?3,?4,?5)",
                params![document.id, document.commit_oid, document.content_hash, document.text, record],
            )?;
            let existing: String = tx.query_row(
                "SELECT record FROM documents WHERE id=?1",
                [&document.id],
                |r| r.get(0),
            )?;
            if existing != record {
                return Err(AppError::new(
                    "source_conflict",
                    format!(
                        "Cached evidence {} differs from its immutable source",
                        document.id
                    ),
                ));
            }
        }
        for omission in omissions {
            tx.execute(
                "INSERT OR IGNORE INTO omissions(commit_oid,record) VALUES(?1,?2)",
                params![omission.commit_oid, serde_json::to_string(omission)?],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn metadata(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM metadata WHERE key=?1", [key], |r| {
                r.get(0)
            })
            .optional()?)
    }

    pub fn set_metadata(&mut self, key: &str, value: &str) -> Result<()> {
        self.require_writable()?;
        self.conn.execute(
            "INSERT INTO metadata(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key,value],
        )?;
        Ok(())
    }

    /// Install eligibility before ranking. Empty scope must produce empty results.
    pub fn set_scope(&mut self, oids: &[String]) -> Result<()> {
        if !self.conn.is_autocommit() {
            return Err(AppError::new(
                "index_busy",
                "Finish the query snapshot before replacing its scope",
            ));
        }
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM query_scope", [])?;
        {
            let mut insert =
                tx.prepare("INSERT OR IGNORE INTO query_scope(oid,position) VALUES(?1,?2)")?;
            for (position, oid) in oids.iter().enumerate() {
                insert.execute(params![oid, to_i64(position)?])?;
            }
        }
        tx.commit()?;
        self.scope_installed = true;
        Ok(())
    }

    /// Keep model metadata and vector reads in one WAL snapshot across generation cutover.
    pub fn begin_snapshot(&mut self) -> Result<()> {
        self.require_scope()?;
        if !self.conn.is_autocommit() {
            return Err(AppError::new(
                "index_busy",
                "A query snapshot is already open",
            ));
        }
        self.conn.execute_batch("BEGIN DEFERRED")?;
        // BEGIN alone does not establish a SQLite read snapshot.
        self.conn
            .query_row("SELECT COUNT(*) FROM generations", [], |r| {
                r.get::<_, i64>(0)
            })?;
        Ok(())
    }

    pub fn end_snapshot(&mut self) -> Result<()> {
        if !self.conn.is_autocommit() {
            self.conn.execute_batch("COMMIT")?;
        }
        Ok(())
    }

    pub fn get_document(&self, id: &str) -> Result<Option<DocumentRecord>> {
        self.conn
            .query_row("SELECT record FROM documents WHERE id=?1", [id], |r| {
                r.get::<_, String>(0)
            })
            .optional()?
            .map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    pub fn get_commit(&self, oid: &str) -> Result<Option<CommitRecord>> {
        self.conn
            .query_row("SELECT record FROM commits WHERE oid=?1", [oid], |r| {
                r.get::<_, String>(0)
            })
            .optional()?
            .map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    pub fn counts(&self) -> Result<StoreCounts> {
        Ok(StoreCounts {
            commits: self.count("SELECT COUNT(*) FROM commits")?,
            documents: self.count("SELECT COUNT(*) FROM documents")?,
            omissions: self.count("SELECT COUNT(*) FROM omissions")?,
            vectors: self.count("SELECT COUNT(*) FROM embeddings")?,
        })
    }

    fn count(&self, sql: &str) -> Result<usize> {
        let count: i64 = self.conn.query_row(sql, [], |r| r.get(0))?;
        to_usize(count)
    }

    pub fn scope_coverage(&self) -> Result<Coverage> {
        self.require_scope()?;
        let generation = self.active_generation()?.map(|g| g.id);
        let embedded: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM embedded_documents e
             JOIN documents d ON d.id=e.document_id JOIN query_scope s ON s.oid=d.commit_oid
             WHERE e.generation_id=?1",
            [generation],
            |r| r.get(0),
        )?;
        Ok(Coverage {
            total_commits: self.count("SELECT COUNT(*) FROM query_scope")?,
            indexed_commits: self
                .count("SELECT COUNT(*) FROM commits c JOIN query_scope s ON s.oid=c.oid")?,
            documents: self.count(
                "SELECT COUNT(*) FROM documents d JOIN query_scope s ON s.oid=d.commit_oid",
            )?,
            embedded_documents: to_usize(embedded)?,
            omissions: self.count(
                "SELECT COUNT(*) FROM omissions o JOIN query_scope s ON s.oid=o.commit_oid",
            )?,
        })
    }

    pub fn scoped_omissions(&self) -> Result<Vec<ExtractionOmission>> {
        self.require_scope()?;
        let mut statement = self.conn.prepare(
            "SELECT o.record FROM omissions o JOIN query_scope s ON s.oid=o.commit_oid ORDER BY s.position,o.rowid",
        )?;
        let rows = statement.query_map([], |r| r.get::<_, String>(0))?;
        rows.map(|row| serde_json::from_str(&row?).map_err(Into::into))
            .collect()
    }

    /// Read path-selection metadata without materializing source text.
    pub fn document_paths_in_scope(&self, execution: &Execution) -> Result<Vec<DocumentPaths>> {
        self.with_execution(execution, || {
            self.require_scope()?;
            let mut statement = self.conn.prepare(
                "SELECT d.id,d.commit_oid,json_extract(d.record,'$.parent_index'),
                        json_extract(d.record,'$.kind'),json_extract(d.record,'$.old_path'),
                        json_extract(d.record,'$.new_path')
                 FROM documents d JOIN query_scope s ON s.oid=d.commit_oid ORDER BY d.rowid",
            )?;
            let mut rows = statement.query([])?;
            let mut documents = Vec::new();
            while let Some(row) = rows.next()? {
                execution.check()?;
                let kind: String = row.get(3)?;
                let kind = match kind.as_str() {
                    "message" => DocumentKind::Message,
                    "diff" => DocumentKind::Diff,
                    _ => {
                        return Err(AppError::new(
                            "invalid_data",
                            format!("Unknown indexed document kind {kind}"),
                        ));
                    }
                };
                let old_path: Option<String> = row.get(4)?;
                let new_path: Option<String> = row.get(5)?;
                documents.push(DocumentPaths {
                    id: row.get(0)?,
                    commit_oid: row.get(1)?,
                    parent_index: to_usize(row.get(2)?)?,
                    kind,
                    old_path: old_path.map(|s| serde_json::from_str(&s)).transpose()?,
                    new_path: new_path.map(|s| serde_json::from_str(&s)).transpose()?,
                });
            }
            Ok(documents)
        })
    }

    /// FTS syntax is not part of the CLI contract. Quote every word and OR the terms.
    pub fn lexical_candidates(
        &self,
        query: &str,
        limit: usize,
        filter: &CandidateFilter,
        execution: &Execution,
    ) -> Result<Vec<Candidate>> {
        self.with_execution(execution, || {
            self.require_scope()?;
            let expression = lexical_expression(query);
            if expression.is_empty() || limit == 0 {
                return Ok(Vec::new());
            }
            let (allowed, excluded) = filter_json(filter)?;
            let mut statement = self.conn.prepare(
                "SELECT d.id,d.commit_oid,-document_fts.rank AS score
                 FROM document_fts JOIN documents d ON d.rowid=document_fts.rowid
                 JOIN query_scope s ON s.oid=d.commit_oid
                 WHERE document_fts MATCH ?1
                   AND document_fts.rank MATCH 'bm25()'
                   AND (?2 IS NULL OR d.id IN (SELECT value FROM json_each(?2)))
                   AND d.id NOT IN (SELECT value FROM json_each(?3))
                 ORDER BY document_fts.rank",
            )?;
            let mut rows = statement.query(params![expression, allowed, excluded])?;
            let mut best: HashMap<String, (String, f64)> = HashMap::new();
            let mut cutoff = None;
            while let Some(row) = rows.next()? {
                execution.check()?;
                let score: f64 = row.get(2)?;
                if cutoff.is_some_and(|boundary| score < boundary) {
                    break;
                }
                let document_id: String = row.get(0)?;
                let commit_oid: String = row.get(1)?;
                if best.get(&commit_oid).is_none_or(|current| {
                    score > current.1 || (score == current.1 && document_id < current.0)
                }) {
                    best.insert(commit_oid, (document_id, score));
                }
                // A secondary SQL sort disables FTS5's rank-ordered scan. Finish
                // the boundary's equal-score group so ID tie-breaking stays stable
                // without sorting/materializing every matching source document.
                if best.len() >= limit && cutoff.is_none() {
                    cutoff = Some(score);
                }
            }
            let mut selected: Vec<_> = best.into_values().collect();
            selected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            selected.truncate(limit);
            selected
                .into_iter()
                .map(|(id, score)| {
                    execution.check()?;
                    let document = self.get_document(&id)?.ok_or_else(|| {
                        AppError::new(
                            "evidence_missing",
                            format!(
                                "Selected evidence {id} disappeared; retry within a query snapshot"
                            ),
                        )
                    })?;
                    let byte_end = document.text.len();
                    Ok(Candidate {
                        document,
                        score,
                        byte_start: 0,
                        byte_end,
                    })
                })
                .collect()
        })
    }

    pub fn semantic_candidates(
        &self,
        query: &[f32],
        generation_id: i64,
        limit: usize,
        filter: &CandidateFilter,
        execution: &Execution,
    ) -> Result<Vec<Candidate>> {
        self.with_execution(execution, || {
            self.require_scope()?;
            let generation = self.require_generation(generation_id)?;
            if generation.state == "staging" {
                return Err(AppError::new(
                    "semantic_unavailable",
                    "A staging generation cannot serve search",
                ));
            }
            let query = normalized(query)?;
            if let Some(dimensions) = generation.dimensions {
                if dimensions != query.len() {
                    return Err(AppError::new(
                        "embedding_dimension_mismatch",
                        format!(
                            "Query has {} dimensions; generation {generation_id} requires {dimensions}",
                            query.len()
                        ),
                    ));
                }
            } else if self.generation_vector_count(generation_id)? != 0 {
                return Err(AppError::new(
                    "invalid_vector",
                    "Generation has vectors but no declared dimensions",
                ));
            }
            if limit == 0 {
                return Ok(Vec::new());
            }
            let (allowed, excluded) = filter_json(filter)?;
            let mut statement = self.conn.prepare(
                "SELECT d.id,d.commit_oid,e.byte_start,e.byte_end,e.dimensions,e.vector,e.chunk_index
                 FROM embeddings e JOIN documents d ON d.id=e.document_id
                 JOIN query_scope s ON s.oid=d.commit_oid
                 WHERE e.generation_id=?1
                   AND (?2 IS NULL OR d.id IN (SELECT value FROM json_each(?2)))
                   AND d.id NOT IN (SELECT value FROM json_each(?3))",
            )?;
            let mut rows = statement.query(params![generation_id, allowed, excluded])?;
            let mut best: HashMap<String, (String, f64, usize, usize, usize)> = HashMap::new();
            while let Some(row) = rows.next()? {
                execution.check()?;
                let dimensions = to_usize(row.get(4)?)?;
                if dimensions != query.len() {
                    return Err(AppError::new(
                        "invalid_vector",
                        format!("Stored vector dimensions disagree with generation {generation_id}"),
                    ));
                }
                let vector = row.get_ref(5)?.as_blob().map_err(|error| {
                    AppError::new("invalid_vector", format!("Stored vector is not a BLOB: {error}"))
                })?;
                let score = score_vector(vector, &query)?;
                let document_id: String = row.get(0)?;
                let commit_oid: String = row.get(1)?;
                let byte_start = to_usize(row.get(2)?)?;
                let byte_end = to_usize(row.get(3)?)?;
                let chunk_index = to_usize(row.get(6)?)?;
                // Rank lightweight metadata in Rust: SQL ordering would copy all
                // vector BLOBs into a temporary sort. Explicit ties keep evidence
                // stable regardless of the SQLite scan direction or query plan.
                if best.get(&commit_oid).is_none_or(|current| {
                    score > current.1
                        || (score == current.1
                            && (&document_id, chunk_index) < (&current.0, current.4))
                }) {
                    best.insert(commit_oid, (document_id, score, byte_start, byte_end, chunk_index));
                }
            }
            let mut selected: Vec<_> = best.into_values().collect();
            selected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            selected.truncate(limit);
            selected.into_iter().map(|(id, score, byte_start, byte_end, _)| {
                execution.check()?;
                let document = self.get_document(&id)?.ok_or_else(|| {
                    AppError::new("evidence_missing", format!("Selected evidence {id} disappeared; retry within a query snapshot"))
                })?;
                validate_range(&document.text, byte_start, byte_end)?;
                Ok(Candidate { document, score, byte_start, byte_end })
            }).collect()
        })
    }

    fn generation_vector_count(&self, id: i64) -> Result<usize> {
        to_usize(self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE generation_id=?1",
            [id],
            |r| r.get(0),
        )?)
    }

    pub fn active_generation(&self) -> Result<Option<Generation>> {
        self.generation_where("state='active'")
    }

    pub fn staging_generation(&self) -> Result<Option<Generation>> {
        self.generation_where("state='staging'")
    }

    fn generation_where(&self, condition: &str) -> Result<Option<Generation>> {
        let sql = format!(
            "SELECT id,fingerprint,model,dimensions,watermark,state,created_at FROM generations WHERE {condition}"
        );
        let mut statement = self.conn.prepare(&sql)?;
        let mut rows = statement.query([])?;
        rows.next()?.map(generation_from_row).transpose()
    }

    pub fn get_generation(&self, id: i64) -> Result<Option<Generation>> {
        let mut statement = self.conn.prepare(
            "SELECT id,fingerprint,model,dimensions,watermark,state,created_at FROM generations WHERE id=?1",
        )?;
        let mut rows = statement.query([id])?;
        rows.next()?.map(generation_from_row).transpose()
    }

    fn require_generation(&self, id: i64) -> Result<Generation> {
        self.get_generation(id)?.ok_or_else(|| {
            AppError::new(
                "generation_missing",
                format!("Embedding generation {id} does not exist"),
            )
        })
    }

    /// Repeated setup resumes the same staging generation. A different profile supersedes it.
    pub fn ensure_generation(
        &mut self,
        fingerprint: &str,
        model: &Value,
        force: bool,
    ) -> Result<Generation> {
        self.require_writable()?;
        if fingerprint.is_empty() {
            return Err(AppError::new(
                "invalid_input",
                "Embedding fingerprint cannot be empty",
            ));
        }
        if let Some(staging) = self.staging_generation()?
            && staging.fingerprint == fingerprint
        {
            let allow_reuse: bool = self.conn.query_row(
                "SELECT allow_prior_reuse FROM generations WHERE id=?1",
                [staging.id],
                |r| r.get(0),
            )?;
            if !force || !allow_reuse {
                return Ok(staging);
            }
        }
        if !force
            && let Some(active) = self.active_generation()?
            && active.fingerprint == fingerprint
        {
            self.conn.execute(
                "UPDATE generations SET state='retired' WHERE state='staging' AND fingerprint<>?1",
                [fingerprint],
            )?;
            return Ok(active);
        }
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| {
                AppError::new("clock_error", format!("Read generation creation time: {e}"))
            })?
            .as_secs();
        let created_at = i64::try_from(created_at).map_err(|_| {
            AppError::new("clock_error", "Generation creation time is out of range")
        })?;
        let tx = self.conn.transaction()?;
        tx.execute(
            "UPDATE generations SET state='retired' WHERE state='staging'",
            [],
        )?;
        tx.execute(
            "INSERT INTO generations(fingerprint,model,watermark,state,created_at,allow_prior_reuse)
             VALUES(?1,?2,(SELECT COALESCE(MAX(rowid),0) FROM documents),'staging',?3,?4)",
            params![fingerprint, serde_json::to_string(model)?, created_at, !force],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        self.require_generation(id)
    }

    /// Caller must verify document fingerprint compatibility before updating query/runtime metadata.
    pub fn update_generation_model(&mut self, id: i64, model: &Value) -> Result<()> {
        self.require_writable()?;
        let generation = self.require_generation(id)?;
        if generation.state == "retired" {
            return Err(AppError::new(
                "generation_retired",
                format!("Cannot modify retired generation {id}"),
            ));
        }
        self.conn.execute(
            "UPDATE generations SET model=?2 WHERE id=?1",
            params![id, serde_json::to_string(model)?],
        )?;
        Ok(())
    }

    pub fn pending_documents(
        &self,
        generation_id: i64,
        limit: usize,
        execution: &Execution,
    ) -> Result<Vec<DocumentRecord>> {
        self.pending(generation_id, limit, false, execution)
    }

    pub fn pending_documents_in_scope(
        &self,
        generation_id: i64,
        limit: usize,
        execution: &Execution,
    ) -> Result<Vec<DocumentRecord>> {
        self.pending(generation_id, limit, true, execution)
    }

    fn pending(
        &self,
        generation_id: i64,
        limit: usize,
        scoped: bool,
        execution: &Execution,
    ) -> Result<Vec<DocumentRecord>> {
        self.with_execution(execution, || {
            if scoped {
                self.require_scope()?;
            }
            let generation = self.require_generation(generation_id)?;
            if generation.state == "retired" {
                return Err(AppError::new(
                    "generation_retired",
                    format!("Cannot extend retired generation {generation_id}"),
                ));
            }
            let scope_join = if scoped {
                "JOIN query_scope s ON s.oid=d.commit_oid"
            } else {
                ""
            };
            // ID ordering lets the metadata indexes prove that nothing is pending
            // without scanning source JSON. The corpus boundary is still rowid-based.
            let sql = format!(
                "SELECT d.id FROM documents d {scope_join}
                 WHERE (?2='active' OR d.rowid<=?3) AND NOT EXISTS(
                   SELECT 1 FROM embedded_documents e WHERE e.document_id=d.id AND e.generation_id=?1
                 ) ORDER BY d.id LIMIT ?4",
            );
            let mut statement = self.conn.prepare(&sql)?;
            let ids = statement.query_map(
                params![
                    generation_id,
                    generation.state,
                    generation.watermark,
                    to_i64(limit)?
                ],
                |r| r.get::<_, String>(0),
            )?.collect::<std::result::Result<Vec<_>, _>>()?;
            ids.into_iter().map(|id| {
                execution.check()?;
                self.get_document(&id)?.ok_or_else(|| {
                    AppError::new("evidence_missing", format!("Pending document {id} disappeared while preparing an embedding batch"))
                })
            }).collect()
        })
    }

    /// Copy compatible content before unreachable commit associations are pruned.
    /// Keyset paging keeps memory bounded even when the active corpus is incomplete.
    pub fn reuse_pending_documents(
        &mut self,
        generation_id: i64,
        execution: &Execution,
    ) -> Result<usize> {
        self.require_writable()?;
        execution.check()?;
        if self.require_generation(generation_id)?.state != "active" {
            return Err(AppError::new(
                "generation_not_active",
                "Pending content reuse requires the active generation",
            ));
        }
        let watermark: i64 =
            self.conn
                .query_row("SELECT COALESCE(MAX(rowid),0) FROM documents", [], |r| {
                    r.get(0)
                })?;
        let mut cursor = 0;
        let mut reused = 0;
        loop {
            execution.check()?;
            let pending = self.with_execution(execution, || {
                let mut statement = self.conn.prepare(
                    "SELECT d.rowid,d.id FROM documents d WHERE d.rowid>?2 AND d.rowid<=?3
                     AND NOT EXISTS(SELECT 1 FROM embedded_documents e
                       WHERE e.generation_id=?1 AND e.document_id=d.id)
                     ORDER BY d.rowid LIMIT 128",
                )?;
                Ok(statement
                    .query_map(params![generation_id, cursor, watermark], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?)
            })?;
            if pending.is_empty() {
                break;
            }
            for (rowid, id) in pending {
                execution.check()?;
                if self.reuse_document(generation_id, &id)? {
                    reused += 1;
                }
                cursor = rowid;
            }
        }
        execution.check()?;
        Ok(reused)
    }

    /// Validate and normalize before entering the short write transaction.
    pub fn complete_document(
        &mut self,
        generation_id: i64,
        document_id: &str,
        chunks: &[VectorChunk],
    ) -> Result<()> {
        self.require_writable()?;
        let generation = self.require_generation(generation_id)?;
        if generation.state == "retired" {
            return Err(AppError::new(
                "generation_retired",
                format!("Cannot extend retired generation {generation_id}"),
            ));
        }
        let document = self.get_document(document_id)?.ok_or_else(|| {
            AppError::new(
                "evidence_missing",
                format!("Document {document_id} does not exist"),
            )
        })?;
        let rowid: i64 = self.conn.query_row(
            "SELECT rowid FROM documents WHERE id=?1",
            [document_id],
            |r| r.get(0),
        )?;
        if generation.state == "staging" && rowid > generation.watermark {
            return Err(AppError::new(
                "generation_watermark",
                "Document arrived after this generation's frozen corpus; index it after activation",
            ));
        }
        if chunks.is_empty() && !document.text.is_empty() {
            return Err(AppError::new(
                "invalid_vector",
                format!("Document {document_id} has no embedding chunks"),
            ));
        }
        let dimensions = chunks.first().map(|c| c.vector.len());
        if generation
            .dimensions
            .zip(dimensions)
            .is_some_and(|(old, new)| old != new)
        {
            return Err(AppError::new(
                "embedding_dimension_mismatch",
                format!("Embedding dimensions changed within generation {generation_id}"),
            ));
        }
        let mut covered_end = 0;
        let mut vectors = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            validate_range(&document.text, chunk.byte_start, chunk.byte_end)?;
            if chunk.byte_start > covered_end {
                return Err(AppError::new(
                    "embedding_coverage_gap",
                    format!("Embedding chunks leave source bytes uncovered in {document_id}"),
                ));
            }
            covered_end = covered_end.max(chunk.byte_end);
            if Some(chunk.vector.len()) != dimensions {
                return Err(AppError::new(
                    "embedding_dimension_mismatch",
                    "Chunks in one document have different dimensions",
                ));
            }
            vectors.push(encode_vector(&normalized(&chunk.vector)?));
        }
        if covered_end != document.text.len() {
            return Err(AppError::new(
                "embedding_coverage_gap",
                format!("Embedding chunks do not cover the end of {document_id}"),
            ));
        }
        let tx = self.conn.transaction()?;
        let completed: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM embedded_documents WHERE generation_id=?1 AND document_id=?2)",
            params![generation_id, document_id], |r| r.get(0),
        )?;
        if completed {
            return Ok(());
        }
        tx.execute(
            "UPDATE generations SET dimensions=?2 WHERE id=?1 AND dimensions IS NULL",
            params![generation_id, dimensions.map(to_i64).transpose()?],
        )?;
        tx.execute(
            "INSERT INTO embedded_documents(generation_id,document_id) VALUES(?1,?2)",
            params![generation_id, document_id],
        )?;
        {
            let mut insert = tx.prepare(
                "INSERT INTO embeddings(generation_id,document_id,chunk_index,byte_start,byte_end,dimensions,vector)
                 VALUES(?1,?2,?3,?4,?5,?6,?7)",
            )?;
            for (index, (chunk, vector)) in chunks.iter().zip(vectors).enumerate() {
                insert.execute(params![
                    generation_id,
                    document_id,
                    to_i64(index)?,
                    to_i64(chunk.byte_start)?,
                    to_i64(chunk.byte_end)?,
                    to_i64(chunk.vector.len())?,
                    vector
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Only identical document text under the exact embedding fingerprint is reusable.
    pub fn reuse_document(&mut self, generation_id: i64, document_id: &str) -> Result<bool> {
        self.require_writable()?;
        let generation = self.require_generation(generation_id)?;
        if generation.state == "retired" {
            return Err(AppError::new(
                "generation_retired",
                format!("Cannot extend retired generation {generation_id}"),
            ));
        }
        let document = self.get_document(document_id)?.ok_or_else(|| {
            AppError::new(
                "evidence_missing",
                format!("Document {document_id} does not exist"),
            )
        })?;
        let allow_prior_reuse: bool = self.conn.query_row(
            "SELECT allow_prior_reuse FROM generations WHERE id=?1",
            [generation_id],
            |r| r.get(0),
        )?;
        let source: Option<(i64, String)> = self
            .conn
            .query_row(
                "SELECT e.generation_id,e.document_id FROM embedded_documents e
             JOIN documents d ON d.id=e.document_id JOIN generations g ON g.id=e.generation_id
             WHERE d.content_hash=?1 AND d.text=?2 AND g.fingerprint=?3
               AND (?4 OR e.generation_id=?5)
             ORDER BY (e.generation_id=?5) DESC,e.generation_id DESC LIMIT 1",
                params![
                    document.content_hash,
                    document.text,
                    generation.fingerprint,
                    allow_prior_reuse,
                    generation_id
                ],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((source_generation, source_document)) = source else {
            return Ok(false);
        };
        let chunks = {
            let mut statement = self.conn.prepare(
                "SELECT byte_start,byte_end,dimensions,vector FROM embeddings
                 WHERE generation_id=?1 AND document_id=?2 ORDER BY chunk_index",
            )?;
            let mut rows = statement.query(params![source_generation, source_document])?;
            let mut chunks = Vec::new();
            while let Some(row) = rows.next()? {
                let dimensions = to_usize(row.get(2)?)?;
                chunks.push(VectorChunk {
                    byte_start: to_usize(row.get(0)?)?,
                    byte_end: to_usize(row.get(1)?)?,
                    vector: decode_vector(&row.get::<_, Vec<u8>>(3)?, dimensions)?,
                });
            }
            chunks
        };
        self.complete_document(generation_id, document_id, &chunks)?;
        Ok(true)
    }

    pub fn activate_generation(&mut self, id: i64) -> Result<()> {
        self.require_writable()?;
        let generation = self.require_generation(id)?;
        if generation.state == "active" {
            return Ok(());
        }
        if generation.state != "staging" {
            return Err(AppError::new(
                "generation_retired",
                format!("Cannot activate retired generation {id}"),
            ));
        }
        let tx = self.conn.transaction()?;
        let pending: i64 = tx.query_row(
            "SELECT COUNT(*) FROM documents d WHERE d.rowid<=?2 AND NOT EXISTS(
               SELECT 1 FROM embedded_documents e WHERE e.generation_id=?1 AND e.document_id=d.id
             )",
            params![id, generation.watermark],
            |r| r.get(0),
        )?;
        if pending != 0 {
            return Err(AppError::new(
                "generation_incomplete",
                format!("Generation {id} still has {pending} unembedded documents"),
            ));
        }
        tx.execute(
            "UPDATE generations SET state='retired' WHERE state='active'",
            [],
        )?;
        tx.execute("UPDATE generations SET state='active' WHERE id=?1", [id])?;
        tx.commit()?;
        Ok(())
    }

    /// WAL readers retain their old pages until their snapshots finish.
    pub fn cleanup_retired(&mut self) -> Result<usize> {
        self.require_writable()?;
        Ok(self
            .conn
            .execute("DELETE FROM generations WHERE state='retired'", [])?)
    }
}

fn generation_from_row(row: &rusqlite::Row<'_>) -> Result<Generation> {
    let dimensions: Option<i64> = row.get(3)?;
    Ok(Generation {
        id: row.get(0)?,
        fingerprint: row.get(1)?,
        model: serde_json::from_str(&row.get::<_, String>(2)?)?,
        dimensions: dimensions.map(to_usize).transpose()?,
        watermark: row.get(4)?,
        state: row.get(5)?,
        created_at: row.get(6)?,
    })
}

fn filter_json(filter: &CandidateFilter) -> Result<(Option<String>, String)> {
    Ok((
        filter
            .allowed_documents
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?,
        serde_json::to_string(&filter.excluded_documents)?,
    ))
}

fn lexical_expression(query: &str) -> String {
    query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|word| !word.is_empty())
        .map(|word| format!("\"{word}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn normalized(vector: &[f32]) -> Result<Vec<f32>> {
    if vector.is_empty() || vector.iter().any(|x| !x.is_finite()) {
        return Err(AppError::new(
            "invalid_vector",
            "Embedding must contain finite, nonempty float values",
        ));
    }
    let norm: f64 = vector
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(AppError::new(
            "invalid_vector",
            "Embedding norm must be finite and greater than zero",
        ));
    }
    Ok(vector
        .iter()
        .map(|x| (f64::from(*x) / norm) as f32)
        .collect())
}

fn encode_vector(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn decode_vector(bytes: &[u8], dimensions: usize) -> Result<Vec<f32>> {
    if dimensions == 0 || dimensions.checked_mul(4) != Some(bytes.len()) {
        return Err(AppError::new(
            "invalid_vector",
            "Stored vector byte length does not match its dimensions",
        ));
    }
    let vector: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let norm: f64 = vector.iter().map(|x| f64::from(*x).powi(2)).sum();
    if vector.iter().any(|x| !x.is_finite()) || !norm.is_finite() || (norm - 1.0).abs() > 0.001 {
        return Err(AppError::new(
            "invalid_vector",
            "Stored vector is non-finite or not L2-normalized",
        ));
    }
    Ok(vector)
}

/// Score borrowed SQLite bytes in one pass without allocating a decoded vector.
/// Every stored vector is still validated; a damaged index must fail explicitly.
fn score_vector(bytes: &[u8], query: &[f32]) -> Result<f64> {
    if query.is_empty() || query.len().checked_mul(4) != Some(bytes.len()) {
        return Err(AppError::new(
            "invalid_vector",
            "Stored vector byte length does not match its dimensions",
        ));
    }
    let mut norm = 0.0_f64;
    let mut score = 0.0_f64;
    for (encoded, query_value) in bytes.as_chunks::<4>().0.iter().zip(query) {
        let value = f64::from(f32::from_le_bytes(*encoded));
        norm += value * value;
        score += value * f64::from(*query_value);
    }
    if !norm.is_finite() || (norm - 1.0).abs() > 0.001 {
        return Err(AppError::new(
            "invalid_vector",
            "Stored vector is non-finite or not L2-normalized",
        ));
    }
    Ok(score)
}

fn validate_range(text: &str, start: usize, end: usize) -> Result<()> {
    if start > end
        || end > text.len()
        || !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
    {
        return Err(AppError::new(
            "invalid_vector",
            "Embedding byte range is not a valid UTF-8 slice of its source document",
        ));
    }
    Ok(())
}

fn to_i64(value: usize) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| AppError::new("index_error", "Index count exceeds SQLite integer range"))
}

fn to_usize(value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        AppError::new(
            "invalid_data",
            "Index contains an invalid negative or oversized count",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::DocumentKind;
    use serde_json::json;

    fn source(oid: &str, text: &str) -> (CommitRecord, DocumentRecord) {
        let commit = CommitRecord {
            oid: oid.into(),
            tree_oid: format!("tree-{oid}"),
            parents: vec![],
            author_name: "Author".into(),
            author_email: "author@example.invalid".into(),
            authored_at: 1,
            committed_at: 1,
            message: text.into(),
        };
        let document = DocumentRecord {
            id: format!("doc-{oid}"),
            commit_oid: oid.into(),
            parent_oid: None,
            parent_index: 0,
            kind: DocumentKind::Message,
            old_path: None,
            new_path: None,
            old_blob: None,
            new_blob: None,
            old_start: 0,
            old_lines: 0,
            new_start: 0,
            new_lines: 0,
            text: text.into(),
            content_hash: format!("hash-{text}"),
            truncated: false,
        };
        (commit, document)
    }

    fn ingest(store: &mut Store, oid: &str, text: &str) -> DocumentRecord {
        let (commit, document) = source(oid, text);
        store
            .ingest(&[commit], std::slice::from_ref(&document), &[])
            .unwrap();
        document
    }

    fn embed(store: &mut Store, generation: i64, doc: &DocumentRecord, vector: Vec<f32>) {
        store
            .complete_document(
                generation,
                &doc.id,
                &[VectorChunk {
                    byte_start: 0,
                    byte_end: doc.text.len(),
                    vector,
                }],
            )
            .unwrap();
    }

    #[test]
    fn explicit_maintenance_adds_lookup_index_without_changing_sources_or_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        let mut writer = Store::open(&path).unwrap();
        let document = ingest(&mut writer, "commit", "cache invalidation");
        let generation = writer
            .ensure_generation("model-a", &json!({}), false)
            .unwrap();
        embed(&mut writer, generation.id, &document, vec![1.0, 0.0]);
        writer.activate_generation(generation.id).unwrap();
        writer
            .conn
            .execute_batch("DROP INDEX documents_identity_commit")
            .unwrap();
        drop(writer);

        let mut reader = Store::read_only(&path).unwrap();
        reader.set_scope(&["commit".into()]).unwrap();
        let previous = reader
            .semantic_candidates(
                &[1.0, 0.0],
                generation.id,
                1,
                &CandidateFilter::default(),
                &Execution::unlimited(),
            )
            .unwrap();
        assert_eq!(
            reader.optimize(&Execution::unlimited()).unwrap_err().code,
            "index_read_only"
        );
        assert_eq!(
            reader
                .count("SELECT COUNT(*) FROM sqlite_schema WHERE name='documents_identity_commit'")
                .unwrap(),
            0
        );
        drop(reader);

        let mut writer = Store::open(&path).unwrap();
        assert_eq!(
            writer
                .optimize(&Execution::unlimited().bounded(Duration::ZERO))
                .unwrap_err()
                .code,
            "timeout"
        );
        assert_eq!(
            writer
                .count("SELECT COUNT(*) FROM sqlite_schema WHERE name='documents_identity_commit'")
                .unwrap(),
            0
        );
        writer.optimize(&Execution::unlimited()).unwrap();
        assert_eq!(
            writer
                .count("SELECT COUNT(*) FROM sqlite_stat1 WHERE idx='documents_identity_commit'")
                .unwrap(),
            1
        );
        writer.set_scope(&["commit".into()]).unwrap();
        let current = writer
            .semantic_candidates(
                &[1.0, 0.0],
                generation.id,
                1,
                &CandidateFilter::default(),
                &Execution::unlimited(),
            )
            .unwrap();
        assert_eq!(
            serde_json::to_value(previous).unwrap(),
            serde_json::to_value(current).unwrap()
        );
        let counts = writer.counts().unwrap();
        assert_eq!(
            (counts.commits, counts.documents, counts.vectors),
            (1, 1, 1)
        );
        assert_eq!(
            writer.active_generation().unwrap().unwrap().id,
            generation.id
        );
    }

    #[test]
    fn live_scope_filters_before_commit_limit_and_fts_input_is_literal() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        ingest(&mut store, "old", "cache cache cache cache");
        ingest(&mut store, "live", "cache invalidation");
        assert_eq!(
            store
                .lexical_candidates(
                    "cache",
                    1,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap_err()
                .code,
            "scope_required"
        );
        store
            .set_scope(&["live".into(), "not-indexed".into()])
            .unwrap();
        let candidates = store
            .lexical_candidates(
                "cache \" OR (NEAR(*))",
                1,
                &CandidateFilter::default(),
                &Execution::unlimited(),
            )
            .unwrap();
        assert_eq!(candidates[0].document.commit_oid, "live");
        let coverage = store.scope_coverage().unwrap();
        assert_eq!((coverage.total_commits, coverage.indexed_commits), (2, 1));
        store.set_scope(&[]).unwrap();
        assert!(
            store
                .lexical_candidates(
                    "cache",
                    1,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn staging_is_resumable_and_frozen_but_active_catches_up() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        let first = ingest(&mut store, "first", "first text");
        let generation = store
            .ensure_generation("model-a", &json!({"name":"a"}), false)
            .unwrap();
        embed(&mut store, generation.id, &first, vec![3.0, 4.0]);
        let second = ingest(&mut store, "second", "second text");
        assert_eq!(
            store
                .ensure_generation("model-a", &json!({"name":"a"}), false)
                .unwrap()
                .id,
            generation.id
        );
        assert!(
            store
                .pending_documents(generation.id, 20, &Execution::unlimited())
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            store
                .complete_document(
                    generation.id,
                    &second.id,
                    &[VectorChunk {
                        byte_start: 0,
                        byte_end: second.text.len(),
                        vector: vec![1.0, 0.0]
                    }]
                )
                .unwrap_err()
                .code,
            "generation_watermark"
        );
        store.activate_generation(generation.id).unwrap();
        assert_eq!(
            store
                .pending_documents(generation.id, 20, &Execution::unlimited())
                .unwrap()[0]
                .id,
            second.id
        );
        embed(&mut store, generation.id, &second, vec![1.0, 0.0]);
        let replacement = store
            .ensure_generation("model-b", &json!({"name":"b"}), false)
            .unwrap();
        assert_eq!(
            store.activate_generation(replacement.id).unwrap_err().code,
            "generation_incomplete"
        );
        assert_eq!(
            store.active_generation().unwrap().unwrap().id,
            generation.id
        );
        assert!(!store.reuse_document(replacement.id, &first.id).unwrap());
        assert_eq!(
            store
                .ensure_generation("model-a", &json!({"name":"a"}), false)
                .unwrap()
                .id,
            generation.id
        );
        assert!(store.staging_generation().unwrap().is_none());
        assert_eq!(
            store.get_generation(replacement.id).unwrap().unwrap().state,
            "retired"
        );
    }

    #[test]
    fn rebased_text_reuses_vectors_only_under_the_same_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        let original = ingest(&mut store, "original", "identical source text");
        let generation = store
            .ensure_generation("model-a", &json!({}), false)
            .unwrap();
        embed(&mut store, generation.id, &original, vec![2.0, 0.0]);
        store.activate_generation(generation.id).unwrap();
        let unmatched = ingest(&mut store, "unmatched", "different source text");
        let rewritten = ingest(&mut store, "rewritten", "identical source text");
        for index in 0..128 {
            ingest(
                &mut store,
                &format!("rewrite-{index}"),
                "identical source text",
            );
        }
        assert_eq!(
            store
                .reuse_pending_documents(generation.id, &Execution::unlimited())
                .unwrap(),
            129
        );
        assert_eq!(
            store
                .pending_documents(generation.id, 10, &Execution::unlimited())
                .unwrap()[0]
                .id,
            unmatched.id
        );
        store.set_scope(&["rewritten".into()]).unwrap();
        let hits = store
            .semantic_candidates(
                &[1.0, 0.0],
                generation.id,
                1,
                &CandidateFilter::default(),
                &Execution::unlimited(),
            )
            .unwrap();
        assert_eq!(hits[0].document.commit_oid, "rewritten");
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        let forced = store
            .ensure_generation("model-a", &json!({}), true)
            .unwrap();
        assert!(!store.reuse_document(forced.id, &original.id).unwrap());
        embed(&mut store, forced.id, &original, vec![0.0, 1.0]);
        assert!(store.reuse_document(forced.id, &rewritten.id).unwrap());
    }

    #[test]
    fn snapshot_keeps_model_and_vectors_after_cutover_and_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        let mut writer = Store::open(&path).unwrap();
        let document = ingest(&mut writer, "commit", "cache");
        let first = writer
            .ensure_generation("model-a", &json!({"model":"a"}), false)
            .unwrap();
        embed(&mut writer, first.id, &document, vec![1.0, 0.0]);
        writer.activate_generation(first.id).unwrap();
        let mut reader = Store::read_only(&path).unwrap();
        reader.set_scope(&["commit".into()]).unwrap();
        reader.begin_snapshot().unwrap();
        let next = writer
            .ensure_generation("model-b", &json!({"model":"b"}), false)
            .unwrap();
        embed(&mut writer, next.id, &document, vec![0.0, 1.0]);
        writer.activate_generation(next.id).unwrap();
        writer.cleanup_retired().unwrap();
        assert_eq!(reader.active_generation().unwrap().unwrap().id, first.id);
        assert_eq!(
            reader
                .semantic_candidates(
                    &[1.0, 0.0],
                    first.id,
                    1,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap()
                .len(),
            1
        );
        reader.end_snapshot().unwrap();
        assert_eq!(reader.active_generation().unwrap().unwrap().id, next.id);
        assert_eq!(
            reader
                .semantic_candidates(
                    &[1.0, 0.0],
                    first.id,
                    1,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap_err()
                .code,
            "generation_missing"
        );
    }

    #[test]
    fn invalid_vectors_and_partial_source_coverage_never_mark_a_document_complete() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        let document = ingest(&mut store, "commit", "café text");
        let generation = store
            .ensure_generation("model-a", &json!({}), false)
            .unwrap();
        let mut chunk = VectorChunk {
            byte_start: 0,
            byte_end: document.text.len(),
            vector: vec![f32::NAN, 1.0],
        };
        assert_eq!(
            store
                .complete_document(generation.id, &document.id, &[chunk.clone()])
                .unwrap_err()
                .code,
            "invalid_vector"
        );
        chunk.vector = vec![1.0, 0.0];
        chunk.byte_end -= 1;
        assert_eq!(
            store
                .complete_document(generation.id, &document.id, &[chunk.clone()])
                .unwrap_err()
                .code,
            "embedding_coverage_gap"
        );
        assert_eq!(
            store
                .pending_documents(generation.id, 10, &Execution::unlimited())
                .unwrap()
                .len(),
            1
        );
        chunk.byte_end = 4; // Ends inside é.
        assert_eq!(
            store
                .complete_document(generation.id, &document.id, &[chunk])
                .unwrap_err()
                .code,
            "invalid_vector"
        );
        embed(&mut store, generation.id, &document, vec![1.0, 0.0]);
        store.activate_generation(generation.id).unwrap();
        store.set_scope(&["commit".into()]).unwrap();
        assert_eq!(
            store
                .semantic_candidates(
                    &[1.0],
                    generation.id,
                    1,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap_err()
                .code,
            "embedding_dimension_mismatch"
        );
        for corrupt in [
            encode_vector(&[0.0, 0.0]),
            encode_vector(&[f32::NAN, 1.0]),
            vec![0; 3],
        ] {
            store
                .conn
                .execute("UPDATE embeddings SET vector=?1", [corrupt])
                .unwrap();
            assert_eq!(
                store
                    .semantic_candidates(
                        &[1.0, 0.0],
                        generation.id,
                        1,
                        &CandidateFilter::default(),
                        &Execution::unlimited()
                    )
                    .unwrap_err()
                    .code,
                "invalid_vector"
            );
        }
    }

    #[test]
    fn canonical_reextraction_preserves_unchanged_vectors_and_removes_excluded_sources() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        let (commit, message) = source("commit", "commit message");
        let mut patch = message.clone();
        patch.id = "patch-document".into();
        patch.kind = DocumentKind::Diff;
        patch.text = "excluded patch content".into();
        patch.content_hash = "hash-patch".into();
        store
            .ingest(
                std::slice::from_ref(&commit),
                &[message.clone(), patch.clone()],
                &[],
            )
            .unwrap();
        let generation = store
            .ensure_generation("model-a", &json!({}), false)
            .unwrap();
        embed(&mut store, generation.id, &message, vec![1.0, 0.0]);
        embed(&mut store, generation.id, &patch, vec![0.0, 1.0]);
        store.activate_generation(generation.id).unwrap();
        let omission = ExtractionOmission {
            commit_oid: commit.oid.clone(),
            path: None,
            reason: "excluded".into(),
            detail: "Explicit indexing policy".into(),
        };
        store
            .ingest(&[commit], std::slice::from_ref(&message), &[omission])
            .unwrap();
        assert!(store.get_document(&patch.id).unwrap().is_none());
        assert_eq!(store.counts().unwrap().vectors, 1);
        assert!(
            store
                .pending_documents(generation.id, 10, &Execution::unlimited())
                .unwrap()
                .is_empty()
        );
        store.set_scope(&["commit".into()]).unwrap();
        assert!(
            store
                .lexical_candidates(
                    "excluded",
                    10,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap()
                .is_empty()
        );
        assert_eq!(store.scope_coverage().unwrap().omissions, 1);
        store.set_metadata("index_policy", "policy-2").unwrap();
        assert_eq!(
            store.metadata("index_policy").unwrap().as_deref(),
            Some("policy-2")
        );

        let (old_commit, old_document) = source("unreachable", "old patch");
        let old_omission = ExtractionOmission {
            commit_oid: old_commit.oid.clone(),
            path: None,
            reason: "missing_blob".into(),
            detail: "Pruned source".into(),
        };
        store
            .ingest(
                &[old_commit],
                std::slice::from_ref(&old_document),
                &[old_omission],
            )
            .unwrap();
        embed(&mut store, generation.id, &old_document, vec![0.0, 1.0]);
        store
            .set_scope(&["commit".into(), "unreachable".into()])
            .unwrap();
        let mut reader = Store::read_only(dir.path().join("index.db")).unwrap();
        reader.set_scope(&["unreachable".into()]).unwrap();
        reader.begin_snapshot().unwrap();
        let replacement = store
            .ensure_generation("model-b", &json!({}), false)
            .unwrap();
        assert_eq!(
            store
                .prune_commits_except(&HashSet::from(["commit".into()]))
                .unwrap(),
            1
        );
        assert!(store.get_document(&old_document.id).unwrap().is_none());
        assert_eq!(store.counts().unwrap().vectors, 1);
        assert_eq!(store.counts().unwrap().omissions, 1);
        let coverage = store.scope_coverage().unwrap();
        assert_eq!((coverage.total_commits, coverage.indexed_commits), (2, 1));
        assert_eq!(
            store
                .pending_documents(replacement.id, 10, &Execution::unlimited())
                .unwrap()
                .len(),
            1
        );
        embed(&mut store, replacement.id, &message, vec![1.0, 0.0]);
        store.activate_generation(replacement.id).unwrap();
        assert_eq!(
            reader
                .semantic_candidates(
                    &[0.0, 1.0],
                    generation.id,
                    1,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap()[0]
                .document
                .commit_oid,
            "unreachable"
        );
        reader.end_snapshot().unwrap();
        assert!(reader.get_document(&old_document.id).unwrap().is_none());
    }

    #[test]
    fn empty_sources_are_complete_without_vectors_and_scoped_pending_excludes_other_history() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        let empty = ingest(&mut store, "empty", "");
        let generation = store
            .ensure_generation("model-a", &json!({}), false)
            .unwrap();
        store
            .complete_document(generation.id, &empty.id, &[])
            .unwrap();
        store.activate_generation(generation.id).unwrap();
        store.set_scope(&["empty".into()]).unwrap();
        assert_eq!(store.scope_coverage().unwrap().embedded_documents, 1);
        assert!(
            store
                .semantic_candidates(
                    &[1.0],
                    generation.id,
                    10,
                    &CandidateFilter::default(),
                    &Execution::unlimited()
                )
                .unwrap()
                .is_empty()
        );
        ingest(&mut store, "outside", "old branch");
        let selected = ingest(&mut store, "inside", "current branch");
        store.set_scope(&["inside".into()]).unwrap();
        assert_eq!(
            store
                .pending_documents_in_scope(generation.id, 1, &Execution::unlimited())
                .unwrap()[0]
                .id,
            selected.id
        );
        assert_eq!(
            store
                .pending_documents(generation.id, 10, &Execution::unlimited())
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn document_filters_refill_with_other_evidence_before_collapsing_commits() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        let (commit, message) = source("a", "needle");
        let mut bad = message.clone();
        bad.id = "a-bad".into();
        bad.kind = DocumentKind::Diff;
        let mut good = bad.clone();
        good.id = "a-good".into();
        good.old_path = Some(crate::git::GitPath {
            display: "old.rs".into(),
            bytes_hex: "6f6c642e7273".into(),
        });
        good.new_path = Some(crate::git::GitPath {
            display: "new.rs".into(),
            bytes_hex: "6e65772e7273".into(),
        });
        store
            .ingest(
                &[commit],
                &[message.clone(), bad.clone(), good.clone()],
                &[],
            )
            .unwrap();
        let other = ingest(&mut store, "b", "needle");
        let generation = store
            .ensure_generation("model-a", &json!({}), false)
            .unwrap();
        store
            .complete_document(
                generation.id,
                &bad.id,
                &[
                    VectorChunk {
                        byte_start: 0,
                        byte_end: 3,
                        vector: vec![1.0, 0.0],
                    },
                    VectorChunk {
                        byte_start: 3,
                        byte_end: 6,
                        vector: vec![1.0, 0.0],
                    },
                ],
            )
            .unwrap();
        embed(&mut store, generation.id, &good, vec![1.0, 0.0]);
        embed(&mut store, generation.id, &message, vec![0.0, 1.0]);
        embed(&mut store, generation.id, &other, vec![0.0, 1.0]);
        store.activate_generation(generation.id).unwrap();
        store.set_scope(&["a".into(), "b".into()]).unwrap();
        store
            .conn
            .execute_batch("PRAGMA reverse_unordered_selects=ON")
            .unwrap();
        let execution = Execution::unlimited();
        let metadata = store.document_paths_in_scope(&execution).unwrap();
        let good_metadata = metadata.iter().find(|d| d.id == good.id).unwrap();
        assert_eq!(
            good_metadata.old_path.as_ref().unwrap().bytes_hex,
            "6f6c642e7273"
        );
        assert_eq!(
            good_metadata.new_path.as_ref().unwrap().bytes_hex,
            "6e65772e7273"
        );
        let mut filter = CandidateFilter::default();
        for hits in [
            store
                .lexical_candidates("needle", 1, &filter, &execution)
                .unwrap(),
            store
                .semantic_candidates(&[1.0, 0.0], generation.id, 1, &filter, &execution)
                .unwrap(),
        ] {
            assert_eq!(hits[0].document.id, bad.id);
            assert_eq!(hits[0].byte_start, 0);
        }
        filter.excluded_documents.insert(bad.id);
        for hits in [
            store
                .lexical_candidates("needle", 1, &filter, &execution)
                .unwrap(),
            store
                .semantic_candidates(&[1.0, 0.0], generation.id, 1, &filter, &execution)
                .unwrap(),
        ] {
            assert_eq!(hits[0].document.id, good.id);
        }
        filter.allowed_documents = Some(HashSet::from([other.id.clone()]));
        for hits in [
            store
                .lexical_candidates("needle", 1, &filter, &execution)
                .unwrap(),
            store
                .semantic_candidates(&[1.0, 0.0], generation.id, 1, &filter, &execution)
                .unwrap(),
        ] {
            assert_eq!(hits[0].document.id, other.id);
        }
        filter.allowed_documents = Some(HashSet::new());
        assert!(
            store
                .lexical_candidates("needle", 1, &filter, &execution)
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .semantic_candidates(&[1.0, 0.0], generation.id, 1, &filter, &execution)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn cancellation_interrupts_sql_work_and_does_not_poison_the_next_query() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        use std::time::Instant;
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path().join("index.db")).unwrap();
        ingest(&mut store, "commit", "needle");
        store.set_scope(&["commit".into()]).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let execution = Execution::new(None, cancelled.clone());
        let started = Instant::now();
        let error = store.with_execution(&execution, || {
            cancelled.store(true, Ordering::Relaxed);
            let _: i64 = store.conn.query_row(
                "WITH RECURSIVE n(v) AS (VALUES(0) UNION ALL SELECT v+1 FROM n WHERE v<100000000) SELECT SUM(v) FROM n",
                [], |r| r.get(0),
            )?;
            Ok(())
        }).unwrap_err();
        assert_eq!(error.code, "interrupted");
        assert!(started.elapsed() < Duration::from_secs(2));
        let filter = CandidateFilter::default();
        assert_eq!(
            store
                .lexical_candidates("needle", 1, &filter, &Execution::unlimited())
                .unwrap()
                .len(),
            1
        );
        let expired = Execution::unlimited().bounded(Duration::ZERO);
        assert_eq!(
            store
                .lexical_candidates("needle", 1, &filter, &expired)
                .unwrap_err()
                .code,
            "timeout"
        );
        assert_eq!(
            store
                .semantic_candidates(&[1.0], 999, 1, &filter, &expired)
                .unwrap_err()
                .code,
            "timeout"
        );
        assert_eq!(
            store
                .pending_documents_in_scope(999, 1, &expired)
                .unwrap_err()
                .code,
            "timeout"
        );
    }
}
