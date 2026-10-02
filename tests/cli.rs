use serde_json::Value;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, Output, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, Sender},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tempfile::TempDir;

struct Repository {
    _directory: TempDir,
    path: PathBuf,
    data: PathBuf,
    config: PathBuf,
    hf_home: PathBuf,
}

impl Repository {
    fn new() -> Self {
        let directory = tempfile::tempdir().expect("create test directory");
        let path = directory.path().join("repository");
        let data = directory.path().join("nalcos-data");
        let config = directory.path().join("config.toml");
        let hf_home = directory.path().join("hugging-face");
        fs::create_dir(&path).expect("create repository directory");
        fs::write(&config, "version = 1\n").expect("write isolated configuration");
        let repository = Self {
            _directory: directory,
            path,
            data,
            config,
            hf_home,
        };
        repository.git(&["init", "--quiet", "--initial-branch=main"]);
        repository
    }

    fn git_command(&self) -> Command {
        let mut command = Command::new("git");
        command
            .current_dir(&self.path)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_AUTHOR_NAME", "NaLCoS Test")
            .env("GIT_AUTHOR_EMAIL", "nalcos-test@example.invalid")
            .env("GIT_COMMITTER_NAME", "NaLCoS Test")
            .env("GIT_COMMITTER_EMAIL", "nalcos-test@example.invalid");
        for variable in [
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        ] {
            command.env_remove(variable);
        }
        command
    }

    fn git(&self, args: &[&str]) -> String {
        let output = self.git_command().args(args).output().expect("execute Git");
        assert_success(&output);
        String::from_utf8(output.stdout)
            .expect("Git output is UTF-8")
            .trim()
            .to_owned()
    }

    fn write(&self, path: &str, text: &str) {
        let path = self.path.join(path);
        fs::create_dir_all(path.parent().expect("file parent")).expect("create fixture parents");
        fs::write(path, text).expect("write fixture file");
    }

    fn commit(&self, message: &str, date: &str) -> String {
        self.git(&["add", "--all"]);
        let output = self
            .git_command()
            .env("GIT_AUTHOR_DATE", date)
            .env("GIT_COMMITTER_DATE", date)
            .args(["commit", "--quiet", "-m", message])
            .output()
            .expect("create fixture commit");
        assert_success(&output);
        self.git(&["rev-parse", "HEAD"])
    }

    fn command_at(&self, path: &Path) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_nalcos"));
        command
            .env("NALCOS_DATA_DIR", &self.data)
            .env("NALCOS_CONFIG", &self.config)
            .env("HF_HOME", &self.hf_home)
            .env("HF_HUB_CACHE", self.hf_home.join("hub"))
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("no_proxy", "127.0.0.1,localhost")
            .arg("--repo")
            .arg(path)
            .args(["--json", "--timeout", "15s"]);
        command
    }

    fn run_at(&self, path: &Path, args: &[&str]) -> Output {
        self.command_at(path)
            .arg("--offline")
            .args(args)
            .output()
            .expect("execute NaLCoS")
    }

    fn run(&self, args: &[&str]) -> Output {
        self.run_at(&self.path, args)
    }

    fn json(&self, args: &[&str]) -> Value {
        let output = self.run(args);
        assert_success(&output);
        parse_json(&output)
    }

    fn run_provider(&self, args: &[&str]) -> Output {
        self.command_at(&self.path)
            .args(args)
            .output()
            .expect("execute NaLCoS with the loopback provider")
    }

    fn provider_json(&self, args: &[&str]) -> Value {
        let output = self.run_provider(args);
        assert_success(&output);
        parse_json(&output)
    }

    fn configure_provider(&self, server: &MockProvider, model: &str, query_prefix: &str) {
        self.configure_provider_profiles(server, &[("mock", model, query_prefix)], None);
    }

    fn configure_provider_profiles(
        &self,
        server: &MockProvider,
        profiles: &[(&str, &str, &str)],
        default_model: Option<&str>,
    ) {
        let tokenizer = self._directory.path().join("tokenizer.json");
        if !tokenizer.exists() {
            let definition = serde_json::json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {"type": "WhitespaceSplit"},
                "post_processor": null,
                "decoder": null,
                "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
            });
            fs::write(&tokenizer, serde_json::to_vec(&definition).unwrap())
                .expect("write local provider tokenizer");
        }
        let quote = |text: &str| serde_json::to_string(text).unwrap();
        let mut configuration = "version = 1\n".to_owned();
        if let Some(model) = default_model {
            configuration.push_str(&format!("default_model = {}\n", quote(model)));
        }
        configuration.push_str("[runtime]\nbatch_size = 8\n");
        for (name, model, query_prefix) in profiles {
            configuration.push_str(&format!(
                "[profiles.{name}]\n\
                 id = {}\n\
                 revision = {}\n\
                 backend = 'open_ai'\n\
                 dimensions = 4\n\
                 max_tokens = 256\n\
                 tokenizer = {}\n\
                 endpoint = {}\n\
                 document_prefix = 'passage: '\n\
                 query_prefix = {}\n",
                quote(model),
                quote(&format!("{model}-immutable-release")),
                quote(tokenizer.to_str().expect("fixture path is UTF-8")),
                quote(&server.endpoint()),
                quote(query_prefix),
            ));
        }
        fs::write(&self.config, configuration).expect("configure loopback embedding provider");
    }

    fn populate_provider_history(&self) -> String {
        self.write("cobalt.txt", "cobalt recovery logic\n");
        let cobalt = self.commit("fixture-cobalt", "2024-01-01T12:00:00Z");
        for index in 2..=12 {
            self.write(
                &format!("neutral-{index}.txt"),
                &format!("neutral setting {index}\n"),
            );
            self.commit(
                &format!("fixture-neutral-{index}"),
                &format!("2024-01-{index:02}T12:00:00Z"),
            );
        }
        cobalt
    }
}

#[derive(Clone)]
struct ProviderRequest {
    model: String,
    inputs: Vec<String>,
    succeeded: bool,
}

#[derive(Default)]
struct ProviderState {
    requests: Vec<ProviderRequest>,
    fail_after_documents: Option<(String, usize)>,
    successful_documents: usize,
    drift: bool,
    gate: Option<ProviderGate>,
}

struct ProviderGate {
    model: String,
    entered: Sender<()>,
    release: Receiver<()>,
}

struct HeldRequest {
    entered: Receiver<()>,
    release: Option<Sender<()>>,
}

impl HeldRequest {
    fn wait_until_called(&self) {
        self.entered
            .recv_timeout(Duration::from_secs(10))
            .expect("writer must reach the gated provider request");
    }

    fn release(mut self) {
        self.release
            .take()
            .expect("provider gate release")
            .send(())
            .expect("release provider response");
    }
}

impl Drop for HeldRequest {
    fn drop(&mut self) {
        if let Some(release) = self.release.take()
            && release.send(()).is_err()
            && !thread::panicking()
        {
            panic!("provider exited before its held request was released");
        }
    }
}

struct RunningCli(Option<Child>);

impl RunningCli {
    fn spawn(mut command: Command) -> Self {
        Self(Some(
            command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("start concurrent CLI process"),
        ))
    }

    fn take_stderr(&mut self) -> ChildStderr {
        self.0
            .as_mut()
            .unwrap()
            .stderr
            .take()
            .expect("piped CLI stderr")
    }

    fn finish(mut self) -> Output {
        self.0
            .take()
            .unwrap()
            .wait_with_output()
            .expect("wait for concurrent CLI process")
    }
}

impl Drop for RunningCli {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            if !matches!(child.try_wait(), Ok(Some(_)))
                && let Err(error) = child.kill()
            {
                eprintln!("could not stop fixture CLI process: {error}");
            }
            if let Err(error) = child.wait() {
                eprintln!("could not reap fixture CLI process: {error}");
            }
        }
    }
}

struct MockProvider {
    address: SocketAddr,
    state: Arc<Mutex<ProviderState>>,
    stopping: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl MockProvider {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback provider");
        let address = listener.local_addr().unwrap();
        let state = Arc::new(Mutex::new(ProviderState::default()));
        let stopping = Arc::new(AtomicBool::new(false));
        let shared_state = Arc::clone(&state);
        let shared_stopping = Arc::clone(&stopping);
        let worker = thread::spawn(move || {
            for connection in listener.incoming() {
                let mut stream = connection.expect("accept provider request");
                if shared_stopping.load(Ordering::Acquire) {
                    break;
                }
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let payload = read_provider_request(&mut stream);
                let model = payload["model"].as_str().expect("request model").to_owned();
                let inputs: Vec<String> = payload["input"]
                    .as_array()
                    .expect("request input array")
                    .iter()
                    .map(|input| input.as_str().expect("text input").to_owned())
                    .collect();
                let mut state = shared_state.lock().unwrap();
                if state.gate.as_ref().is_some_and(|gate| gate.model == model) {
                    let gate = state.gate.take().unwrap();
                    drop(state);
                    gate.entered
                        .send(())
                        .expect("signal gated provider request");
                    gate.release
                        .recv_timeout(Duration::from_secs(10))
                        .expect("wait for provider gate release");
                    state = shared_state.lock().unwrap();
                }
                let documents = inputs
                    .iter()
                    .filter(|text| text.contains("fixture-"))
                    .count();
                let fail = documents > 0
                    && state.fail_after_documents.as_ref().is_some_and(
                        |(selected_model, after)| {
                            model == *selected_model && state.successful_documents >= *after
                        },
                    );
                state.requests.push(ProviderRequest {
                    model: model.clone(),
                    inputs: inputs.clone(),
                    succeeded: !fail,
                });
                let (status, body) = if fail {
                    (
                        "503 Service Unavailable",
                        serde_json::json!({"error": {"message": "injected provider outage"}}),
                    )
                } else {
                    state.successful_documents += documents;
                    // Return reversed indices to exercise the real adapter's correspondence checks.
                    let data: Vec<_> = inputs
                        .iter()
                        .enumerate()
                        .rev()
                        .map(|(index, text)| {
                            serde_json::json!({"index": index, "embedding": provider_vector(text, &model, state.drift)})
                        })
                        .collect();
                    ("200 OK", serde_json::json!({"data": data}))
                };
                drop(state);
                let body = serde_json::to_vec(&body).unwrap();
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .expect("write provider headers");
                stream.write_all(&body).expect("write provider response");
            }
        });
        Self {
            address,
            state,
            stopping,
            worker: Some(worker),
        }
    }

    fn endpoint(&self) -> String {
        format!("http://{}/v1", self.address)
    }

    fn requests(&self) -> Vec<ProviderRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    fn clear_requests(&self) {
        let mut state = self.state.lock().unwrap();
        state.requests.clear();
        state.successful_documents = 0;
    }

    fn fail_after_documents(&self, model: &str, count: usize) {
        self.state.lock().unwrap().fail_after_documents = Some((model.into(), count));
    }

    fn recover(&self) {
        self.state.lock().unwrap().fail_after_documents = None;
    }

    fn change_vectors_under_alias(&self) {
        self.state.lock().unwrap().drift = true;
    }

    fn hold_next_request(&self, model: &str) -> HeldRequest {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        self.state.lock().unwrap().gate = Some(ProviderGate {
            model: model.into(),
            entered: entered_tx,
            release: release_rx,
        });
        HeldRequest {
            entered: entered_rx,
            release: Some(release_tx),
        }
    }
}

impl Drop for MockProvider {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        if let Ok(connection) = TcpStream::connect_timeout(&self.address, Duration::from_secs(1)) {
            drop(connection);
        }
        if let Some(worker) = self.worker.take()
            && let Err(error) = worker.join()
            && !thread::panicking()
        {
            std::panic::resume_unwind(error);
        }
    }
}

fn read_provider_request(stream: &mut TcpStream) -> Value {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).expect("read request line");
    assert_eq!(line, "POST /v1/embeddings HTTP/1.1\r\n");
    let mut length = None;
    loop {
        line.clear();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        if line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = Some(value.trim().parse::<usize>().unwrap());
        }
    }
    let length = length.expect("JSON request content length");
    assert!(length < 1_048_576, "unexpectedly large fixture request");
    let mut body = vec![0; length];
    reader.read_exact(&mut body).expect("read request body");
    serde_json::from_slice(&body).expect("JSON provider request")
}

fn provider_vector(text: &str, model: &str, drift: bool) -> [f32; 4] {
    let axis = if text.contains("cobalt") {
        0
    } else if text.contains("amber") {
        1
    } else if text.contains("fixture-") {
        2
    } else {
        3
    };
    let model_rotation = usize::from(model == "mock-v2");
    let drift_rotation = usize::from(drift);
    let mut vector = [0.0; 4];
    vector[(axis + model_rotation + drift_rotation) % 4] = 1.0;
    vector
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "command exited with {}\nstdout: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn parse_json(output: &Output) -> Value {
    let json: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "stdout must contain one JSON object: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(json["schema_version"], 1);
    json
}

fn result_ids(json: &Value) -> Vec<&str> {
    json["results"]
        .as_array()
        .expect("search results array")
        .iter()
        .map(|result| result["commit"]["oid"].as_str().expect("full commit OID"))
        .collect()
}

#[test]
fn cached_search_rechecks_reset_and_deleted_branch_reachability() {
    let repository = Repository::new();
    repository.write("settings.txt", "retry_timeout = 30\n");
    let base = repository.commit("Initial settings", "2024-01-01T12:00:00Z");
    repository.write("settings.txt", "retry_timeout = 30\nsupersededquartz\n");
    let changed = repository.commit("Adjust settings", "2024-01-02T12:00:00Z");
    repository.git(&["branch", "topic"]);

    let indexed = repository.json(&[
        "search",
        "supersededquartz",
        "--mode",
        "lexical",
        "--freshness",
        "wait",
    ]);
    assert_eq!(result_ids(&indexed), [changed.as_str()]);
    assert_eq!(indexed["history_coverage"]["complete"], true);
    assert!(repository.data.exists(), "search must persist its index");

    repository.git(&["reset", "--hard", &base]);
    let reset = repository.json(&[
        "search",
        "supersededquartz",
        "--mode",
        "lexical",
        "--freshness",
        "cached",
    ]);
    assert!(result_ids(&reset).is_empty());
    assert_eq!(reset["history_coverage"]["total"], 1);

    let all_refs = repository.json(&[
        "search",
        "supersededquartz",
        "--mode",
        "lexical",
        "--freshness",
        "cached",
        "--all-refs",
    ]);
    assert_eq!(result_ids(&all_refs), [changed.as_str()]);

    repository.git(&["branch", "-D", "topic"]);
    let deleted = repository.json(&[
        "search",
        "supersededquartz",
        "--mode",
        "lexical",
        "--freshness",
        "cached",
        "--all-refs",
    ]);
    assert!(result_ids(&deleted).is_empty());
}

#[test]
fn worktrees_share_index_but_keep_their_own_head_scope() {
    let repository = Repository::new();
    repository.write("shared.txt", "commoncitrine\n");
    let base = repository.commit("Shared starting point", "2024-01-01T12:00:00Z");
    repository.write("later.txt", "lateramethyst\n");
    repository.commit("A later change", "2024-01-02T12:00:00Z");
    repository.json(&[
        "search",
        "commoncitrine",
        "--mode",
        "lexical",
        "--freshness",
        "wait",
    ]);

    let worktree = repository._directory.path().join("worktree");
    repository.git(&[
        "worktree",
        "add",
        "--detach",
        worktree.to_str().expect("fixture path is UTF-8"),
        &base,
    ]);
    let output = repository.run_at(
        &worktree,
        &[
            "search",
            "commoncitrine",
            "--mode",
            "lexical",
            "--freshness",
            "cached",
        ],
    );
    assert_success(&output);
    let shared = parse_json(&output);
    assert_eq!(result_ids(&shared), [base.as_str()]);
    assert_eq!(shared["history_coverage"]["indexed"], 1);
    assert_eq!(shared["history_coverage"]["complete"], true);

    let output = repository.run_at(
        &worktree,
        &[
            "search",
            "lateramethyst",
            "--mode",
            "lexical",
            "--freshness",
            "cached",
        ],
    );
    assert_success(&output);
    assert!(result_ids(&parse_json(&output)).is_empty());
}

#[test]
fn patch_search_matches_rename_paths_dates_and_verified_evidence() {
    let repository = Repository::new();
    let original =
        "component configuration\nretry_timeout = 30\ncache_size = 40\nrender_threads = 4\n";
    repository.write("src/original.txt", original);
    repository.commit("Initial component", "2024-01-01T12:00:00Z");
    repository.git(&["mv", "src/original.txt", "src/renamed.txt"]);
    repository.write("src/renamed.txt", &format!("{original}renameneedle\n"));
    let renamed = repository.commit("Move component", "2024-01-03T12:00:00Z");

    for path in ["src/original.txt", "src/renamed.txt"] {
        let found = repository.json(&[
            "search",
            "renameneedle",
            "--mode",
            "lexical",
            "--freshness",
            "wait",
            "--path",
            path,
            "--since",
            "2024-01-03",
            "--until",
            "2024-01-03T23:59:59Z",
        ]);
        assert_eq!(result_ids(&found), [renamed.as_str()]);
        let evidence = found["results"][0]["evidence"]
            .as_array()
            .expect("evidence array")
            .iter()
            .find(|evidence| {
                evidence["excerpt"]
                    .as_str()
                    .is_some_and(|text| text.contains("renameneedle"))
            })
            .expect("result contains matching patch evidence");
        let shown = repository.json(&[
            "show",
            "--evidence",
            evidence["id"].as_str().expect("evidence ID"),
        ]);
        assert_eq!(shown["commit"]["oid"], renamed);
        assert_eq!(shown["evidence"]["verified"], true);
        assert!(
            shown["evidence"]["excerpt"]
                .as_str()
                .expect("verified excerpt")
                .contains("renameneedle")
        );
    }

    let outside_dates = repository.json(&[
        "search",
        "renameneedle",
        "--mode",
        "lexical",
        "--freshness",
        "cached",
        "--until",
        "2024-01-02T23:59:59Z",
    ]);
    assert!(result_ids(&outside_dates).is_empty());
}

#[test]
fn default_hybrid_reports_lexical_fallback_but_semantic_fails_without_a_model() {
    let repository = Repository::new();
    repository.write("cache.txt", "ambercache\n");
    let commit = repository.commit("Initial cache", "2024-01-01T12:00:00Z");
    let fallback = repository.json(&["search", "ambercache", "--freshness", "wait"]);
    assert_eq!(result_ids(&fallback), [commit.as_str()]);
    assert_eq!(fallback["mode_requested"], "hybrid");
    assert_eq!(fallback["mode_used"], "lexical");
    assert!(
        !fallback["warnings"]
            .as_array()
            .expect("fallback warnings")
            .is_empty()
    );

    let output = repository.run(&[
        "search",
        "ambercache",
        "--mode",
        "semantic",
        "--freshness",
        "cached",
    ]);
    assert_eq!(output.status.code(), Some(1));
    let error = parse_json(&output);
    assert_eq!(error["command"], "error");
    assert!(
        error["error"]["code"]
            .as_str()
            .is_some_and(|code| !code.is_empty())
    );
    assert!(error.get("results").is_none());
    assert!(
        !repository.hf_home.exists(),
        "search must not download models"
    );
}

#[test]
fn setup_defaults_are_explicit_and_dry_runs_do_not_create_files() {
    let repository = Repository::new();
    repository.write("readme.txt", "unindexed repository\n");
    repository.commit("Initial commit", "2024-01-01T12:00:00Z");
    assert_eq!(repository.json(&["status"])["command"], "status");
    assert!(!repository.data.exists(), "status is read-only");
    let sync = repository.json(&["sync", "--dry-run"]);
    assert_eq!(sync["command"], "sync");
    assert!(sync["model"].is_null());
    let output = repository.run(&["init", "--dry-run"]);
    let init = parse_json(&output);
    if init["error"]["code"] == "unsupported_cpu" && !cfg!(target_arch = "aarch64") {
        #[cfg(target_arch = "x86_64")]
        assert!(!std::is_x86_feature_detected!("avx2"));
        assert_eq!(output.status.code(), Some(1));
        assert!(!repository.data.exists());
        assert!(!repository.hf_home.exists());
        return;
    }
    assert_success(&output);
    assert_eq!(
        init["model"],
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    );
    assert_eq!(init["model_profile"]["device"], "cpu");
    assert_eq!(
        init["model_profile"]["revision"],
        "b207367332321f8e44f96e224ef15bc607f4dbf0"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        init["model_profile"]["artifact"],
        "onnx/model_qint8_arm64.onnx"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        init["model_profile"]["artifact"],
        "onnx/model_quint8_avx2.onnx"
    );
    let explicit = repository.json(&["init", "--model", "minilm", "--dry-run"]);
    assert_eq!(explicit["model_profile"]["artifact"], "onnx/model.onnx");
    assert!(
        !repository.data.exists(),
        "dry-run must not create an index"
    );
    assert!(
        !repository.hf_home.exists(),
        "dry-run must not download models"
    );
    let offline = repository.run(&["init"]);
    assert_eq!(offline.status.code(), Some(1));
    assert_eq!(parse_json(&offline)["error"]["code"], "model_not_cached");
    assert_eq!(repository.json(&["status"])["readiness"], "uninitialized");
}

#[test]
fn invalid_model_selection_leaves_a_fresh_index_uninitialized() {
    let repository = Repository::new();
    repository.write("readme.txt", "unindexed repository\n");
    repository.commit("Initial commit", "2024-01-01T12:00:00Z");
    for (args, expected_code, exit_code) in [
        (vec!["init", "--model", "unknown/model"], "unknown_model", 1),
        (
            vec!["sync", "--model", "profile:missing"],
            "invalid_config",
            2,
        ),
    ] {
        let output = repository.run(&args);
        assert_eq!(output.status.code(), Some(exit_code));
        assert_eq!(parse_json(&output)["error"]["code"], expected_code);
        let status = repository.json(&["status"]);
        assert_eq!(status["readiness"], "uninitialized");
        assert!(!Path::new(status["index_path"].as_str().unwrap()).exists());
    }
    assert!(!repository.hf_home.exists());
}

#[test]
fn legacy_source_refresh_is_explicit_retryable_and_preserves_embeddings() {
    let repository = Repository::new();
    repository.write("base.txt", "neutral configuration\n");
    let base = repository.commit("fixture-base", "2024-01-01T12:00:00Z");
    repository.write("main.txt", "amber implementation\n");
    repository.commit("fixture-main", "2024-01-02T12:00:00Z");
    repository.git(&["checkout", "--quiet", "-b", "retained", &base]);
    repository.write("side.txt", "cobalt retained implementation\n");
    let retained = repository.commit("fixture-retained", "2024-01-03T12:00:00Z");
    repository.git(&["checkout", "--quiet", "main"]);
    let first = repository.json(&[
        "search",
        "cobalt",
        "--mode",
        "lexical",
        "--freshness",
        "wait",
        "--all-refs",
    ]);
    assert_eq!(first["source_refresh_required"], false);

    let provider = MockProvider::new();
    repository.configure_provider(&provider, "mock-v1", "");
    let initialized = repository.provider_json(&["init", "--model", "profile:mock", "--all-refs"]);
    let status = repository.json(&["status"]);
    let path = Path::new(status["index_path"].as_str().unwrap());
    let connection = rusqlite::Connection::open(path).unwrap();
    let initial_version: String = connection
        .query_row(
            "SELECT value FROM metadata WHERE key='canonical_extraction_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(!initial_version.is_empty());
    connection
        .execute(
            "DELETE FROM metadata WHERE key='canonical_extraction_version'",
            [],
        )
        .unwrap();
    drop(connection);
    provider.clear_requests();

    let before = fs::read(path).unwrap();
    let status = repository.json(&["status"]);
    assert_eq!(status["source_refresh_required"], true);
    assert_eq!(status["readiness"], "source_refresh_required");
    let preview = repository.provider_json(&["sync", "--dry-run", "--ref", "main"]);
    assert_eq!(preview["source_refresh_required"], true);
    assert_eq!(preview["full_reembedding"], false);
    assert_eq!(preview["writes_performed"], false);
    assert_eq!(
        fs::read(path).unwrap(),
        before,
        "inspection must not repair sources"
    );
    for freshness in ["cached", "auto"] {
        let found = repository.json(&[
            "search",
            "cobalt",
            "--mode",
            "lexical",
            "--freshness",
            freshness,
            "--all-refs",
        ]);
        assert_eq!(result_ids(&found), [retained.as_str()]);
        assert_eq!(found["source_refresh_required"], true);
        assert!(found["warnings"].as_array().unwrap().iter().any(|warning| {
            warning
                .as_str()
                .is_some_and(|text| text.contains("nalcos sync"))
        }));
        assert!(found["results"].as_array().unwrap().iter().all(|result| {
            result["evidence"]
                .as_array()
                .unwrap()
                .iter()
                .all(|evidence| evidence["verified"] == true)
        }));
    }
    assert!(provider.requests().is_empty());

    // Fail the canonical transaction on a retained branch outside the requested
    // sync scope. A failed refresh must not publish the new extraction version.
    let connection = rusqlite::Connection::open(path).unwrap();
    connection
        .execute_batch(&format!(
            "CREATE TRIGGER fail_source_refresh BEFORE INSERT ON commits
         WHEN NEW.oid='{retained}' BEGIN SELECT RAISE(ABORT, 'fixture source write failure'); END;"
        ))
        .unwrap();
    let failed = repository.run_provider(&["sync", "--ref", "main"]);
    assert_eq!(failed.status.code(), Some(1));
    assert_eq!(parse_json(&failed)["error"]["code"], "index_error");
    assert_eq!(
        repository.json(&["status"])["source_refresh_required"],
        true
    );
    connection
        .execute_batch("DROP TRIGGER fail_source_refresh")
        .unwrap();
    drop(connection);

    let refreshed = repository.provider_json(&["sync", "--ref", "main"]);
    assert_eq!(
        refreshed["commits_added"], 3,
        "refresh must cover retained indexed branches"
    );
    assert_eq!(refreshed["source_refresh_required"], false);
    assert_eq!(refreshed["documents_embedded"], 0);
    assert_eq!(
        refreshed["active_generation"],
        initialized["active_generation"]
    );
    assert!(
        provider.requests().is_empty(),
        "unchanged sources keep their vectors"
    );
    let connection = rusqlite::Connection::open(path).unwrap();
    let refreshed_version: String = connection
        .query_row(
            "SELECT value FROM metadata WHERE key='canonical_extraction_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(refreshed_version, initial_version);
    let unchanged = repository.provider_json(&["sync", "--ref", "main"]);
    assert_eq!(unchanged["commits_added"], 0);
    assert_eq!(unchanged["documents_embedded"], 0);
}

#[test]
fn evidence_budget_and_argument_errors_are_machine_readable() {
    let repository = Repository::new();
    repository.write(
        "large.txt",
        &format!("budgetmarker {}\n", "readable patch content ".repeat(80)),
    );
    repository.commit("Add a large text file", "2024-01-01T12:00:00Z");
    let found = repository.json(&[
        "search",
        "budgetmarker",
        "--mode",
        "lexical",
        "--freshness",
        "wait",
        "--max-bytes",
        "32",
    ]);
    assert!(!result_ids(&found).is_empty());
    assert_eq!(found["output_truncated"], true);
    let excerpt_bytes: usize = found["results"]
        .as_array()
        .expect("results array")
        .iter()
        .flat_map(|result| result["evidence"].as_array().expect("evidence array"))
        .map(|evidence| evidence["excerpt"].as_str().expect("excerpt").len())
        .sum();
    assert!(
        excerpt_bytes <= 32,
        "excerpts exceeded the requested byte budget"
    );

    let output = repository.run(&["search", "budgetmarker", "--limit", "101"]);
    assert_eq!(output.status.code(), Some(2));
    let invalid = parse_json(&output);
    assert_eq!(invalid["command"], "error");
    assert_eq!(invalid["error"]["code"], "invalid_input");
}

#[test]
fn provider_model_changes_reembed_retained_history_but_query_prefix_changes_do_not() {
    let repository = Repository::new();
    let provider = MockProvider::new();
    repository.configure_provider(&provider, "mock-v1", "");
    repository.write("base.txt", "neutral configuration\n");
    let base = repository.commit("fixture-base", "2024-01-01T12:00:00Z");
    repository.write("amber.txt", "amber cache invalidation\n");
    let amber = repository.commit("fixture-amber", "2024-01-02T12:00:00Z");
    repository.write("cobalt.txt", "cobalt network retry\n");
    let cobalt = repository.commit("fixture-cobalt", "2024-01-03T12:00:00Z");

    let initialized = repository.provider_json(&[
        "init",
        "--model",
        "profile:mock",
        "--revision",
        "cli-pinned-release",
    ]);
    let first_generation = initialized["active_generation"]["id"].clone();
    let full_document_count = initialized["embedding_coverage"]["total"]
        .as_u64()
        .expect("document count");
    assert!(full_document_count >= 6);
    assert_eq!(initialized["embedding_coverage"]["complete"], true);
    let found = repository.provider_json(&[
        "search",
        "cobalt symptom",
        "--mode",
        "semantic",
        "--freshness",
        "cached",
        "--limit",
        "1",
    ]);
    assert_eq!(result_ids(&found), [cobalt.as_str()]);
    assert_eq!(found["mode_used"], "semantic");

    provider.clear_requests();
    let unchanged = repository.provider_json(&["init"]);
    assert_eq!(unchanged["active_generation"]["id"], first_generation);
    assert_eq!(
        unchanged["active_generation"]["revision"],
        "cli-pinned-release"
    );
    assert_eq!(unchanged["documents_embedded"], 0);
    assert!(
        provider
            .requests()
            .iter()
            .all(|request| request.model == "mock-v1"
                && request
                    .inputs
                    .iter()
                    .all(|input| !input.contains("fixture-"))),
        "repeated init may probe the selected model but must not reembed unchanged documents"
    );

    let invalid_device = repository.run_provider(&["sync", "--device", "cpu"]);
    assert_eq!(invalid_device.status.code(), Some(1));
    assert_eq!(
        parse_json(&invalid_device)["error"]["code"],
        "explicit_device_unavailable"
    );
    let after_rejection = repository.json(&["status"]);
    assert_eq!(after_rejection["active_generation"]["id"], first_generation);
    assert!(after_rejection["staging_generation"].is_null());

    let configuration = fs::read_to_string(&repository.config).expect("read fixture configuration");
    assert!(configuration.contains("batch_size = 8"));
    fs::write(
        &repository.config,
        configuration.replace("batch_size = 8", "batch_size = 4\nthreads = 2"),
    )
    .expect("change provider runtime configuration");
    provider.clear_requests();
    let runtime_change = repository.provider_json(&["sync"]);
    assert_eq!(runtime_change["active_generation"]["id"], first_generation);
    assert_eq!(runtime_change["documents_embedded"], 0);
    assert_eq!(runtime_change["runtime"]["index"]["batch_size"], 4);
    let probes = provider.requests();
    assert!(
        !probes.is_empty(),
        "changed runtime settings must be checked"
    );
    assert!(
        probes
            .iter()
            .flat_map(|request| &request.inputs)
            .all(|input| !input.contains("fixture-"))
    );
    provider.clear_requests();
    let unchanged_runtime = repository.provider_json(&["sync"]);
    assert_eq!(
        unchanged_runtime["active_generation"]["id"],
        first_generation
    );
    assert_eq!(unchanged_runtime["documents_embedded"], 0);
    assert!(provider.requests().is_empty());

    repository.configure_provider(&provider, "mock-v1", "query: ");
    provider.clear_requests();
    let preview = repository.provider_json(&["sync", "--dry-run"]);
    assert_eq!(preview["full_reembedding"], false);
    let original_config = fs::read_to_string(&repository.config).unwrap();
    fs::write(
        &repository.config,
        format!("{original_config}\n[index]\nchunk_overlap_tokens = 16\n"),
    )
    .unwrap();
    assert_eq!(
        repository.provider_json(&["sync", "--dry-run"])["full_reembedding"],
        true
    );
    fs::write(&repository.config, original_config).unwrap();
    let query_change = repository.provider_json(&["sync"]);
    assert_eq!(query_change["active_generation"]["id"], first_generation);
    assert_eq!(
        query_change["active_generation"]["fingerprint"],
        initialized["active_generation"]["fingerprint"]
    );
    assert_eq!(
        query_change["active_generation"]["revision"],
        "cli-pinned-release"
    );
    assert_eq!(query_change["documents_embedded"], 0);
    assert_eq!(query_change["active_generation"]["query_prefix"], "query: ");
    assert!(
        provider
            .requests()
            .iter()
            .flat_map(|request| &request.inputs)
            .all(|input| !input.contains("fixture-")),
        "query-only configuration may probe the encoder but must not reembed repository content"
    );
    let found = repository.provider_json(&[
        "search",
        "cobalt symptom",
        "--mode",
        "semantic",
        "--freshness",
        "cached",
        "--limit",
        "1",
    ]);
    assert_eq!(result_ids(&found), [cobalt.as_str()]);
    assert!(provider.requests().iter().any(|request| {
        request.model == "mock-v1" && request.inputs == ["query: cobalt symptom"]
    }));

    let configuration = fs::read_to_string(&repository.config).expect("read fixture configuration");
    assert!(configuration.contains("revision = \"mock-v1-immutable-release\""));
    fs::write(
        &repository.config,
        configuration.replace(
            "revision = \"mock-v1-immutable-release\"",
            "revision = \"configured-release-2\"",
        ),
    )
    .expect("change the configured model revision");
    let configured_revision = repository.provider_json(&["sync"]);
    let second_generation = configured_revision["active_generation"]["id"].clone();
    assert_ne!(second_generation, first_generation);
    assert_eq!(
        configured_revision["active_generation"]["revision"],
        "configured-release-2"
    );
    assert_eq!(
        configured_revision["documents_embedded"],
        full_document_count
    );

    repository.configure_provider(&provider, "mock-v2", "query: ");
    let range = format!("{base}..{amber}");
    let migrated = repository.provider_json(&["sync", "--range", &range]);
    let third_generation = migrated["active_generation"]["id"].clone();
    assert_ne!(third_generation, second_generation);
    assert_ne!(
        migrated["active_generation"]["fingerprint"],
        configured_revision["active_generation"]["fingerprint"]
    );
    assert_eq!(migrated["active_generation"]["dimensions"], 4);
    assert_eq!(migrated["history_coverage"]["total"], 1);
    assert_eq!(
        migrated["documents_embedded"], full_document_count,
        "model replacement must cover retained documents outside the requested sync range"
    );
    let found = repository.provider_json(&[
        "search",
        "cobalt symptom",
        "--mode",
        "semantic",
        "--freshness",
        "cached",
        "--limit",
        "1",
    ]);
    assert_eq!(result_ids(&found), [cobalt.as_str()]);
    assert_eq!(found["embedding_coverage"]["total"], full_document_count);
    assert_eq!(found["embedding_coverage"]["complete"], true);
    assert_eq!(found["active_generation"]["id"], third_generation);

    let configuration = fs::read_to_string(&repository.config).expect("read fixture configuration");
    fs::write(
        &repository.config,
        configuration.replace("[profiles.mock]", "[profiles.mock]\ndevice = 'cpu'"),
    )
    .expect("configure an explicit profile device");
    // Setup can explicitly override a profile's device; subsequent searches still
    // honor that profile unless they also provide an override.
    let profile_device = repository.provider_json(&["sync", "--device", "auto"]);
    assert_eq!(profile_device["active_generation"]["id"], third_generation);
    assert_eq!(profile_device["documents_embedded"], 0);
    let rejected = repository.run_provider(&[
        "search",
        "cobalt symptom",
        "--mode",
        "hybrid",
        "--freshness",
        "cached",
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(
        parse_json(&rejected)["error"]["code"],
        "explicit_device_unavailable"
    );

    fs::write(&repository.config, &configuration).expect("restore the automatic profile device");
    repository.provider_json(&["sync"]);
    fs::write(
        &repository.config,
        configuration.replace("[runtime]", "[runtime]\nquery_device = 'cpu'"),
    )
    .expect("configure an explicit query device");
    let rejected = repository.run_provider(&[
        "search",
        "cobalt symptom",
        "--mode",
        "hybrid",
        "--freshness",
        "cached",
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(
        parse_json(&rejected)["error"]["code"],
        "explicit_device_unavailable"
    );

    fs::write(
        &repository.config,
        configuration.replace("[runtime]", "[runtime]\nindex_device = 'cpu'"),
    )
    .expect("configure an explicit indexing device");
    repository.write("new.txt", "neutral incremental content\n");
    repository.commit("fixture-new", "2024-01-04T12:00:00Z");
    let rejected = repository.run_provider(&[
        "search",
        "cobalt symptom",
        "--mode",
        "hybrid",
        "--freshness",
        "auto",
    ]);
    assert_eq!(rejected.status.code(), Some(1));
    assert_eq!(
        parse_json(&rejected)["error"]["code"],
        "explicit_device_unavailable"
    );
    let after_rejection = repository.json(&["status"]);
    assert_eq!(after_rejection["active_generation"]["id"], third_generation);
    assert!(after_rejection["staging_generation"].is_null());
}

#[test]
fn failed_provider_migration_preserves_active_generation_resumes_and_detects_alias_drift() {
    let repository = Repository::new();
    let provider = MockProvider::new();
    repository.configure_provider_profiles(
        &provider,
        &[("A", "mock-v1", ""), ("B", "mock-v2", "")],
        Some("profile:A"),
    );
    let cobalt = repository.populate_provider_history();
    let initialized = repository.provider_json(&["init"]);
    let first_generation = initialized["active_generation"]["id"].clone();
    let full_document_count = initialized["embedding_coverage"]["total"]
        .as_u64()
        .expect("document count");
    assert!(full_document_count > 16);

    provider.clear_requests();
    provider.fail_after_documents("mock-v2", 16);
    let failed = repository.run_provider(&["sync", "--model", "profile:B"]);
    assert_eq!(failed.status.code(), Some(1));
    assert_eq!(parse_json(&failed)["error"]["code"], "provider_http_error");
    let successful_inputs: usize = provider
        .requests()
        .iter()
        .filter(|request| request.succeeded && request.model == "mock-v2")
        .flat_map(|request| &request.inputs)
        .filter(|input| input.contains("fixture-"))
        .count();
    assert!(
        successful_inputs > 0,
        "the failure must occur after partial work"
    );

    let interrupted = repository.json(&["status"]);
    let staging_generation = interrupted["staging_generation"]["id"].clone();
    assert!(staging_generation.is_number());
    assert_ne!(staging_generation, first_generation);
    assert_eq!(interrupted["active_generation"]["id"], first_generation);
    assert_eq!(interrupted["embedding_coverage"]["complete"], true);
    let old_generation_search = repository.provider_json(&[
        "search",
        "cobalt symptom",
        "--mode",
        "semantic",
        "--freshness",
        "cached",
        "--limit",
        "1",
    ]);
    assert_eq!(result_ids(&old_generation_search), [cobalt.as_str()]);
    assert_eq!(
        old_generation_search["active_generation"]["id"],
        first_generation
    );

    provider.recover();
    provider.clear_requests();
    let resumed = repository.provider_json(&["init"]);
    let remaining = resumed["documents_embedded"]
        .as_u64()
        .expect("resumed document count");
    assert!(remaining > 0 && remaining < full_document_count);
    assert_eq!(resumed["active_generation"]["id"], staging_generation);
    assert_eq!(resumed["active_generation"]["model"], "mock-v2");
    assert_eq!(resumed["embedding_coverage"]["complete"], true);
    assert!(resumed["staging_generation"].is_null());

    provider.change_vectors_under_alias();
    let drifted = repository.run_provider(&[
        "search",
        "cobalt",
        "--mode",
        "semantic",
        "--freshness",
        "cached",
    ]);
    assert_eq!(drifted.status.code(), Some(1));
    assert_eq!(parse_json(&drifted)["error"]["code"], "provider_changed");
    let fallback = repository.provider_json(&["search", "cobalt", "--freshness", "cached"]);
    assert_eq!(fallback["mode_requested"], "hybrid");
    assert_eq!(fallback["mode_used"], "lexical");
    assert_eq!(result_ids(&fallback), [cobalt.as_str()]);
    assert!(
        fallback["warnings"]
            .as_array()
            .expect("fallback warnings")
            .iter()
            .any(|warning| warning
                .as_str()
                .is_some_and(|text| text.contains("provider_changed")))
    );
    assert_eq!(fallback["active_generation"]["id"], staging_generation);
}

#[test]
fn explicit_model_selection_cancels_staging_and_forced_reembedding_resumes_its_own_probe() {
    let repository = Repository::new();
    let provider = MockProvider::new();
    repository.configure_provider_profiles(
        &provider,
        &[("A", "mock-v1", ""), ("B", "mock-v2", "")],
        None,
    );
    let cobalt = repository.populate_provider_history();
    let initialized = repository.provider_json(&["init", "--model", "profile:A"]);
    let first_generation = initialized["active_generation"]["id"].clone();
    let fingerprint = initialized["active_generation"]["fingerprint"].clone();
    let full_document_count = initialized["embedding_coverage"]["total"]
        .as_u64()
        .expect("document count");

    provider.clear_requests();
    provider.fail_after_documents("mock-v2", 16);
    let failed = repository.run_provider(&["sync", "--model", "profile:B"]);
    assert_eq!(failed.status.code(), Some(1));
    assert_eq!(parse_json(&failed)["error"]["code"], "provider_http_error");
    let interrupted = repository.json(&["status"]);
    assert_eq!(interrupted["active_generation"]["id"], first_generation);
    assert_eq!(interrupted["staging_generation"]["model"], "mock-v2");

    let cancelled = repository.provider_json(&["sync", "--model", "profile:A"]);
    assert_eq!(cancelled["active_generation"]["id"], first_generation);
    assert!(cancelled["staging_generation"].is_null());
    provider.clear_requests();
    let stays_active = repository.provider_json(&["sync"]);
    assert_eq!(stays_active["active_generation"]["id"], first_generation);
    assert!(stays_active["staging_generation"].is_null());
    assert!(provider.requests().is_empty());

    provider.change_vectors_under_alias();
    provider.fail_after_documents("mock-v1", 16);
    let failed_reembedding = repository.run_provider(&["sync", "--reembed"]);
    assert_eq!(failed_reembedding.status.code(), Some(1));
    assert_eq!(
        parse_json(&failed_reembedding)["error"]["code"],
        "provider_http_error",
        "an explicit forced reembedding must establish a new provider probe"
    );
    let interrupted = repository.json(&["status"]);
    let staging_generation = interrupted["staging_generation"]["id"].clone();
    assert!(staging_generation.is_number());
    assert_ne!(staging_generation, first_generation);
    assert_eq!(interrupted["active_generation"]["id"], first_generation);
    assert_eq!(
        interrupted["staging_generation"]["fingerprint"],
        fingerprint
    );

    provider.recover();
    provider.clear_requests();
    let resumed = repository.provider_json(&["sync"]);
    assert_eq!(resumed["active_generation"]["id"], staging_generation);
    assert_eq!(resumed["active_generation"]["fingerprint"], fingerprint);
    assert!(resumed["staging_generation"].is_null());
    let remaining = resumed["documents_embedded"]
        .as_u64()
        .expect("resumed document count");
    assert!(remaining > 0 && remaining < full_document_count);
    let verified = repository.provider_json(&[
        "search",
        "cobalt symptom",
        "--mode",
        "semantic",
        "--freshness",
        "cached",
        "--limit",
        "1",
    ]);
    assert_eq!(result_ids(&verified), [cobalt.as_str()]);
    assert_eq!(verified["active_generation"]["id"], staging_generation);
}

#[test]
fn path_filters_limit_evidence_and_missing_winner_is_replaced_by_valid_evidence() {
    let repository = Repository::new();
    let provider = MockProvider::new();
    repository.configure_provider(&provider, "mock-v1", "");
    repository.write("kept.txt", "cobalt keptneedle\n");
    repository.write("gone.txt", "cobalt goneneedle\n");
    let commit = repository.commit("fixture-mixed", "2024-01-01T12:00:00Z");
    repository.provider_json(&["init", "--model", "profile:mock"]);

    let unrelated = repository.json(&[
        "search",
        "goneneedle",
        "--mode",
        "lexical",
        "--freshness",
        "cached",
        "--path",
        "kept.txt",
    ]);
    assert!(
        result_ids(&unrelated).is_empty(),
        "touching the selected path must not make every patch in a commit searchable"
    );
    for mode in ["lexical", "semantic", "hybrid"] {
        let selected = repository.provider_json(&[
            "search",
            "cobalt",
            "--mode",
            mode,
            "--freshness",
            "cached",
            "--path",
            "kept.txt",
        ]);
        assert_eq!(result_ids(&selected), [commit.as_str()]);
        assert!(
            selected["results"][0]["evidence"]
                .as_array()
                .expect("evidence array")
                .iter()
                .all(|evidence| evidence["new_path"]["display"] == "kept.txt")
        );
    }

    let winner = repository.json(&[
        "search",
        "cobalt",
        "--mode",
        "lexical",
        "--freshness",
        "cached",
    ]);
    let missing_evidence = winner["results"][0]["evidence"][0]["id"]
        .as_str()
        .expect("winning evidence ID");
    let missing_blob = winner["results"][0]["evidence"][0]["new_blob"]
        .as_str()
        .expect("winning patch blob");
    let object = repository
        .path
        .join(".git/objects")
        .join(&missing_blob[..2])
        .join(&missing_blob[2..]);
    let held_object = repository
        ._directory
        .path()
        .join("temporarily-missing-blob");
    fs::rename(&object, &held_object).expect("temporarily withhold a fixture source object");
    for mode in ["lexical", "semantic", "hybrid"] {
        let refilled = repository.provider_json(&[
            "search",
            "cobalt",
            "--mode",
            mode,
            "--freshness",
            "cached",
            "--limit",
            "1",
        ]);
        assert_eq!(result_ids(&refilled), [commit.as_str()]);
        assert!(
            refilled["results"][0]["evidence"]
                .as_array()
                .expect("refilled evidence")
                .iter()
                .all(|evidence| evidence["id"] != missing_evidence
                    && evidence["new_blob"] != missing_blob
                    && evidence["verified"] == true)
        );
    }
    fs::rename(held_object, object).expect("restore the fixture source object");
}

#[test]
fn queued_sync_observes_the_model_selected_by_the_writer_ahead_of_it() {
    let repository = Repository::new();
    let provider = MockProvider::new();
    repository.configure_provider_profiles(
        &provider,
        &[("A", "mock-v1", ""), ("B", "mock-v2", "")],
        Some("profile:A"),
    );
    repository.write("cobalt.txt", "cobalt concurrency fixture\n");
    repository.commit("fixture-cobalt", "2024-01-01T12:00:00Z");
    let initialized = repository.provider_json(&["init", "--model", "profile:A"]);
    let first_generation = initialized["active_generation"]["id"].clone();

    provider.clear_requests();
    let held_request = provider.hold_next_request("mock-v2");
    let mut first_command = repository.command_at(&repository.path);
    first_command.args(["sync", "--model", "profile:B"]);
    let first_writer = RunningCli::spawn(first_command);
    held_request.wait_until_called();
    let during_migration = repository.json(&["status"]);
    assert_eq!(
        during_migration["active_generation"]["id"],
        first_generation
    );
    assert_eq!(during_migration["staging_generation"]["model"], "mock-v2");

    let mut second_command = repository.command_at(&repository.path);
    second_command.args(["sync", "--verbose"]);
    let mut second_writer = RunningCli::spawn(second_command);
    let stderr = second_writer.take_stderr();
    let (waiting_tx, waiting_rx) = mpsc::channel();
    let stderr_reader = thread::spawn(move || {
        let mut notification = Some(waiting_tx);
        let mut collected = String::new();
        for line in BufReader::new(stderr).lines() {
            let line = line.expect("read queued writer progress");
            if line.starts_with("Waiting for index writer lock: ")
                && let Some(waiting) = notification.take()
            {
                waiting.send(()).expect("report writer contention");
            }
            collected.push_str(&line);
            collected.push('\n');
        }
        collected.into_bytes()
    });
    waiting_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("second CLI writer must report actual lock contention");
    held_request.release();

    let first_output = first_writer.finish();
    assert_success(&first_output);
    let first = parse_json(&first_output);
    assert_eq!(first["active_generation"]["model"], "mock-v2");
    assert_ne!(first["active_generation"]["id"], first_generation);
    let mut second_output = second_writer.finish();
    second_output.stderr = stderr_reader.join().expect("join queued writer progress");
    assert_success(&second_output);
    let second = parse_json(&second_output);
    assert_eq!(
        second["active_generation"]["id"],
        first["active_generation"]["id"]
    );
    assert_eq!(second["active_generation"]["model"], "mock-v2");
    assert_eq!(second["documents_embedded"], 0);
    let final_status = repository.json(&["status"]);
    assert_eq!(
        final_status["active_generation"]["id"],
        first["active_generation"]["id"]
    );
    assert!(final_status["staging_generation"].is_null());
    assert!(
        provider
            .requests()
            .iter()
            .all(|request| request.model == "mock-v2")
    );
}

#[test]
fn amended_rebased_and_squashed_history_returns_only_commits_in_the_requested_graph() {
    let repository = Repository::new();
    repository.write("base.txt", "initial settings\n");
    repository.commit("Initial settings", "2024-01-01T12:00:00Z");
    repository.git(&["switch", "-c", "topic"]);
    repository.write("feature.txt", "amethystneedle initial design\n");
    let original = repository.commit("Add violet capability", "2024-01-02T12:00:00Z");
    let search = |freshness: &str| {
        repository.json(&[
            "search",
            "amethystneedle",
            "--mode",
            "lexical",
            "--freshness",
            freshness,
        ])
    };
    assert_eq!(result_ids(&search("wait")), [original.as_str()]);

    repository.write("feature.txt", "amethystneedle refined design\n");
    repository.git(&["add", "--all"]);
    let amended_output = repository
        .git_command()
        .env("GIT_COMMITTER_DATE", "2024-01-03T12:00:00Z")
        .args([
            "commit",
            "--amend",
            "--quiet",
            "-m",
            "Refine violet capability",
        ])
        .output()
        .expect("amend fixture commit");
    assert_success(&amended_output);
    let amended = repository.git(&["rev-parse", "HEAD"]);
    assert_ne!(amended, original);
    assert!(result_ids(&search("cached")).is_empty());
    assert_eq!(result_ids(&search("wait")), [amended.as_str()]);

    repository.git(&["switch", "main"]);
    repository.write("main.txt", "unrelated mainline work\n");
    repository.commit("Advance mainline", "2024-01-04T12:00:00Z");
    repository.git(&["switch", "topic"]);
    repository.git(&["rebase", "main"]);
    let rebased = repository.git(&["rev-parse", "HEAD"]);
    assert_ne!(rebased, amended);
    assert!(result_ids(&search("cached")).is_empty());
    assert_eq!(result_ids(&search("wait")), [rebased.as_str()]);

    repository.write("followup.txt", "supporting feature behavior\n");
    repository.commit("Add supporting behavior", "2024-01-05T12:00:00Z");
    search("wait");
    repository.git(&["switch", "main"]);
    repository.git(&["merge", "--squash", "topic"]);
    let squashed = repository.commit("Ship violet capability", "2024-01-06T12:00:00Z");
    assert!(result_ids(&search("cached")).is_empty());
    assert_eq!(result_ids(&search("wait")), [squashed.as_str()]);
    let all_refs = repository.json(&[
        "search",
        "amethystneedle",
        "--mode",
        "lexical",
        "--freshness",
        "cached",
        "--all-refs",
    ]);
    let live = result_ids(&all_refs);
    assert_eq!(live.len(), 2);
    assert!(live.contains(&rebased.as_str()));
    assert!(live.contains(&squashed.as_str()));
    assert!(!live.contains(&original.as_str()));
    assert!(!live.contains(&amended.as_str()));
}
