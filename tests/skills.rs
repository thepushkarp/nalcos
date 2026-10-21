//! Keeps the published agent skill, the CLI help examples, and the Claude Code plugin manifests
//! consistent with the CLI. The skill is installed on its own (for example by `npx skills add`),
//! so every link must stay inside its directory and every documented `nalcos` command must
//! parse with this build.

use clap::{CommandFactory, Parser, error::ErrorKind};
use nalcos::cli::Cli;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

const FRONTMATTER_FIELDS: &[&str] = &[
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
];

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn skill_directories() -> Vec<PathBuf> {
    let mut directories: Vec<_> = fs::read_dir(root().join("skills"))
        .expect("read skills directory")
        .map(|entry| entry.expect("read skills entry").path())
        .filter(|path| path.join("SKILL.md").is_file())
        .collect();
    directories.sort();
    assert!(!directories.is_empty(), "skills/*/SKILL.md is missing");
    directories
}

fn markdown_files(directory: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("read skill directory") {
            let path = entry.expect("read skill entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "md") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

/// Splits the YAML frontmatter into single-line `key: value` fields. Nested or multi-line
/// values are rejected rather than half-parsed, and plain scalars that YAML would read
/// differently (an indicator first character, `: `, or ` #`) must be quoted.
fn frontmatter(path: &Path, text: &str) -> (Vec<(String, String)>, usize) {
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("---"),
        "{}: missing frontmatter",
        path.display()
    );
    let mut fields = Vec::new();
    for (index, line) in lines.enumerate() {
        if line == "---" {
            return (fields, index + 2);
        }
        let (key, value) = line
            .split_once(':')
            .unwrap_or_else(|| panic!("{}: unsupported frontmatter line {line:?}", path.display()));
        let value = value.trim();
        assert!(
            !key.starts_with(' ') && !value.is_empty(),
            "{}: frontmatter field {key:?} must be a single-line scalar",
            path.display()
        );
        let value = if let Some(quoted) = value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
        {
            quoted.to_owned()
        } else {
            assert!(
                !value.starts_with(|c| "[]{}>|*&!%@`'\",#?:-".contains(c))
                    && !value.contains(": ")
                    && !value.contains(" #"),
                "{}: quote the YAML value of {key:?}",
                path.display()
            );
            value.to_owned()
        };
        fields.push((key.to_owned(), value));
    }
    panic!("{}: unterminated frontmatter", path.display());
}

fn valid_skill_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name.split('-').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        })
}

#[test]
fn skill_frontmatter_follows_the_agent_skills_specification() {
    for directory in skill_directories() {
        let path = directory.join("SKILL.md");
        let text = read(&path);
        let (fields, body_start) = frontmatter(&path, &text);
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.as_str())
        };
        for (key, _) in &fields {
            assert!(
                FRONTMATTER_FIELDS.contains(&key.as_str()),
                "{}: unknown frontmatter field {key:?}",
                path.display()
            );
        }
        let name = field("name").expect("frontmatter name");
        assert!(valid_skill_name(name), "invalid skill name {name:?}");
        assert_eq!(
            Some(name),
            directory.file_name().and_then(|name| name.to_str()),
            "skill name must match its directory"
        );
        let description = field("description").expect("frontmatter description");
        assert!(
            (1..=1024).contains(&description.chars().count()),
            "{name}: description must be 1-1024 characters"
        );
        if let Some(compatibility) = field("compatibility") {
            assert!(
                (1..=500).contains(&compatibility.chars().count()),
                "{name}: compatibility must be 1-500 characters"
            );
        }
        let body_lines = text.lines().count() - body_start;
        assert!(
            body_lines < 500,
            "{name}: SKILL.md body has {body_lines} lines"
        );
    }
}

/// Returns the targets of inline Markdown links, `[text](target)`, outside code spans.
fn link_targets(text: &str) -> Vec<String> {
    let mut targets = Vec::new();
    let mut fenced = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        let mut in_code = false;
        let mut rest = line;
        while let Some(position) = rest.find(['`', ']']) {
            let (marker, after) = (rest.as_bytes()[position], &rest[position + 1..]);
            rest = after;
            if marker == b'`' {
                in_code = !in_code;
            } else if !in_code
                && let Some(link) = after.strip_prefix('(')
                && let Some(end) = link.find(')')
            {
                targets.push(link[..end].to_owned());
                rest = &link[end + 1..];
            }
        }
    }
    targets
}

#[test]
fn skill_links_resolve_inside_the_skill() {
    for directory in skill_directories() {
        let skill_root = directory.canonicalize().expect("canonical skill directory");
        for path in markdown_files(&directory) {
            for target in link_targets(&read(&path)) {
                if target.starts_with("https://") || target.starts_with('#') {
                    continue;
                }
                let file = target.split('#').next().expect("link path");
                let resolved = path
                    .parent()
                    .expect("markdown parent")
                    .join(file)
                    .canonicalize()
                    .unwrap_or_else(|error| {
                        panic!("{}: broken link {target:?}: {error}", path.display())
                    });
                assert!(
                    resolved.starts_with(&skill_root),
                    "{}: link {target:?} leaves the skill directory",
                    path.display()
                );
            }
        }
    }
}

/// Splits one shell command line into words, honoring single and double quotes and
/// backslash escapes. Words end at an unquoted comment, pipe, redirection, or separator,
/// because only the `nalcos` invocation itself is checked.
fn shell_words(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut started = false;
    let mut quote = None;
    let mut characters = line.chars();
    while let Some(character) = characters.next() {
        match (quote, character) {
            (Some(open), c) if c == open => quote = None,
            (Some('"'), '\\') => word.extend(characters.next()),
            (Some(_), c) => word.push(c),
            (None, '\'' | '"') => {
                quote = Some(character);
                started = true;
            }
            (None, '\\') => {
                word.extend(characters.next());
                started = true;
            }
            (None, c) if c.is_whitespace() => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            (None, '#') if !started => break,
            (None, '|' | ';' | '&' | '>' | '<') => break,
            (None, c) => {
                word.push(c);
                started = true;
            }
        }
    }
    assert!(quote.is_none(), "unterminated quote in {line:?}");
    if started {
        words.push(word);
    }
    words
}

/// Collects `nalcos` invocations from `sh` and `bash` fences, joining continued lines.
fn documented_commands(text: &str) -> Vec<String> {
    let mut commands = Vec::new();
    let mut shell = None;
    let mut pending = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(info) = trimmed.strip_prefix("```") {
            shell = match shell {
                None => Some(matches!(info.trim(), "sh" | "bash")),
                Some(_) => None,
            };
            continue;
        }
        if shell != Some(true) {
            continue;
        }
        pending.push_str(trimmed.strip_suffix('\\').unwrap_or(trimmed));
        if trimmed.ends_with('\\') {
            pending.push(' ');
            continue;
        }
        let command = std::mem::take(&mut pending);
        if command.starts_with("nalcos ") {
            commands.push(command);
        }
    }
    commands
}

fn assert_parses(source: &str, command: &str) {
    match Cli::try_parse_from(shell_words(command)) {
        Ok(_) => {}
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) => {}
        Err(error) => panic!("{source}: {command}\n{error}"),
    }
}

#[test]
fn documented_skill_commands_parse_with_this_cli() {
    let mut checked = 0;
    for directory in skill_directories() {
        for path in markdown_files(&directory) {
            for command in documented_commands(&read(&path)) {
                assert_parses(&path.display().to_string(), &command);
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "the skill documents no nalcos commands");
}

#[test]
fn help_examples_parse_with_this_cli() {
    let mut checked = 0;
    for subcommand in Cli::command().get_subcommands() {
        let Some(examples) = subcommand.get_after_help() else {
            continue;
        };
        for line in examples.to_string().lines().map(str::trim) {
            if line.starts_with("nalcos ") {
                assert_parses(&format!("{} --help", subcommand.get_name()), line);
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no subcommand help shows examples");
}

#[test]
fn shell_words_respect_quotes_and_stop_at_operators() {
    assert_eq!(
        shell_words(r#"nalcos search "a \"b\" c" --path 'src/**' x\ y # note"#),
        ["nalcos", "search", r#"a "b" c"#, "--path", "src/**", "x y"]
    );
    assert_eq!(
        shell_words("nalcos --json status | jq .readiness"),
        ["nalcos", "--json", "status"]
    );
    assert_eq!(shell_words("nalcos search ''"), ["nalcos", "search", ""]);
}

fn manifest(name: &str) -> Value {
    let path = root().join(".claude-plugin").join(name);
    serde_json::from_str(&read(&path))
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn nonempty<'a>(value: &'a Value, field: &str) -> &'a str {
    value[field]
        .as_str()
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| panic!("plugin manifest needs a nonempty {field:?}"))
}

#[test]
fn plugin_marketplace_lists_this_repository_as_its_plugin() {
    let plugin = manifest("plugin.json");
    let name = nonempty(&plugin, "name");
    nonempty(&plugin, "description");
    assert_eq!(plugin["license"], env!("CARGO_PKG_LICENSE"));
    // An explicit plugin version pins installs; one that differs from the crate drifts.
    if let Some(version) = plugin.get("version") {
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    let marketplace = manifest("marketplace.json");
    nonempty(&marketplace, "name");
    nonempty(&marketplace["owner"], "name");
    let entries = marketplace["plugins"]
        .as_array()
        .expect("marketplace plugins array");
    let entry = entries
        .iter()
        .find(|entry| entry["name"] == name)
        .unwrap_or_else(|| panic!("marketplace does not list plugin {name:?}"));
    nonempty(entry, "description");
    let source = nonempty(entry, "source");
    assert!(
        source.starts_with("./") && !source.contains(".."),
        "plugin source must be a relative path inside the repository"
    );
    let source = root().join(source);
    assert!(source.join(".claude-plugin/plugin.json").is_file());
    assert!(
        source.join("skills").is_dir(),
        "the plugin source must contain the skills directory"
    );
}
