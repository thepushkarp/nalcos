use crate::{
    embedding::{Backend, ModelProfile, RuntimeInfo},
    error::{AppError, Result},
};
use serde_json::{Value, json};
use std::{
    io::{self, Write},
    path::Path,
};

pub fn compact_search(value: &Value, repo: &Path) -> Result<Value> {
    let mut compact = json!({"compact":true});
    for field in [
        "schema_version",
        "command",
        "mode_requested",
        "mode_used",
        "freshness",
        "history_coverage",
        "embedding_coverage",
        "warnings",
        "output_truncated",
    ] {
        compact[field] = value[field].clone();
    }
    compact["source_coverage"] = json!({
        "recorded_omissions":value["source_coverage"]["recorded_omissions"],
        "categories":value["source_coverage"]["categories"],
    });
    let mut results = Vec::new();
    for (index, result) in value["results"]
        .as_array()
        .expect("search results")
        .iter()
        .enumerate()
    {
        let commit = &result["commit"];
        let oid = commit["oid"].as_str().expect("commit OID");
        let timestamp = commit["committed_at"].as_i64().expect("commit timestamp");
        let committed_at = chrono::DateTime::from_timestamp(timestamp, 0)
            .ok_or_else(|| {
                AppError::new(
                    "invalid_index",
                    format!("Cannot format committed_at {timestamp} for commit {oid} as RFC 3339"),
                )
            })?
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let evidence: Vec<_> = result["evidence"]
            .as_array()
            .expect("search evidence")
            .iter()
            .map(|item| {
                let old = item["old_path"]["display"].as_str();
                let new = item["new_path"]["display"].as_str();
                let path = match (old, new) {
                    (Some(old), Some(new)) if old != new => Some(format!("{old} -> {new}")),
                    _ => new.or(old).map(str::to_owned),
                };
                let show = json!(["nalcos", "--repo", repo, "show", "--evidence", item["id"]]);
                let mut evidence = json!({"path":path,"show":show});
                for field in [
                    "id",
                    "kind",
                    "role",
                    "excerpt",
                    "verified",
                    "excerpt_truncated",
                    "metadata_only",
                ] {
                    evidence[field] = item[field].clone();
                }
                if item["kind"] == "diff" {
                    for field in ["old_start", "old_lines", "new_start", "new_lines"] {
                        evidence[field] = item[field].clone();
                    }
                }
                evidence
            })
            .collect();
        results.push(json!({
            "rank":index + 1,"commit":oid,"short":&oid[..oid.len().min(12)],
            "subject":commit["message"].as_str().unwrap_or("").lines().next().unwrap_or(""),
            "committed_at":committed_at,"author":commit["author_name"],"evidence":evidence,
        }));
    }
    compact["results"] = json!(results);
    Ok(compact)
}

pub fn bound_search(value: &mut Value, limit: usize) -> Result<()> {
    value["dropped_results"] = json!(0);
    let mut dropped = 0;
    while search_output_bytes(value)? > limit {
        value["output_truncated"] = json!(true);
        let results = value["results"].as_array_mut().expect("search results");
        if results.len() > 1 {
            results.pop();
            dropped += 1;
            value["dropped_results"] = json!(dropped);
            continue;
        }
        let Some(index) = value["results"][0]["evidence"]
            .as_array()
            .and_then(|items| {
                items.iter().rposition(|item| {
                    item["excerpt"]
                        .as_str()
                        .is_some_and(|text| !text.is_empty())
                })
            })
        else {
            break;
        };
        let original = value["results"][0]["evidence"][index]["excerpt"]
            .as_str()
            .expect("evidence excerpt")
            .to_owned();
        // JSON escaping and output_bytes itself change the size, so search for the
        // longest excerpt prefix whose serialized response fits.
        let (mut low, mut high) = (0, original.len() - 1);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            shorten_excerpt(
                &mut value["results"][0]["evidence"][index],
                &original,
                middle,
            );
            if search_output_bytes(value)? <= limit {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        shorten_excerpt(&mut value["results"][0]["evidence"][index], &original, low);
    }
    Ok(())
}

fn shorten_excerpt(item: &mut Value, original: &str, limit: usize) {
    let (excerpt, _) = truncate_utf8(original, limit);
    item["excerpt"] = json!(excerpt);
    item["excerpt_truncated"] = json!(true);
    if let Some(start) = item["excerpt_byte_start"].as_u64() {
        item["excerpt_byte_end"] = json!(start + excerpt.len() as u64);
    }
}

fn search_output_bytes(value: &mut Value) -> Result<usize> {
    loop {
        // emit() appends one newline. Iterate until the decimal width of this
        // self-inclusive count no longer changes the serialized size.
        let bytes = serde_json::to_vec(value)?.len() + 1;
        if value["output_bytes"].as_u64() == Some(bytes as u64) {
            return Ok(bytes);
        }
        value["output_bytes"] = json!(bytes);
    }
}

pub fn emit(value: &Value, json: bool) -> Result<()> {
    let mut out = io::stdout().lock();
    if json {
        serde_json::to_writer(&mut out, value)?;
        writeln!(out)?;
        return Ok(());
    }
    match value["command"].as_str().unwrap_or("") {
        "search" => {
            let mode = value["mode_used"].as_str().unwrap_or("unknown");
            writeln!(
                out,
                "{} search · {}/{} commits indexed",
                mode, value["history_coverage"]["indexed"], value["history_coverage"]["total"]
            )?;
            if let Some(results) = value["results"].as_array() {
                if results.is_empty() {
                    writeln!(out, "No matches in the available coverage.")?;
                }
                for result in results {
                    let commit = &result["commit"];
                    let oid = commit["oid"].as_str().unwrap_or("");
                    let title = commit["message"]
                        .as_str()
                        .unwrap_or("")
                        .lines()
                        .next()
                        .unwrap_or("");
                    writeln!(out, "\n{}  {}", &oid[..oid.len().min(12)], title)?;
                    if let Some(evidence) = result["evidence"].as_array() {
                        for item in evidence {
                            if let Some(path) = item["new_path"]["display"]
                                .as_str()
                                .or_else(|| item["old_path"]["display"].as_str())
                            {
                                writeln!(out, "  {path}")?;
                            }
                            let label = if item["metadata_only"] == true {
                                " (file metadata only)"
                            } else if item["role"] == "supporting_code" {
                                " (supporting code)"
                            } else {
                                ""
                            };
                            writeln!(
                                out,
                                "  evidence: {}{label}",
                                item["id"].as_str().unwrap_or("")
                            )?;
                            for line in item["excerpt"].as_str().unwrap_or("").lines() {
                                writeln!(out, "    {line}")?;
                            }
                            write_patch_hint(&mut out, item, &value["repository"]["work_dir"])?;
                        }
                    }
                }
            }
        }
        "show" => {
            if let Some(evidence) = value.get("evidence") {
                if evidence["metadata_only"] == true {
                    writeln!(
                        out,
                        "File metadata only; this evidence ID has no textual hunk."
                    )?;
                }
                writeln!(out, "{}", evidence["excerpt"].as_str().unwrap_or(""))?;
                write_patch_hint(&mut out, evidence, &value["repository"]["work_dir"])?;
            } else {
                writeln!(
                    out,
                    "commit {}",
                    value["commit"]["oid"].as_str().unwrap_or("")
                )?;
                writeln!(out, "{}", value["commit"]["message"].as_str().unwrap_or(""))?;
                writeln!(out, "{}", value["patch"].as_str().unwrap_or(""))?;
            }
        }
        "status" => {
            writeln!(
                out,
                "Index: {}",
                value["readiness"].as_str().unwrap_or("unknown")
            )?;
            writeln!(
                out,
                "History: {}/{} commits",
                value["history_coverage"]["indexed"], value["history_coverage"]["total"]
            )?;
            writeln!(
                out,
                "Embeddings: {}/{} documents",
                value["embedding_coverage"]["indexed"], value["embedding_coverage"]["total"]
            )?;
            if value["writer_active"] == true {
                writeln!(out, "Indexing in progress in another nalcos process")?;
            }
            if value["staging_progress"].is_object() {
                writeln!(
                    out,
                    "Pending generation: {}/{} documents completed (across retained history)",
                    value["staging_progress"]["completed"], value["staging_progress"]["total"]
                )?;
                if let Some(command) = value["staging_progress"]["resume_command"].as_str() {
                    writeln!(out, "Resume with: {command}")?;
                }
            }
            if let Some(profile) = value["active_generation"]["model"].as_str() {
                writeln!(out, "Model: {profile}")?;
            }
            if let Some(device) = value["runtime"]["selected_device"].as_str() {
                writeln!(out, "Last indexing device: {device}")?;
            }
            if value["check"].is_object() {
                serde_json::to_writer_pretty(&mut out, &value["check"])?;
                writeln!(out)?;
            }
        }
        "omissions" => write_omissions(&mut out, value)?,
        "init" | "sync" => {
            if value["dry_run"] == true {
                serde_json::to_writer_pretty(&mut out, value)?;
                writeln!(out)?;
            } else {
                writeln!(
                    out,
                    "Indexed {} new commits; embedded {} documents.",
                    value["commits_added"], value["documents_embedded"]
                )?;
                writeln!(
                    out,
                    "History: {}/{} commits; embeddings: {}/{} documents",
                    value["history_coverage"]["indexed"],
                    value["history_coverage"]["total"],
                    value["embedding_coverage"]["indexed"],
                    value["embedding_coverage"]["total"]
                )?;
                if let Some(runtime) = value["indexing_runtime"].as_object() {
                    let info: RuntimeInfo = serde_json::from_value(Value::Object(runtime.clone()))?;
                    writeln!(out, "Final indexing device: {}", runtime_device(&info))?;
                }
            }
        }
        _ => {
            serde_json::to_writer_pretty(&mut out, value)?;
            writeln!(out)?;
        }
    }
    if value["output_truncated"] == true {
        writeln!(
            out,
            "\nEvidence output was truncated; use show for more context."
        )?;
    }
    if let Some(warnings) = value["warnings"].as_array() {
        for warning in warnings {
            eprintln!(
                "warning: {}",
                warning.as_str().unwrap_or("incomplete coverage")
            );
        }
    }
    Ok(())
}

fn write_omissions(out: &mut impl Write, value: &Value) -> Result<()> {
    writeln!(
        out,
        "History: {}/{} commits indexed",
        value["history_coverage"]["indexed"], value["history_coverage"]["total"]
    )?;
    writeln!(
        out,
        "Source omissions: {} recorded across all paths in selected commits",
        value["source_coverage"]["recorded_omissions"]
    )?;
    if let Some(reasons) = value["source_coverage"]["reasons"].as_array() {
        for reason in reasons {
            writeln!(
                out,
                "  {}: {} ({}) — {}",
                reason["reason"].as_str().unwrap_or("other"),
                reason["count"],
                reason["category"].as_str().unwrap_or("other"),
                reason["action"].as_str().unwrap_or("")
            )?;
        }
    }
    if let Some(records) = value["records"].as_array() {
        for record in records {
            let oid = record["commit_oid"].as_str().unwrap_or("");
            let path = record["path"]["display"]
                .as_str()
                .unwrap_or("(commit message or parent)");
            writeln!(
                out,
                "\n{}  {} [{}]",
                &oid[..oid.len().min(12)],
                path,
                record["reason"].as_str().unwrap_or("other")
            )?;
            writeln!(out, "  {}", record["detail"].as_str().unwrap_or(""))?;
        }
    }
    if let Some(offset) = value["next_offset"].as_u64() {
        writeln!(
            out,
            "\nMore records: repeat the command with --offset {offset} and the same scope options."
        )?;
    }
    Ok(())
}

fn runtime_device(info: &RuntimeInfo) -> String {
    if matches!(info.provider.as_str(), "OpenAi" | "Ollama") {
        return format!("remote ({}, provider-managed)", info.provider);
    }
    let backend = match info.provider.as_str() {
        "Onnx" => "ONNX Runtime",
        "Gguf" => "GGUF / llama.cpp",
        other => other,
    };
    let device = info.selected_device.to_string();
    if info.device_name.is_empty() || info.device_name.eq_ignore_ascii_case(&device) {
        format!("{device} ({backend})")
    } else {
        format!("{device} ({backend}; {})", info.device_name)
    }
}

/// Only runtime metadata from an encoder doing indexing work is passed here.
pub fn indexing_runtime(
    model: &ModelProfile,
    current: &RuntimeInfo,
    previous: Option<&RuntimeInfo>,
) -> Result<()> {
    write_indexing_runtime(&mut io::stderr().lock(), model, current, previous)
}

fn write_indexing_runtime(
    out: &mut impl Write,
    model: &ModelProfile,
    current: &RuntimeInfo,
    previous: Option<&RuntimeInfo>,
) -> Result<()> {
    if previous.is_none() {
        writeln!(out, "Embedding model: {}", model.id)?;
        let kind = if matches!(model.backend, Backend::OpenAi | Backend::Ollama) {
            "provider"
        } else {
            "device"
        };
        writeln!(out, "Indexing {kind}: {}", runtime_device(current))?;
    } else if previous.is_some_and(|previous| {
        previous.selected_device != current.selected_device
            || previous.device_name != current.device_name
            || previous.provider != current.provider
    }) {
        writeln!(out, "Indexing device changed: {}", runtime_device(current))?;
    }
    let reported = previous.map_or(0, |info| info.fallback_reasons.len());
    for reason in current.fallback_reasons.iter().skip(reported) {
        writeln!(out, "Indexing runtime: {reason}")?;
    }
    Ok(())
}

fn write_patch_hint(out: &mut impl Write, evidence: &Value, repository: &Value) -> Result<()> {
    if let Some(args) = evidence["patch_lookup"]["args"].as_array() {
        let args: Vec<_> = args
            .iter()
            .filter_map(Value::as_str)
            .map(shell_quote)
            .collect();
        let scope = if evidence["patch_lookup"]["file_scoped"] == false {
            "Commit/parent patch"
        } else {
            "File patch"
        };
        writeln!(
            out,
            "  {scope}: nalcos --repo {} {}",
            shell_quote(repository.as_str().unwrap_or(".")),
            args.join(" ")
        )?;
    }
    Ok(())
}

fn shell_quote(argument: &str) -> String {
    if !argument.is_empty()
        && argument
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_./:=-".contains(&byte))
    {
        argument.to_owned()
    } else {
        format!("'{}'", argument.replace('\'', "'\\''"))
    }
}

pub fn truncate_utf8(text: &str, limit: usize) -> (&str, bool) {
    if text.len() <= limit {
        return (text, false);
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (&text[..end], true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{Device, profile};

    #[test]
    fn indexing_messages_use_selected_runtime_and_report_transitions_once() {
        let model = profile("minilm-int8").unwrap();
        let mut info = RuntimeInfo {
            requested_device: Device::Auto,
            selected_device: Device::Metal,
            provider: "Gguf".into(),
            runtime_version: "test".into(),
            device_name: "Example GPU".into(),
            fallback_reasons: vec![],
            calibration: vec![],
            qualified: false,
            batch_size: 8,
        };
        let mut out = Vec::new();
        write_indexing_runtime(&mut out, &model, &info, None).unwrap();
        let initial = String::from_utf8(out.clone()).unwrap();
        assert!(initial.contains("metal (GGUF / llama.cpp; Example GPU)"));
        let previous = info.clone();
        info.selected_device = Device::Cpu;
        info.device_name = "cpu".into();
        info.fallback_reasons
            .push("Accelerator allocation failed; CPU selected".into());
        write_indexing_runtime(&mut out, &model, &info, Some(&previous)).unwrap();
        let reported_len = out.len();
        write_indexing_runtime(&mut out, &model, &info, Some(&info)).unwrap();
        assert_eq!(out.len(), reported_len, "unchanged runtime must stay quiet");
        let rendered = String::from_utf8(out).unwrap();
        assert!(rendered.contains("Indexing device changed: cpu (GGUF / llama.cpp)"));
        assert_eq!(rendered.matches("Accelerator allocation failed").count(), 1);
    }
}
