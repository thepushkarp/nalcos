use crate::{
    embedding::{Backend, ModelProfile, RuntimeInfo},
    error::Result,
};
use serde_json::Value;
use std::io::{self, Write};

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
