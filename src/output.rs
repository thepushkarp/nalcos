use crate::error::Result;
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
                            writeln!(out, "  evidence: {}", item["id"].as_str().unwrap_or(""))?;
                            for line in item["excerpt"].as_str().unwrap_or("").lines() {
                                writeln!(out, "    {line}")?;
                            }
                        }
                    }
                }
            }
        }
        "show" => {
            if let Some(evidence) = value.get("evidence") {
                writeln!(out, "{}", evidence["excerpt"].as_str().unwrap_or(""))?;
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
