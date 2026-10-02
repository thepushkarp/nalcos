use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub(crate) struct HfTokenizer {
    pub inner: tokenizers::Tokenizer,
    pub pad_id: u32,
    pub pad_type_id: u32,
}

impl HfTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let mut inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            AppError::new(
                "tokenizer_invalid",
                format!("Cannot load {}: {e}", path.display()),
            )
        })?;
        let pad_id = inner
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| inner.token_to_id("[PAD]"))
            .or_else(|| inner.token_to_id("<pad>"))
            .unwrap_or(0);
        let pad_type_id = inner.get_padding().map_or(0, |p| p.pad_type_id);
        inner.with_padding(None);
        inner
            .with_truncation(None)
            .map_err(|e| AppError::new("tokenizer_invalid", e.to_string()))?;
        Ok(Self {
            inner,
            pad_id,
            pad_type_id,
        })
    }

    pub fn encode(&self, text: &str) -> Result<tokenizers::Encoding> {
        self.inner
            .encode(text, true)
            .map_err(|e| AppError::new("tokenization_failed", e.to_string()))
    }

    pub fn count(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChunkOptions {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            max_tokens: 384,
            overlap_tokens: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
    /// Includes the document prefix and model special tokens, matching encode's limit.
    pub token_count: usize,
}

/// Tokenization determines bounds; character boundaries are used only to retain original text.
/// Whitespace and Unicode bytes are never reconstructed through a lossy tokenizer decode.
pub(crate) fn chunk_text(
    text: &str,
    options: ChunkOptions,
    model_limit: usize,
    count: impl Fn(&str) -> Result<usize>,
) -> Result<Vec<DocumentChunk>> {
    let limit = options.max_tokens.min(model_limit);
    let overhead = count("")?;
    if limit <= overhead || options.overlap_tokens >= limit.saturating_sub(overhead) {
        return Err(AppError::invalid(
            "Chunk max_tokens must exceed prefix/special-token overhead, and overlap_tokens must leave room for new content",
        ));
    }
    if text.is_empty() {
        return Ok(vec![]);
    }
    let boundaries: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut start = 0;
    let last = boundaries.len() - 1;
    let mut chunks = Vec::new();
    while start < last {
        // Bound each tokenization call even for very large histories or whitespace runs.
        let mut high = (start + limit.saturating_mul(8).max(1)).min(last);
        let mut low = start + 1;
        let mut end = start;
        let mut token_count = 0;
        while low <= high {
            let mid = low + (high - low) / 2;
            let candidate = &text[boundaries[start]..boundaries[mid]];
            let tokens = count(candidate)?;
            if tokens <= limit {
                end = mid;
                token_count = tokens;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        if end == start {
            return Err(AppError::new(
                "token_limit",
                "A single Unicode character plus the configured prefix exceeds the chunk token budget",
            ));
        }
        let byte_start = boundaries[start];
        let byte_end = boundaries[end];
        chunks.push(DocumentChunk {
            text: text[byte_start..byte_end].into(),
            byte_start,
            byte_end,
            token_count,
        });
        if end == last {
            break;
        }
        if options.overlap_tokens == 0 {
            start = end;
            continue;
        }
        let overlap_limit = overhead + options.overlap_tokens;
        let mut low = start + 1;
        let mut high = end;
        let mut next = end;
        while low <= high {
            let mid = low + (high - low) / 2;
            if count(&text[boundaries[mid]..boundaries[end]])? <= overlap_limit {
                next = mid;
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        // Even overlapping all the available content must make forward progress.
        start = next.max(start + 1);
    }
    Ok(chunks)
}
