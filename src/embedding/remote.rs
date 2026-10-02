use super::hub::{checked_response, network_error, request_timeout};
use super::{Backend, ModelProfile};
use crate::error::{AppError, Result};
use crate::execution::Execution;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::time::Duration;

pub(crate) struct RemoteEncoder {
    client: Client,
    profile: ModelProfile,
    url: String,
}

impl RemoteEncoder {
    pub fn load(profile: &ModelProfile) -> Result<Self> {
        let endpoint = profile
            .endpoint
            .as_deref()
            .ok_or_else(|| AppError::invalid("Provider endpoint is required"))?;
        let suffix = match profile.backend {
            Backend::OpenAi => "/embeddings",
            Backend::Ollama => "/api/embed",
            _ => return Err(AppError::invalid("Not a remote provider")),
        };
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(network_error)?;
        Ok(Self {
            client,
            profile: profile.clone(),
            url: format!("{}{suffix}", endpoint.trim_end_matches('/')),
        })
    }

    pub fn encode(&self, texts: &[String], execution: &Execution) -> Result<Vec<Vec<f32>>> {
        execution.check()?;
        let mut request = self
            .client
            .post(&self.url)
            .timeout(request_timeout(execution));
        if let Some(variable) = &self.profile.api_key_env {
            let token = std::env::var(variable).map_err(|_| {
                AppError::new(
                    "provider_auth_missing",
                    format!("Set {variable} to the embedding provider API key"),
                )
            })?;
            request = request.bearer_auth(token);
        }
        let payload = match self.profile.backend {
            Backend::OpenAi => {
                serde_json::json!({"model": self.profile.id, "input": texts, "encoding_format": "float"})
            }
            Backend::Ollama => {
                serde_json::json!({"model": self.profile.id, "input": texts, "truncate": false})
            }
            _ => return Err(AppError::invalid("Not a remote provider")),
        };
        let response = request.json(&payload).send().map_err(network_error)?;
        let response = checked_response(response, "embed text")?;
        let output = match self.profile.backend {
            Backend::OpenAi => {
                let response: OpenAiResponse = response.json().map_err(network_error)?;
                reorder_openai(response, texts.len())?
            }
            Backend::Ollama => {
                let response: OllamaResponse = response.json().map_err(network_error)?;
                response.embeddings
            }
            _ => unreachable!("provider backend validated on load"),
        };
        execution.check()?;
        if output.len() != texts.len() {
            return Err(AppError::new(
                "provider_invalid_response",
                format!(
                    "Provider returned {} vectors for {} texts",
                    output.len(),
                    texts.len()
                ),
            ));
        }
        Ok(output)
    }
}

#[derive(Deserialize)]
struct OpenAiResponse {
    data: Vec<OpenAiEmbedding>,
}
#[derive(Deserialize)]
struct OpenAiEmbedding {
    index: usize,
    embedding: Vec<f32>,
}
#[derive(Deserialize)]
struct OllamaResponse {
    embeddings: Vec<Vec<f32>>,
}

fn reorder_openai(response: OpenAiResponse, count: usize) -> Result<Vec<Vec<f32>>> {
    let mut ordered = vec![None; count];
    for embedding in response.data {
        let slot = ordered.get_mut(embedding.index).ok_or_else(|| {
            AppError::new(
                "provider_invalid_response",
                "Provider returned an embedding with an out-of-range index",
            )
        })?;
        if slot.is_some() {
            return Err(AppError::new(
                "provider_invalid_response",
                "Provider returned duplicate embedding indices",
            ));
        }
        *slot = Some(embedding.embedding);
    }
    ordered
        .into_iter()
        .map(|v| {
            v.ok_or_else(|| {
                AppError::new(
                    "provider_invalid_response",
                    "Provider omitted an embedding index",
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn provider_indices_define_correspondence_and_must_be_complete() {
        let decoded = serde_json::from_value(serde_json::json!({"data":[{"index":1,"embedding":[2.0]},{"index":0,"embedding":[1.0]}]})).unwrap();
        assert_eq!(
            reorder_openai(decoded, 2).unwrap(),
            vec![vec![1.0], vec![2.0]]
        );
        let duplicate = OpenAiResponse {
            data: vec![
                OpenAiEmbedding {
                    index: 0,
                    embedding: vec![1.0],
                },
                OpenAiEmbedding {
                    index: 0,
                    embedding: vec![2.0],
                },
            ],
        };
        assert!(reorder_openai(duplicate, 2).is_err());
        assert!(reorder_openai(OpenAiResponse { data: vec![] }, 1).is_err());
    }
}
