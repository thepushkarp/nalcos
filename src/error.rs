use serde::Serialize;
use std::fmt;

pub type Result<T> = std::result::Result<T, AppError>;

#[derive(Debug, Clone, Serialize)]
pub struct AppError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
}

impl AppError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            action: None,
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_input", message)
    }

    pub fn action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }

    pub fn exit_code(&self) -> i32 {
        match self.code.as_str() {
            "invalid_input" | "invalid_config" | "unsupported_schema" => 2,
            "timeout" => 124,
            "interrupted" => 130,
            _ => 1,
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)?;
        if let Some(action) = &self.action {
            write!(f, " ({action})")?;
        }
        Ok(())
    }
}

impl std::error::Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(value: std::io::Error) -> Self {
        Self::new("io_error", value.to_string())
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(value: rusqlite::Error) -> Self {
        let code = match &value {
            rusqlite::Error::SqliteFailure(error, _)
                if matches!(
                    error.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                ) =>
            {
                "index_busy"
            }
            _ => "index_error",
        };
        Self::new(code, value.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(value: serde_json::Error) -> Self {
        Self::new("invalid_data", value.to_string())
    }
}
