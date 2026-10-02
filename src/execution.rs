use crate::error::{AppError, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

/// A single deadline and cancellation token are shared across subprocesses and inference.
#[derive(Clone)]
pub struct Execution {
    deadline: Option<Instant>,
    cancelled: Arc<AtomicBool>,
}

impl Execution {
    pub fn new(timeout: Option<Duration>, cancelled: Arc<AtomicBool>) -> Self {
        Self {
            deadline: timeout.map(|t| Instant::now() + t),
            cancelled,
        }
    }

    #[cfg(test)]
    pub fn unlimited() -> Self {
        Self::new(None, Arc::new(AtomicBool::new(false)))
    }

    pub fn bounded(&self, budget: Duration) -> Self {
        let limit = Instant::now() + budget;
        Self {
            deadline: Some(self.deadline.map_or(limit, |d| d.min(limit))),
            cancelled: self.cancelled.clone(),
        }
    }

    pub fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(AppError::new(
                "interrupted",
                "Operation interrupted; completed index batches are retained",
            ));
        }
        if self.deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(AppError::new(
                "timeout",
                "Operation exceeded its time budget; completed index batches are retained",
            ));
        }
        Ok(())
    }

    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
}
