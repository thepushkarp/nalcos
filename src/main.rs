use clap::Parser;
use nalcos::{app, cli::Cli, error::AppError, execution::Execution, output};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    let json = args.iter().any(|arg| arg == "--json");
    let cli = match Cli::try_parse_from(&args) {
        Ok(cli) => cli,
        Err(error) => {
            if error.use_stderr() && json {
                fail(AppError::invalid(error.to_string()), true);
            }
            error.exit();
        }
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let token = Arc::clone(&cancelled);
    if let Err(error) = ctrlc::set_handler(move || token.store(true, Ordering::Relaxed)) {
        fail(
            AppError::new("signal_handler_failed", error.to_string()),
            cli.json,
        );
    }
    let execution = Execution::new(cli.timeout, cancelled);
    match app::run(&cli, &execution) {
        Ok(value) => {
            if let Err(error) = output::emit(&value, cli.json) {
                fail(error, cli.json);
            }
        }
        Err(error) => fail(error, cli.json),
    }
}

fn fail(error: AppError, json: bool) -> ! {
    let code = error.exit_code();
    if json {
        let value = serde_json::json!({"schema_version":1,"command":"error","error":error});
        if let Err(write_error) = output::emit(&value, true) {
            eprintln!("{write_error}");
        }
    } else {
        eprintln!("{error}");
    }
    std::process::exit(code);
}
