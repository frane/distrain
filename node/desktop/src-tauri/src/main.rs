//! Distrain Desktop — Tauri application entry point.
//!
//! Wraps the Burn-based training node into a desktop GUI with
//! tray icon, start/stop controls, and real-time stats display.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

fn main() {
    tracing_subscriber::fmt::init();

    let app_state = std::sync::Arc::new(commands::AppState::default());

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::get_node_status,
            commands::start_training,
            commands::stop_training,
            commands::calibrate_device,
            commands::get_training_stats,
            commands::get_logs,
            commands::get_config,
            commands::save_config,
            commands::get_training_params,
            commands::save_training_params,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
