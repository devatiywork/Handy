//! Settings commands for OpenRouter cloud transcription. The API key goes
//! through `change_post_process_api_key_setting("openrouter", …)`, since the
//! key is shared with post-processing.

use crate::cloud_transcription::{self, CloudKeyCheck, CloudModelOption};
use crate::settings::{get_settings, write_settings, SecretString};
use tauri::AppHandle;

#[tauri::command]
#[specta::specta]
pub fn change_cloud_transcription_model_setting(
    app: AppHandle,
    model: String,
) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.cloud_transcription_model = match model.trim() {
        "" => cloud_transcription::DEFAULT_CLOUD_MODEL.to_string(),
        model => model.to_string(),
    };
    write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn fetch_cloud_transcription_models(
    app: AppHandle,
) -> Result<Vec<CloudModelOption>, String> {
    let settings = get_settings(&app);
    cloud_transcription::fetch_models(&settings).await
}

/// Store (or, with an empty string, clear) the OpenRouter management key that
/// reads the account balance.
#[tauri::command]
#[specta::specta]
pub fn change_openrouter_management_key_setting(app: AppHandle, key: String) -> Result<(), String> {
    let mut settings = get_settings(&app);
    settings.openrouter_management_key = SecretString::new(key.trim().to_string());
    write_settings(&app, settings);
    Ok(())
}

/// Check the saved OpenRouter key and read its spending, its limit and the
/// account balance.
#[tauri::command]
#[specta::specta]
pub async fn check_cloud_transcription_key(app: AppHandle) -> Result<CloudKeyCheck, String> {
    let settings = get_settings(&app);
    cloud_transcription::check_key(&settings).await
}
