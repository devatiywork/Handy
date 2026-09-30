//! Cloud speech-to-text through OpenRouter's `/audio/transcriptions` endpoint.
//!
//! Every other engine runs on-device; this is the one path that sends recorded
//! audio off the machine, and it is only reached when the user selects the cloud
//! entry ([`CLOUD_MODEL_ID`]) in the model list. The API key is the OpenRouter
//! key post-processing already uses (`post_process_api_keys["openrouter"]`), so
//! it only has to be entered once.

use std::collections::HashSet;
use std::io::Cursor;
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use base64::Engine as _;
use log::{debug, info, warn};
use reqwest::header::{HeaderMap, HeaderValue, REFERER, USER_AGENT};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use specta::Type;

use crate::llm_client::{report_reqwest_error, sanitized_url_for_log};
use crate::managers::model::{EngineType, ModelInfo, ModelSource};
use crate::settings::AppSettings;

/// Registry id of the cloud entry in the model list.
pub const CLOUD_MODEL_ID: &str = "openrouter-cloud";

/// Post-processing provider whose API key and base URL cloud transcription shares.
pub const OPENROUTER_PROVIDER_ID: &str = "openrouter";

/// Remote model used until the user picks another one.
pub const DEFAULT_CLOUD_MODEL: &str = "openai/gpt-transcribe";

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const SAMPLE_RATE: u32 = 16_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// Fixed part of a request's time budget. The recording's own duration is added
/// on top, so long dictations are not cut off while short ones still fail fast.
const REQUEST_TIMEOUT_BASE: Duration = Duration::from_secs(60);
const REQUEST_TIMEOUT_MAX: Duration = Duration::from_secs(600);
/// Pause before the single retry of a connection failure or gateway error.
const RETRY_DELAY: Duration = Duration::from_millis(750);
const MAX_KEYWORDS: usize = 100;
/// How much of a provider error message is surfaced to the user.
const MAX_ERROR_MESSAGE_CHARS: usize = 300;

/// The model-list entry for cloud transcription. It has no files on disk, so it
/// is always "downloaded" (selectable); whether it can run depends on the API
/// key, which [`CloudConfig::from_settings`] checks before every use.
pub fn model_info(supported_languages: Vec<String>) -> ModelInfo {
    ModelInfo {
        id: CLOUD_MODEL_ID.to_string(),
        name: "OpenRouter Cloud".to_string(),
        description: "Transcribes in the cloud via OpenRouter (openai/gpt-transcribe by default). \
                      Audio leaves your device and requires an API key."
            .to_string(),
        filename: String::new(),
        source: ModelSource::Cloud,
        size_mb: 0,
        is_downloaded: true,
        is_downloading: false,
        partial_size: 0,
        is_directory: false,
        engine_type: EngineType::OpenRouter,
        // 0/0 is the UI's "no benchmark" sentinel: quality depends on which
        // remote model is configured, so no score bars are shown.
        accuracy_score: 0.0,
        speed_score: 0.0,
        supports_translation: false,
        is_recommended: false,
        supports_language_selection: supported_languages.len() > 1,
        supported_languages,
        is_custom: false,
        supports_streaming: false,
        supports_language_detection: true,
    }
}

pub fn is_cloud_model(model: &ModelInfo) -> bool {
    matches!(model.source, ModelSource::Cloud)
}

/// Where and as whom a cloud request is sent. Resolved from settings at call
/// time, so a changed key or model applies to the next dictation without a
/// model reload.
#[derive(Clone)]
pub struct CloudConfig {
    base_url: String,
    api_key: String,
    model: String,
}

impl CloudConfig {
    pub fn from_settings(settings: &AppSettings) -> Result<Self, String> {
        let api_key = settings
            .post_process_api_keys
            .get(OPENROUTER_PROVIDER_ID)
            .map(|key| key.trim().to_string())
            .unwrap_or_default();
        if api_key.is_empty() {
            return Err(
                "OpenRouter API key is not set. Add it under Models → Cloud Transcription."
                    .to_string(),
            );
        }
        Ok(Self {
            base_url: base_url(settings),
            api_key,
            model: model_id(settings),
        })
    }
}

/// The remote model to request, falling back to the default when unset.
pub fn model_id(settings: &AppSettings) -> String {
    match settings.cloud_transcription_model.trim() {
        "" => DEFAULT_CLOUD_MODEL.to_string(),
        model => model.to_string(),
    }
}

fn base_url(settings: &AppSettings) -> String {
    settings
        .post_process_provider(OPENROUTER_PROVIDER_ID)
        .map(|provider| provider.base_url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
}

/// The `language` hint for a request, or `None` to let the service detect it.
/// OpenRouter expects ISO-639-1, so codes without a two-letter form (Cantonese
/// `yue`, Hawaiian `haw`, …) are left to auto-detection rather than risking a
/// rejected request. Chinese script intents collapse to plain `zh`; the
/// Simplified/Traditional conversion runs on the text afterwards.
pub fn language_hint(effective_language: &str) -> Option<String> {
    let base = match effective_language {
        "" | "auto" => return None,
        // Whisper's legacy code for Javanese.
        "jw" => "jv",
        other => other.split(['-', '_']).next().unwrap_or(other),
    };
    (base.len() == 2 && base.bytes().all(|b| b.is_ascii_lowercase())).then(|| base.to_string())
}

/// Provider options passing the user's custom words to models known to take
/// recognition hints: GPT Transcribe accepts literal `keywords`. Other models get
/// nothing extra — Handy's fuzzy custom-word correction runs on every result.
fn keyword_hints(model: &str, custom_words: &[String]) -> Option<Value> {
    if !model.starts_with("openai/gpt-transcribe") {
        return None;
    }
    // OpenAI rejects keywords containing angle brackets or line breaks.
    let keywords: Vec<&str> = custom_words
        .iter()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty() && !word.contains(['<', '>', '\r', '\n']))
        .take(MAX_KEYWORDS)
        .collect();
    if keywords.is_empty() {
        return None;
    }
    Some(json!({ "options": { "openai": { "keywords": keywords } } }))
}

/// Endpoints (base_url|model) that rejected keyword hints with a 4xx. Remembered
/// for the process lifetime so later dictations skip the failing first attempt.
fn keyword_rejections() -> &'static Mutex<HashSet<String>> {
    static REJECTED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REJECTED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn rejection_key(config: &CloudConfig) -> String {
    format!("{}|{}", config.base_url, config.model)
}

fn is_known_rejected(key: &str) -> bool {
    keyword_rejections()
        .lock()
        .map(|set| set.contains(key))
        .unwrap_or(false)
}

fn remember_rejection(key: String) {
    if let Ok(mut set) = keyword_rejections().lock() {
        set.insert(key);
    }
}

/// Encode 16 kHz mono samples as 16-bit PCM WAV — the format Handy saves its
/// recordings in, and one every OpenRouter speech-to-text provider accepts.
fn encode_wav(samples: &[f32]) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let encode_error = |e: hound::Error| format!("Failed to encode audio: {e}");
    let mut buffer = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    let mut writer = hound::WavWriter::new(&mut buffer, spec).map_err(encode_error)?;
    for &sample in samples {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .map_err(encode_error)?;
    }
    writer.finalize().map_err(encode_error)?;
    Ok(buffer.into_inner())
}

fn request_timeout(sample_count: usize) -> Duration {
    let audio = Duration::from_secs_f64(sample_count as f64 / SAMPLE_RATE as f64);
    (REQUEST_TIMEOUT_BASE + audio).min(REQUEST_TIMEOUT_MAX)
}

/// Runtime that drives every cloud request. `TranscriptionManager::transcribe`
/// is synchronous and runs on Tauri runtime workers (the shortcut path), the
/// blocking pool (history retry) and the main thread (headless CLI). A runtime
/// of its own lets all of them simply wait on a channel: blocking on Tauri's
/// runtime from one of its own workers panics, and on a single-core machine
/// the blocked worker could be the only one left to drive the request.
fn runtime() -> Result<&'static tokio::runtime::Runtime, String> {
    static RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("cloud-transcription")
                .enable_all()
                .build()
                .map_err(|e| format!("Failed to start the cloud transcription runtime: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Shared client, so consecutive dictations reuse a warm TLS connection.
fn http_client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            // OpenRouter app attribution, matching the post-processing client.
            let mut headers = HeaderMap::new();
            headers.insert(
                REFERER,
                HeaderValue::from_static("https://github.com/cjpais/Handy"),
            );
            headers.insert(
                USER_AGENT,
                HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
            );
            headers.insert("X-Title", HeaderValue::from_static("Handy"));
            reqwest::Client::builder()
                .default_headers(headers)
                .connect_timeout(CONNECT_TIMEOUT)
                .build()
                .map_err(|e| report_reqwest_error("Failed to build HTTP client", &e))
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Serialize)]
struct InputAudio<'a> {
    data: &'a str,
    format: &'static str,
}

#[derive(Serialize)]
struct TranscriptionRequest<'a> {
    model: &'a str,
    input_audio: InputAudio<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<&'a Value>,
}

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
    #[serde(default)]
    usage: Option<TranscriptionUsage>,
}

#[derive(Deserialize)]
struct TranscriptionUsage {
    #[serde(default)]
    seconds: Option<f64>,
    #[serde(default)]
    cost: Option<f64>,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Deserialize)]
struct ErrorDetail {
    #[serde(default)]
    message: String,
    #[serde(default)]
    metadata: Option<ErrorMetadata>,
}

#[derive(Deserialize)]
struct ErrorMetadata {
    /// The upstream provider's own error, as a JSON string or object.
    #[serde(default)]
    raw: Option<Value>,
}

/// Transcribe 16 kHz mono `samples` with the configured OpenRouter model,
/// blocking the calling thread until the request completes.
pub fn transcribe_blocking(
    config: &CloudConfig,
    samples: &[f32],
    language: Option<String>,
    custom_words: &[String],
) -> Result<String, String> {
    let request = PreparedRequest {
        audio_base64: base64::engine::general_purpose::STANDARD.encode(encode_wav(samples)?),
        keyword_hints: keyword_hints(&config.model, custom_words),
        config: config.clone(),
        language,
        timeout: request_timeout(samples.len()),
        audio_secs: samples.len() as f64 / SAMPLE_RATE as f64,
    };

    let (tx, rx) = mpsc::channel();
    runtime()?.spawn(async move {
        let _ = tx.send(request.send().await);
    });
    // A panic inside the task drops the sender, which ends the wait here.
    rx.recv()
        .map_err(|_| "Cloud transcription stopped unexpectedly".to_string())?
}

struct PreparedRequest {
    config: CloudConfig,
    audio_base64: String,
    language: Option<String>,
    keyword_hints: Option<Value>,
    timeout: Duration,
    audio_secs: f64,
}

impl PreparedRequest {
    async fn send(self) -> Result<String, String> {
        let client = http_client()?;
        let url = format!("{}/audio/transcriptions", self.config.base_url);
        debug!(
            "Sending cloud transcription request to {} (model '{}', {:.1}s of audio)",
            sanitized_url_for_log(&url),
            self.config.model,
            self.audio_secs
        );

        let rejection_key = rejection_key(&self.config);
        let mut hints = self
            .keyword_hints
            .as_ref()
            .filter(|_| !is_known_rejected(&rejection_key));

        let mut response = self.post_with_retry(client, &url, hints).await?;

        // A 400/422 answer to a request carrying keyword hints most likely means
        // the upstream does not accept them; retry once without.
        if hints.is_some() && matches!(response.status().as_u16(), 400 | 422) {
            let status = response.status();
            let detail = error_message(response).await.unwrap_or_default();
            info!(
                "OpenRouter rejected keyword hints for '{}' ({}: {}); retrying without them",
                self.config.model, status, detail
            );
            hints = None;
            response = self.post_with_retry(client, &url, hints).await?;
            if response.status().is_success() {
                remember_rejection(rejection_key);
            }
        }

        let status = response.status();
        if !status.is_success() {
            return Err(describe_failure(status, error_message(response).await));
        }

        let parsed: TranscriptionResponse = response.json().await.map_err(|e| {
            report_reqwest_error("Failed to parse OpenRouter transcription response", &e)
        })?;
        if let Some(usage) = &parsed.usage {
            info!(
                "OpenRouter transcription with '{}': {:.1}s of audio, billed {}s, cost {}",
                self.config.model,
                self.audio_secs,
                usage
                    .seconds
                    .map_or_else(|| "?".to_string(), |s| format!("{s:.1}")),
                usage
                    .cost
                    .map_or_else(|| "?".to_string(), |c| format!("${c:.6}")),
            );
        }
        Ok(parsed.text)
    }

    /// POST once, retrying a single time after a connection failure or a
    /// transient gateway error (502/503/504).
    async fn post_with_retry(
        &self,
        client: &reqwest::Client,
        url: &str,
        hints: Option<&Value>,
    ) -> Result<reqwest::Response, String> {
        match self.post(client, url, hints).await {
            Ok(response) if !is_transient(response.status()) => return Ok(response),
            Ok(response) => warn!("OpenRouter returned {}; retrying once", response.status()),
            Err(error) if error.is_connect() => {
                warn!("Could not connect to OpenRouter; retrying once");
            }
            Err(error) => {
                return Err(describe_transport_error(
                    TRANSCRIBE_CONTEXT,
                    &error,
                    self.timeout,
                ))
            }
        }
        tokio::time::sleep(RETRY_DELAY).await;
        self.post(client, url, hints)
            .await
            .map_err(|error| describe_transport_error(TRANSCRIBE_CONTEXT, &error, self.timeout))
    }

    async fn post(
        &self,
        client: &reqwest::Client,
        url: &str,
        hints: Option<&Value>,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let body = TranscriptionRequest {
            model: &self.config.model,
            input_audio: InputAudio {
                data: &self.audio_base64,
                format: "wav",
            },
            language: self.language.as_deref(),
            provider: hints,
        };
        client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
    }
}

fn is_transient(status: StatusCode) -> bool {
    matches!(status.as_u16(), 502..=504)
}

const TRANSCRIBE_CONTEXT: &str = "OpenRouter transcription request failed";

fn describe_transport_error(context: &str, error: &reqwest::Error, timeout: Duration) -> String {
    // Logs the full sanitized cause chain; the UI gets the short version.
    let details = report_reqwest_error(context, error);
    if error.is_timeout() {
        format!("OpenRouter did not respond within {}s", timeout.as_secs())
    } else if error.is_connect() {
        "Could not reach OpenRouter. Check your internet connection.".to_string()
    } else {
        details
    }
}

fn describe_failure(status: StatusCode, detail: Option<String>) -> String {
    let summary = match status.as_u16() {
        400 | 422 => "the request was rejected",
        401 => "the API key was rejected",
        402 => "the account has insufficient credits",
        403 => "access was denied",
        404 => "the model was not found",
        408 | 504 => "the request timed out",
        413 => "the recording is too large",
        429 => "rate limit reached, try again shortly",
        500..=599 => "the transcription provider failed",
        _ => "the request failed",
    };
    match detail {
        Some(detail) => format!(
            "OpenRouter error {}: {} ({})",
            status.as_u16(),
            summary,
            detail
        ),
        None => format!("OpenRouter error {}: {}", status.as_u16(), summary),
    }
}

/// The readable part of an error response: OpenRouter's `error.message`, plus
/// the upstream provider's own message when OpenRouter wraps one, else the raw
/// body. Truncated so a verbose upstream error cannot flood the error toast.
async fn error_message(response: reqwest::Response) -> Option<String> {
    let body = response.text().await.ok()?;
    let message = match serde_json::from_str::<ErrorBody>(&body) {
        Ok(parsed) => {
            let upstream = parsed
                .error
                .metadata
                .and_then(|metadata| metadata.raw)
                .and_then(|raw| upstream_message(&raw));
            match upstream {
                Some(upstream) if parsed.error.message.is_empty() => upstream,
                Some(upstream) => format!("{}: {}", parsed.error.message, upstream),
                None => parsed.error.message,
            }
        }
        Err(_) => body,
    };
    let message = message.trim();
    (!message.is_empty()).then(|| truncate_chars(message, MAX_ERROR_MESSAGE_CHARS))
}

fn upstream_message(raw: &Value) -> Option<String> {
    let parsed;
    let raw = match raw {
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(value) => {
                parsed = value;
                &parsed
            }
            Err(_) => return Some(text.clone()),
        },
        other => other,
    };
    raw.pointer("/error/message")
        .or_else(|| raw.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((index, _)) => format!("{}…", &text[..index]),
        None => text.to_string(),
    }
}

/// One speech-to-text model offered through OpenRouter.
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct CloudModelOption {
    pub id: String,
    pub name: String,
    /// OpenRouter's (English) description of the model; may be empty.
    pub description: String,
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelListEntry>,
}

#[derive(Deserialize)]
struct ModelListEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

/// List OpenRouter's speech-to-text models. The catalog is public, so this works
/// before an API key is configured.
pub async fn fetch_models(settings: &AppSettings) -> Result<Vec<CloudModelOption>, String> {
    let base_url = base_url(settings);
    runtime()?
        .spawn(fetch_models_from(base_url))
        .await
        .map_err(|e| format!("Model list request stopped unexpectedly: {e}"))?
}

async fn fetch_models_from(base_url: String) -> Result<Vec<CloudModelOption>, String> {
    let url = format!("{base_url}/models?output_modalities=transcription");
    debug!(
        "Fetching cloud transcription models from {}",
        sanitized_url_for_log(&url)
    );
    let response = http_client()?
        .get(&url)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| report_reqwest_error("Failed to fetch OpenRouter models", &e))?;
    let status = response.status();
    if !status.is_success() {
        return Err(describe_failure(status, error_message(response).await));
    }
    let list: ModelList = response
        .json()
        .await
        .map_err(|e| report_reqwest_error("Failed to parse OpenRouter model list", &e))?;
    let mut models: Vec<CloudModelOption> = list
        .data
        .into_iter()
        .map(|entry| CloudModelOption {
            name: entry
                .name
                .filter(|name| !name.trim().is_empty())
                .unwrap_or_else(|| entry.id.clone()),
            description: entry
                .description
                .map(|description| description.trim().to_string())
                .unwrap_or_default(),
            id: entry.id,
        })
        .collect();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

/// What OpenRouter reports about the configured key. Amounts are USD credits
/// spent through this key; `limit` and `limit_remaining` are `None` when the
/// key has no spending limit. The account's overall balance is only available
/// to management keys, so it is not part of this.
#[derive(Debug, Clone, Serialize, Type)]
pub struct CloudKeyInfo {
    pub label: String,
    pub usage: f64,
    pub usage_daily: f64,
    pub usage_weekly: f64,
    pub usage_monthly: f64,
    pub limit: Option<f64>,
    pub limit_remaining: Option<f64>,
    /// How often the limit resets: "daily", "weekly" or "monthly".
    pub limit_reset: Option<String>,
    pub is_free_tier: bool,
}

/// Outcome of checking the configured key. A rejection (401/403) is a verdict
/// on the key, so it is reported here; failing to reach OpenRouter is an error.
#[derive(Debug, Clone, Serialize, Type)]
pub struct CloudKeyCheck {
    /// Present when OpenRouter accepted the key.
    pub info: Option<CloudKeyInfo>,
    /// OpenRouter's explanation when it rejected the key.
    pub rejected: Option<String>,
    /// The whole account's credit balance, as far as the keys can read it.
    pub account: CloudAccountStatus,
}

/// The account's credit balance (`GET /credits`). OpenRouter documents that
/// endpoint as management-key only, so an inference key is tried first and
/// the optional management key after it.
#[derive(Debug, Clone, PartialEq, Serialize, Type)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CloudAccountStatus {
    /// USD credits bought and spent across the whole account.
    Available {
        total_credits: f64,
        total_usage: f64,
    },
    /// Neither configured key may read the balance.
    NeedsManagementKey,
    /// OpenRouter refused the configured management key.
    ManagementKeyRejected { message: String },
    /// The balance could not be read (network or server error).
    Unavailable { message: String },
}

#[derive(Deserialize)]
struct CreditsResponse {
    data: CreditsData,
}

#[derive(Deserialize)]
struct CreditsData {
    #[serde(default)]
    total_credits: f64,
    #[serde(default)]
    total_usage: f64,
}

enum CreditsAnswer {
    Balance(CreditsData),
    /// 401/403, with OpenRouter's explanation and whether it was a 403
    /// (a valid key without access, as opposed to an invalid one).
    Refused {
        message: String,
        forbidden: bool,
    },
}

async fn fetch_credits(base_url: &str, key: &str) -> Result<CreditsAnswer, String> {
    const TIMEOUT: Duration = Duration::from_secs(20);
    let response = http_client()?
        .get(format!("{base_url}/credits"))
        .bearer_auth(key)
        .timeout(TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_transport_error("OpenRouter balance request failed", &e, TIMEOUT))?;

    let status = response.status();
    if matches!(status.as_u16(), 401 | 403) {
        return Ok(CreditsAnswer::Refused {
            forbidden: status == StatusCode::FORBIDDEN,
            message: describe_failure(status, error_message(response).await),
        });
    }
    if !status.is_success() {
        return Err(describe_failure(status, error_message(response).await));
    }
    let credits: CreditsResponse = response
        .json()
        .await
        .map_err(|e| report_reqwest_error("Failed to parse OpenRouter balance", &e))?;
    Ok(CreditsAnswer::Balance(credits.data))
}

/// Base URLs whose `/credits` refused an inference key with a 403. Remembered
/// so later checks go straight to the management key.
fn inference_key_credit_refusals() -> &'static Mutex<HashSet<String>> {
    static REFUSED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REFUSED.get_or_init(|| Mutex::new(HashSet::new()))
}

async fn account_status(
    base_url: &str,
    api_key: &str,
    management_key: Option<&str>,
) -> CloudAccountStatus {
    let available = |credits: CreditsData| CloudAccountStatus::Available {
        total_credits: credits.total_credits,
        total_usage: credits.total_usage,
    };

    let refused_before = inference_key_credit_refusals()
        .lock()
        .map(|set| set.contains(base_url))
        .unwrap_or(false);
    if !refused_before {
        match fetch_credits(base_url, api_key).await {
            Ok(CreditsAnswer::Balance(credits)) => return available(credits),
            Ok(CreditsAnswer::Refused { forbidden, .. }) => {
                if forbidden {
                    if let Ok(mut set) = inference_key_credit_refusals().lock() {
                        set.insert(base_url.to_string());
                    }
                }
            }
            Err(message) => return CloudAccountStatus::Unavailable { message },
        }
    }

    let Some(management_key) = management_key else {
        return CloudAccountStatus::NeedsManagementKey;
    };
    match fetch_credits(base_url, management_key).await {
        Ok(CreditsAnswer::Balance(credits)) => available(credits),
        Ok(CreditsAnswer::Refused { message, .. }) => {
            CloudAccountStatus::ManagementKeyRejected { message }
        }
        Err(message) => CloudAccountStatus::Unavailable { message },
    }
}

#[derive(Deserialize)]
struct KeyResponse {
    data: KeyData,
}

/// `GET /key` payload. Every field defaults so a trimmed or older response
/// still yields whatever it does carry.
#[derive(Deserialize)]
struct KeyData {
    #[serde(default)]
    label: String,
    #[serde(default)]
    usage: f64,
    #[serde(default)]
    usage_daily: f64,
    #[serde(default)]
    usage_weekly: f64,
    #[serde(default)]
    usage_monthly: f64,
    #[serde(default)]
    limit: Option<f64>,
    #[serde(default)]
    limit_remaining: Option<f64>,
    #[serde(default)]
    limit_reset: Option<String>,
    #[serde(default)]
    is_free_tier: bool,
}

impl From<KeyData> for CloudKeyInfo {
    fn from(data: KeyData) -> Self {
        Self {
            label: data.label,
            usage: data.usage,
            usage_daily: data.usage_daily,
            usage_weekly: data.usage_weekly,
            usage_monthly: data.usage_monthly,
            limit: data.limit,
            limit_remaining: data.limit_remaining,
            limit_reset: data.limit_reset.filter(|reset| !reset.trim().is_empty()),
            is_free_tier: data.is_free_tier,
        }
    }
}

/// Check the configured key with OpenRouter and read its spending (`GET /key`)
/// together with the account's balance (`GET /credits`).
pub async fn check_key(settings: &AppSettings) -> Result<CloudKeyCheck, String> {
    let config = CloudConfig::from_settings(settings)?;
    let management_key = Some(settings.openrouter_management_key.trim())
        .filter(|key| !key.is_empty())
        .map(str::to_string);
    runtime()?
        .spawn(check_key_with(config, management_key))
        .await
        .map_err(|e| format!("Key check stopped unexpectedly: {e}"))?
}

async fn check_key_with(
    config: CloudConfig,
    management_key: Option<String>,
) -> Result<CloudKeyCheck, String> {
    let (key, account) = futures_util::future::join(
        read_key(&config),
        account_status(&config.base_url, &config.api_key, management_key.as_deref()),
    )
    .await;
    let (info, rejected) = key?;
    Ok(CloudKeyCheck {
        info,
        rejected,
        account,
    })
}

/// `GET /key`: the key's details, or OpenRouter's reason for rejecting it.
async fn read_key(config: &CloudConfig) -> Result<(Option<CloudKeyInfo>, Option<String>), String> {
    const TIMEOUT: Duration = Duration::from_secs(20);
    let url = format!("{}/key", config.base_url);
    debug!("Checking OpenRouter key at {}", sanitized_url_for_log(&url));
    let response = http_client()?
        .get(&url)
        .bearer_auth(&config.api_key)
        .timeout(TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_transport_error("OpenRouter key check failed", &e, TIMEOUT))?;

    let status = response.status();
    if matches!(status.as_u16(), 401 | 403) {
        let reason = describe_failure(status, error_message(response).await);
        return Ok((None, Some(reason)));
    }
    if !status.is_success() {
        return Err(describe_failure(status, error_message(response).await));
    }
    let key: KeyResponse = response
        .json()
        .await
        .map_err(|e| report_reqwest_error("Failed to parse OpenRouter key info", &e))?;
    Ok((Some(key.data.into()), None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::thread;

    /// Minimal HTTP server answering each connection with the next scripted
    /// `(status, body)` response. Joining the handle yields the request bodies.
    fn scripted_server(
        responses: Vec<(u16, &'static str)>,
    ) -> (String, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let mut bodies = Vec::new();
            for (status, body) in responses {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream);
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            content_length = value.trim().parse().unwrap();
                        }
                    }
                }
                let mut request_body = vec![0u8; content_length];
                reader.read_exact(&mut request_body).unwrap();
                bodies.push(String::from_utf8(request_body).unwrap());

                let mut stream = reader.into_inner();
                write!(
                    stream,
                    "HTTP/1.1 {status} Scripted\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
            bodies
        });
        (base_url, handle)
    }

    fn test_config(base_url: String) -> CloudConfig {
        CloudConfig {
            base_url,
            api_key: "test-key".to_string(),
            model: DEFAULT_CLOUD_MODEL.to_string(),
        }
    }

    fn words(list: &[&str]) -> Vec<String> {
        list.iter().map(|word| word.to_string()).collect()
    }

    #[test]
    fn sends_openrouter_request_and_returns_text() {
        let (base_url, server) = scripted_server(vec![(
            200,
            r#"{"text":"hello world","usage":{"seconds":1.0,"cost":0.000075}}"#,
        )]);
        let samples = vec![0.25_f32; 16_000];

        let text = transcribe_blocking(
            &test_config(base_url),
            &samples,
            Some("en".to_string()),
            &words(&["Handy"]),
        )
        .unwrap();
        assert_eq!(text, "hello world");

        let bodies = server.join().unwrap();
        let request: Value = serde_json::from_str(&bodies[0]).unwrap();
        assert_eq!(request["model"], DEFAULT_CLOUD_MODEL);
        assert_eq!(request["language"], "en");
        assert_eq!(request["input_audio"]["format"], "wav");
        assert_eq!(
            request["provider"]["options"]["openai"]["keywords"],
            json!(["Handy"])
        );

        let wav = base64::engine::general_purpose::STANDARD
            .decode(request["input_audio"]["data"].as_str().unwrap())
            .unwrap();
        let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.sample_rate, spec.channels, spec.bits_per_sample),
            (16_000, 1, 16)
        );
        assert_eq!(reader.len(), 16_000);
    }

    #[test]
    fn omits_optional_fields_when_unused() {
        let (base_url, server) = scripted_server(vec![(200, r#"{"text":"ok"}"#)]);
        let config = CloudConfig {
            model: "openai/whisper-1".to_string(),
            ..test_config(base_url)
        };

        transcribe_blocking(&config, &[0.0; 1_600], None, &words(&["Handy"])).unwrap();

        let request: Value = serde_json::from_str(&server.join().unwrap()[0]).unwrap();
        assert_eq!(request["model"], "openai/whisper-1");
        assert!(request.get("language").is_none());
        assert!(request.get("provider").is_none());
    }

    #[test]
    fn retries_without_keywords_when_the_upstream_rejects_them() {
        let (base_url, server) = scripted_server(vec![
            (
                400,
                r#"{"error":{"message":"Unsupported parameter: keywords"}}"#,
            ),
            (200, r#"{"text":"ok"}"#),
        ]);
        let config = test_config(base_url);

        let text = transcribe_blocking(&config, &[0.0; 1_600], None, &words(&["Handy"])).unwrap();
        assert_eq!(text, "ok");

        let bodies = server.join().unwrap();
        let first: Value = serde_json::from_str(&bodies[0]).unwrap();
        let second: Value = serde_json::from_str(&bodies[1]).unwrap();
        assert!(first.get("provider").is_some());
        assert!(second.get("provider").is_none());
        assert!(is_known_rejected(&rejection_key(&config)));
    }

    #[test]
    fn retries_once_after_a_gateway_error() {
        let (base_url, server) =
            scripted_server(vec![(503, "{}"), (200, r#"{"text":"after retry"}"#)]);

        let text = transcribe_blocking(&test_config(base_url), &[0.0; 1_600], None, &[]).unwrap();
        assert_eq!(text, "after retry");
        assert_eq!(server.join().unwrap().len(), 2);
    }

    #[test]
    fn surfaces_provider_errors_with_their_message() {
        let (base_url, server) = scripted_server(vec![(
            401,
            r#"{"error":{"message":"User not found.","code":401}}"#,
        )]);

        let error =
            transcribe_blocking(&test_config(base_url), &[0.0; 1_600], None, &[]).unwrap_err();
        server.join().unwrap();
        assert_eq!(
            error,
            "OpenRouter error 401: the API key was rejected (User not found.)"
        );
    }

    #[test]
    fn includes_the_wrapped_upstream_message() {
        let (base_url, server) = scripted_server(vec![(
            400,
            r#"{"error":{"message":"Provider returned error","metadata":{"raw":"{\"error\":{\"message\":\"Invalid language 'xx'.\"}}"}}}"#,
        )]);

        let error =
            transcribe_blocking(&test_config(base_url), &[0.0; 1_600], None, &[]).unwrap_err();
        server.join().unwrap();
        assert!(
            error.ends_with("(Provider returned error: Invalid language 'xx'.)"),
            "{error}"
        );
    }

    #[test]
    fn works_when_called_from_the_only_worker_of_a_busy_runtime() {
        // Mirrors the shortcut path: the synchronous transcribe() runs inside a
        // task that occupies its runtime's only worker thread.
        let (base_url, server) = scripted_server(vec![(200, r#"{"text":"from a worker"}"#)]);
        let config = test_config(base_url);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();

        let text = rt
            .block_on(async move {
                tokio::spawn(async move { transcribe_blocking(&config, &[0.0; 1_600], None, &[]) })
                    .await
                    .unwrap()
            })
            .unwrap();
        assert_eq!(text, "from a worker");
        server.join().unwrap();
    }

    #[test]
    fn lists_models_sorted_by_id() {
        let (base_url, server) = scripted_server(vec![(
            200,
            r#"{"data":[{"id":"openai/whisper-1","name":"OpenAI: Whisper 1"},{"id":"openai/gpt-transcribe","name":"OpenAI: GPT Transcribe","description":"  High-accuracy speech-to-text.  "},{"id":"x/unnamed"}]}"#,
        )]);

        let models = runtime()
            .unwrap()
            .block_on(fetch_models_from(base_url))
            .unwrap();
        server.join().unwrap();
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            ["openai/gpt-transcribe", "openai/whisper-1", "x/unnamed"]
        );
        assert_eq!(models[0].name, "OpenAI: GPT Transcribe");
        assert_eq!(models[0].description, "High-accuracy speech-to-text.");
        assert_eq!(models[1].description, "");
        assert_eq!(models[2].name, "x/unnamed");
    }

    /// One canned answer of `routed_server`, picked by path and, when
    /// `token` is set, by the request's bearer token.
    struct Route {
        path: &'static str,
        token: Option<&'static str>,
        status: u16,
        body: &'static str,
    }

    fn route(path: &'static str, status: u16, body: &'static str) -> Route {
        Route {
            path,
            token: None,
            status,
            body,
        }
    }

    fn route_for(
        token: &'static str,
        path: &'static str,
        status: u16,
        body: &'static str,
    ) -> Route {
        Route {
            token: Some(token),
            ..route(path, status, body)
        }
    }

    /// HTTP server that answers `requests` requests — possibly arriving
    /// concurrently — by route. Joining the handle yields "path token" lines.
    fn routed_server(
        routes: Vec<Route>,
        requests: usize,
    ) -> (String, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let handle = thread::spawn(move || {
            let mut seen = Vec::new();
            for _ in 0..requests {
                let (stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream);
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let path = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("")
                    .to_string();
                let mut token = String::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("authorization") {
                            token = value.trim().trim_start_matches("Bearer ").to_string();
                        }
                    }
                }
                let answer = routes
                    .iter()
                    .find(|r| r.path == path && r.token.is_none_or(|t| t == token))
                    .unwrap_or_else(|| panic!("no route for {path} with token {token}"));
                seen.push(format!("{path} {token}"));

                let mut stream = reader.into_inner();
                write!(
                    stream,
                    "HTTP/1.1 {} Routed\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    answer.status,
                    answer.body.len(),
                    answer.body
                )
                .unwrap();
            }
            seen
        });
        (base_url, handle)
    }

    const KEY_WITH_LIMIT: &str = r#"{"data":{"label":"sk-or-v1-abc...xyz","limit":5,"limit_reset":"monthly","limit_remaining":4.88,"include_byok_in_limit":false,"usage":0.3,"usage_daily":0.01,"usage_weekly":0.05,"usage_monthly":0.12,"byok_usage":0,"is_free_tier":false,"free_model_daily_requests":{"used":0,"limit":1000,"remaining":1000}}}"#;
    const MANAGEMENT_ONLY: &str =
        r#"{"error":{"message":"Only management keys can perform this operation","code":403}}"#;
    const BALANCE: &str = r#"{"data":{"total_credits":20,"total_usage":7.66}}"#;

    fn check_key_against(
        routes: Vec<Route>,
        requests: usize,
        management_key: Option<&str>,
    ) -> (Result<CloudKeyCheck, String>, Vec<String>) {
        let (base_url, server) = routed_server(routes, requests);
        let result = runtime().unwrap().block_on(check_key_with(
            test_config(base_url),
            management_key.map(str::to_string),
        ));
        (result, server.join().unwrap())
    }

    #[test]
    fn key_check_reads_spending_and_limit() {
        let (check, _) = check_key_against(
            vec![
                route("/key", 200, KEY_WITH_LIMIT),
                route("/credits", 403, MANAGEMENT_ONLY),
            ],
            2,
            None,
        );
        let check = check.unwrap();

        assert!(check.rejected.is_none());
        assert_eq!(check.account, CloudAccountStatus::NeedsManagementKey);
        let info = check.info.unwrap();
        assert_eq!(info.label, "sk-or-v1-abc...xyz");
        assert_eq!(info.limit, Some(5.0));
        assert_eq!(info.limit_remaining, Some(4.88));
        assert_eq!(info.limit_reset.as_deref(), Some("monthly"));
        assert_eq!(
            (
                info.usage,
                info.usage_daily,
                info.usage_weekly,
                info.usage_monthly
            ),
            (0.3, 0.01, 0.05, 0.12)
        );
        assert!(!info.is_free_tier);
    }

    #[test]
    fn key_check_tolerates_keys_without_limit_and_missing_fields() {
        let (check, _) = check_key_against(
            vec![
                route(
                    "/key",
                    200,
                    r#"{"data":{"label":"","limit":null,"limit_remaining":null,"limit_reset":"","usage":1.5,"is_free_tier":true}}"#,
                ),
                route("/credits", 403, MANAGEMENT_ONLY),
            ],
            2,
            None,
        );

        let info = check.unwrap().info.unwrap();
        assert_eq!(info.limit, None);
        assert_eq!(info.limit_remaining, None);
        assert_eq!(info.limit_reset, None);
        assert_eq!(info.usage, 1.5);
        assert_eq!(info.usage_daily, 0.0);
        assert!(info.is_free_tier);
    }

    #[test]
    fn key_check_reports_a_rejected_key_as_a_verdict() {
        let unknown_user = r#"{"error":{"message":"User not found.","code":401}}"#;
        let (check, _) = check_key_against(
            vec![
                route("/key", 401, unknown_user),
                route("/credits", 401, unknown_user),
            ],
            2,
            None,
        );
        let check = check.unwrap();

        assert!(check.info.is_none());
        assert_eq!(
            check.rejected.as_deref(),
            Some("OpenRouter error 401: the API key was rejected (User not found.)")
        );
    }

    #[test]
    fn key_check_fails_when_openrouter_errors() {
        let boom = r#"{"error":{"message":"boom"}}"#;
        let (check, _) = check_key_against(
            vec![route("/key", 500, boom), route("/credits", 500, boom)],
            2,
            None,
        );
        let error = check.unwrap_err();
        assert!(error.starts_with("OpenRouter error 500"), "{error}");
    }

    #[test]
    fn account_balance_is_read_with_the_inference_key_when_allowed() {
        let (check, _) = check_key_against(
            vec![
                route("/key", 200, KEY_WITH_LIMIT),
                route("/credits", 200, BALANCE),
            ],
            2,
            None,
        );
        assert_eq!(
            check.unwrap().account,
            CloudAccountStatus::Available {
                total_credits: 20.0,
                total_usage: 7.66
            }
        );
    }

    #[test]
    fn account_balance_falls_back_to_the_management_key_and_remembers_it() {
        let (base_url, server) = routed_server(
            vec![
                route("/key", 200, KEY_WITH_LIMIT),
                route_for("test-key", "/credits", 403, MANAGEMENT_ONLY),
                route_for("mgmt-key", "/credits", 200, BALANCE),
            ],
            5,
        );
        let check = |management_key: &str| {
            runtime()
                .unwrap()
                .block_on(check_key_with(
                    test_config(base_url.clone()),
                    Some(management_key.to_string()),
                ))
                .unwrap()
        };

        let first = check("mgmt-key");
        let second = check("mgmt-key");
        let expected = CloudAccountStatus::Available {
            total_credits: 20.0,
            total_usage: 7.66,
        };
        assert_eq!(first.account, expected);
        assert_eq!(second.account, expected);

        // After the 403, the inference key is no longer offered to /credits.
        let requests = server.join().unwrap();
        let inference_attempts = requests
            .iter()
            .filter(|request| request.as_str() == "/credits test-key")
            .count();
        assert_eq!(inference_attempts, 1, "{requests:?}");
    }

    #[test]
    fn account_balance_reports_a_rejected_management_key() {
        let (check, _) = check_key_against(
            vec![
                route("/key", 200, KEY_WITH_LIMIT),
                route_for("test-key", "/credits", 403, MANAGEMENT_ONLY),
                route_for(
                    "bad-mgmt",
                    "/credits",
                    401,
                    r#"{"error":{"message":"Invalid management key","code":401}}"#,
                ),
            ],
            3,
            Some("bad-mgmt"),
        );
        assert_eq!(
            check.unwrap().account,
            CloudAccountStatus::ManagementKeyRejected {
                message: "OpenRouter error 401: the API key was rejected (Invalid management key)"
                    .to_string()
            }
        );
    }

    #[test]
    fn language_hints_are_iso_639_1_or_absent() {
        assert_eq!(language_hint("auto"), None);
        assert_eq!(language_hint(""), None);
        assert_eq!(language_hint("ru").as_deref(), Some("ru"));
        assert_eq!(language_hint("en-US").as_deref(), Some("en"));
        assert_eq!(language_hint("zh-Hans").as_deref(), Some("zh"));
        assert_eq!(language_hint("zh-Hant").as_deref(), Some("zh"));
        assert_eq!(language_hint("jw").as_deref(), Some("jv"));
        assert_eq!(language_hint("yue"), None);
        assert_eq!(language_hint("haw"), None);
    }

    #[test]
    fn keyword_hints_only_for_gpt_transcribe_and_only_valid_words() {
        assert!(keyword_hints("openai/whisper-1", &words(&["Handy"])).is_none());
        assert!(keyword_hints(DEFAULT_CLOUD_MODEL, &[]).is_none());
        assert!(keyword_hints(DEFAULT_CLOUD_MODEL, &words(&["  ", "<tag>"])).is_none());

        let hints = keyword_hints(
            "openai/gpt-transcribe-20260805",
            &words(&[" Handy ", "a\nb", "OpenRouter"]),
        )
        .unwrap();
        assert_eq!(
            hints["options"]["openai"]["keywords"],
            json!(["Handy", "OpenRouter"])
        );
    }

    #[test]
    fn request_timeout_grows_with_audio_and_is_capped() {
        assert_eq!(request_timeout(0), REQUEST_TIMEOUT_BASE);
        assert_eq!(
            request_timeout(16_000 * 30),
            REQUEST_TIMEOUT_BASE + Duration::from_secs(30)
        );
        assert_eq!(request_timeout(16_000 * 3_600), REQUEST_TIMEOUT_MAX);
    }

    #[test]
    fn long_error_messages_are_truncated() {
        assert_eq!(truncate_chars("short", 10), "short");
        assert_eq!(truncate_chars("абвгд", 3), "абв…");
    }

    #[test]
    fn config_requires_an_api_key_and_defaults_the_model() {
        let mut settings = crate::settings::get_default_settings();
        assert!(CloudConfig::from_settings(&settings).is_err());

        settings.post_process_api_keys.insert(
            OPENROUTER_PROVIDER_ID.to_string(),
            " sk-or-test ".to_string(),
        );
        settings.cloud_transcription_model = "  ".to_string();
        let config = CloudConfig::from_settings(&settings).unwrap();
        assert_eq!(config.api_key, "sk-or-test");
        assert_eq!(config.model, DEFAULT_CLOUD_MODEL);
        assert_eq!(config.base_url, DEFAULT_BASE_URL);
    }
}
