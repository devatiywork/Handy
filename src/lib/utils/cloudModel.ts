import type { CloudModelOption, ModelInfo } from "@/bindings";

// Mirrors the constants in src-tauri/src/cloud_transcription.rs.

/** Registry id of the OpenRouter cloud entry in the model list. */
export const CLOUD_MODEL_ID = "openrouter-cloud";

/** Post-processing provider whose API key cloud transcription shares. */
export const OPENROUTER_PROVIDER_ID = "openrouter";

/** Remote model used until the user picks another one. */
export const DEFAULT_CLOUD_MODEL = "openai/gpt-transcribe";

/** OpenRouter's name for the default, for when its model list isn't loaded. */
export const DEFAULT_CLOUD_MODEL_NAME = "OpenAI: GPT Transcribe";

/** Where keys are created and their spending limits set. */
export const OPENROUTER_KEYS_URL = "https://openrouter.ai/settings/keys";

export const OPENROUTER_CREDITS_URL = "https://openrouter.ai/settings/credits";

/** Management keys are the only keys OpenRouter shows the account balance. */
export const OPENROUTER_MANAGEMENT_KEYS_URL =
  "https://openrouter.ai/settings/management-keys";

/** Cloud models run remotely: nothing to download, delete, or size on disk. */
export const isCloudModel = (model: ModelInfo): boolean =>
  model.source === "Cloud";

/** Prefix of the list ids given to OpenRouter models shown as cards. */
const CLOUD_CARD_PREFIX = "openrouter/";

/** First sentence of OpenRouter's description, short enough for a card. */
const summarize = (description: string): string => {
  const text = description.trim();
  const end = text.search(/[.!?](\s|$)/);
  const sentence = end >= 0 ? text.slice(0, end + 1) : text;
  return sentence.length > 160
    ? `${sentence.slice(0, 157).trimEnd()}…`
    : sentence;
};

/**
 * Present one OpenRouter model as a regular model card. All of them run
 * through the single cloud registry entry (`base`); the card only differs in
 * name, description and the remote id (kept in `filename`).
 */
export const cloudModelCard = (
  option: CloudModelOption,
  base: ModelInfo,
  fallbackDescription: string,
): ModelInfo => ({
  ...base,
  id: `${CLOUD_CARD_PREFIX}${option.id}`,
  name: option.name,
  description: summarize(option.description) || fallbackDescription,
  filename: option.id,
  is_recommended: option.id === DEFAULT_CLOUD_MODEL,
});

/** The OpenRouter model id behind a card built by `cloudModelCard`. */
export const cloudCardModelId = (cardId: string): string | null =>
  cardId.startsWith(CLOUD_CARD_PREFIX)
    ? cardId.slice(CLOUD_CARD_PREFIX.length)
    : null;

/**
 * Format OpenRouter credits (USD). Sub-cent amounts keep four decimals so a
 * few dictations still show up as spending instead of rounding to $0.00.
 */
export const formatUsd = (value: number, locale: string): string => {
  const magnitude = Math.abs(value);
  const digits = magnitude > 0 && magnitude < 0.01 ? 4 : 2;
  try {
    return new Intl.NumberFormat(locale, {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(value);
  } catch {
    return `$${value.toFixed(digits)}`;
  }
};
