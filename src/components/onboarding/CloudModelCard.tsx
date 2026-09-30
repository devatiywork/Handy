import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { openUrl } from "@tauri-apps/plugin-opener";
import { Cloud, ExternalLink, Loader2, XCircle } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { OPENROUTER_KEYS_URL } from "../../lib/utils/cloudModel";

interface CloudModelCardProps {
  /** Key already saved in settings, if any, to prefill the field. */
  initialApiKey: string;
  disabled?: boolean;
  isSelecting?: boolean;
  /** Shown under the field, e.g. when OpenRouter rejected the key. */
  error?: string | null;
  onUse: (apiKey: string) => void;
  onKeyEdit?: () => void;
}

/** Onboarding alternative to downloading a model: OpenRouter cloud transcription. */
const CloudModelCard: React.FC<CloudModelCardProps> = ({
  initialApiKey,
  disabled = false,
  isSelecting = false,
  error = null,
  onUse,
  onKeyEdit,
}) => {
  const { t } = useTranslation();
  const [apiKey, setApiKey] = useState(initialApiKey);

  useEffect(() => {
    setApiKey(initialApiKey);
  }, [initialApiKey]);

  const trimmedKey = apiKey.trim();
  const canUse = trimmedKey !== "" && !disabled;

  return (
    <div className="flex flex-col rounded-xl px-4 py-3 gap-2 text-left border-2 border-mid-gray/20">
      <div className="flex items-center gap-2">
        <span className="flex items-center justify-center w-7 h-7 rounded-lg bg-logo-primary/15 text-background-ui">
          <Cloud className="w-4 h-4" />
        </span>
        <h3 className="text-base font-semibold text-text">
          {t("onboarding.models.openrouter-cloud.name")}
        </h3>
      </div>
      <p className="text-text/60 text-sm leading-relaxed">
        {t("onboarding.cloud.description")}
      </p>
      <form
        className="flex items-center gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (canUse) onUse(trimmedKey);
        }}
      >
        <Input
          type="password"
          value={apiKey}
          onChange={(event) => {
            setApiKey(event.target.value);
            onKeyEdit?.();
          }}
          placeholder={t("onboarding.cloud.apiKeyPlaceholder")}
          variant="compact"
          disabled={disabled}
          spellCheck={false}
          autoComplete="off"
          className="flex-1 min-w-0 font-mono"
        />
        <Button
          type="submit"
          variant="primary"
          size="md"
          disabled={!canUse}
          className="flex items-center gap-1.5 shrink-0"
        >
          {isSelecting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          {t("onboarding.cloud.use")}
        </Button>
      </form>
      {error && (
        <p className="flex items-center gap-1.5 text-xs text-error">
          <XCircle className="w-3.5 h-3.5 shrink-0" />
          {error}
        </p>
      )}
      <button
        type="button"
        onClick={() => void openUrl(OPENROUTER_KEYS_URL)}
        className="flex items-center gap-1 self-start text-xs text-text/50 hover:text-background-ui transition-colors cursor-pointer"
      >
        {t("onboarding.cloud.getKey")}
        <ExternalLink className="w-3 h-3" />
      </button>
    </div>
  );
};

export default CloudModelCard;
