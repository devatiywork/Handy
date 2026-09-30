import React, { useEffect, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Cloud,
  ExternalLink,
  Eye,
  EyeOff,
  Info,
  Loader2,
  RefreshCcw,
  XCircle,
} from "lucide-react";
import type { CloudAccountStatus, CloudKeyInfo } from "@/bindings";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ResetButton } from "@/components/ui/ResetButton";
import { useSettings } from "@/hooks/useSettings";
import { useCloudKeyCheck, type CloudKeyState } from "@/hooks/useCloudKeyCheck";
import { useModelStore } from "@/stores/modelStore";
import {
  CLOUD_MODEL_ID,
  DEFAULT_CLOUD_MODEL,
  OPENROUTER_CREDITS_URL,
  OPENROUTER_KEYS_URL,
  OPENROUTER_MANAGEMENT_KEYS_URL,
  OPENROUTER_PROVIDER_ID,
  formatUsd,
} from "@/lib/utils/cloudModel";

/** Below this share of the key's limit, the balance bar turns to a warning. */
const LOW_BALANCE_RATIO = 0.1;

type StatusTone = "active" | "neutral" | "error";

const STATUS_STYLES: Record<StatusTone, { pill: string; dot: string }> = {
  active: { pill: "bg-logo-primary/20 text-text", dot: "bg-background-ui" },
  neutral: { pill: "bg-mid-gray/15 text-text/60", dot: "bg-mid-gray" },
  error: { pill: "bg-error/15 text-error", dot: "bg-error" },
};

const ExternalLinkButton: React.FC<{
  href: string;
  children: React.ReactNode;
}> = ({ href, children }) => (
  <button
    type="button"
    onClick={() => void openUrl(href)}
    className="inline-flex items-center gap-1 text-xs text-text/60 hover:text-background-ui transition-colors cursor-pointer"
  >
    {children}
    <ExternalLink className="w-3 h-3" />
  </button>
);

/** Password field with a reveal toggle; commits on blur and on Enter. */
const SecretInput: React.FC<{
  id: string;
  value: string;
  onCommit: (value: string) => void;
  disabled: boolean;
  placeholder: string;
  showLabel: string;
  hideLabel: string;
}> = ({ id, value, onCommit, disabled, placeholder, showLabel, hideLabel }) => {
  const [draft, setDraft] = useState(value);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  return (
    <div className="relative">
      <Input
        id={id}
        type={visible ? "text" : "password"}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => onCommit(draft)}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
        placeholder={placeholder}
        variant="compact"
        disabled={disabled}
        spellCheck={false}
        autoComplete="off"
        className="w-full pe-9 font-mono"
      />
      <button
        type="button"
        onClick={() => setVisible((shown) => !shown)}
        aria-label={visible ? hideLabel : showLabel}
        title={visible ? hideLabel : showLabel}
        className="absolute inset-y-0 end-0 flex items-center px-2.5 text-text/50 hover:text-text transition-colors cursor-pointer"
      >
        {visible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
      </button>
    </div>
  );
};

const KeyStatus: React.FC<{ state: CloudKeyState; onRecheck: () => void }> = ({
  state,
  onRecheck,
}) => {
  const { t } = useTranslation();
  const recheck = (
    <button
      type="button"
      onClick={onRecheck}
      className="underline underline-offset-2 hover:text-text transition-colors cursor-pointer"
    >
      {t("settings.models.cloud.apiKey.recheck")}
    </button>
  );

  if (state.status === "none") return null;

  // A refresh keeps showing the previous verdict; the balance header spins.
  const info =
    state.status === "valid"
      ? state.info
      : state.status === "checking"
        ? state.info
        : null;
  if (info) {
    return (
      <p className="flex items-center gap-1.5 text-xs text-text/70">
        <Check className="w-3.5 h-3.5 shrink-0 text-background-ui" />
        <span className="truncate">
          {info.label
            ? t("settings.models.cloud.apiKey.valid", { label: info.label })
            : t("settings.models.cloud.apiKey.validPlain")}
        </span>
      </p>
    );
  }

  switch (state.status) {
    case "checking":
      return (
        <p className="flex items-center gap-1.5 text-xs text-text/60">
          <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin" />
          {t("settings.models.cloud.apiKey.checking")}
        </p>
      );
    case "rejected":
      return (
        <div className="space-y-0.5 text-xs">
          <p className="flex items-center gap-1.5 text-error">
            <XCircle className="w-3.5 h-3.5 shrink-0" />
            {t("settings.models.cloud.apiKey.rejected")}
          </p>
          <p className="ps-5 text-text/50 break-words">{state.message}</p>
          <p className="ps-5 text-text/60">{recheck}</p>
        </div>
      );
    case "error":
      return (
        <div className="space-y-0.5 text-xs">
          <p className="flex items-start gap-1.5 text-warning">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
            <span className="break-words">
              {t("settings.models.cloud.apiKey.checkFailed", {
                error: state.message,
              })}
            </span>
          </p>
          <p className="ps-5 text-text/60">{recheck}</p>
        </div>
      );
    default:
      return null;
  }
};

/** Money left on the whole OpenRouter account, when the keys can read it. */
const accountRemaining = (account: CloudAccountStatus | null): number | null =>
  account?.status === "available"
    ? account.total_credits - account.total_usage
    : null;

/**
 * The account's overall balance. OpenRouter only shows it to a management
 * key, so this also hosts the optional field for one.
 */
const AccountBalance: React.FC<{
  account: CloudAccountStatus;
  managementKey: string;
  isSavingManagementKey: boolean;
  onManagementKeyChange: (key: string) => void;
}> = ({
  account,
  managementKey,
  isSavingManagementKey,
  onManagementKeyChange,
}) => {
  const { t, i18n } = useTranslation();
  const money = (value: number) => formatUsd(value, i18n.language);
  const [editingKey, setEditingKey] = useState(
    account.status === "management_key_rejected",
  );
  const showKeyInput =
    editingKey || account.status === "management_key_rejected";

  const keyInput = (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <label
          htmlFor="openrouter-management-key"
          className="text-xs font-medium"
        >
          {t("settings.models.cloud.balance.account.managementKey")}
        </label>
        <ExternalLinkButton href={OPENROUTER_MANAGEMENT_KEYS_URL}>
          {t("settings.models.cloud.balance.account.createManagementKey")}
        </ExternalLinkButton>
      </div>
      <SecretInput
        id="openrouter-management-key"
        value={managementKey}
        onCommit={(value) => {
          const trimmed = value.trim();
          if (trimmed !== managementKey) onManagementKeyChange(trimmed);
        }}
        disabled={isSavingManagementKey}
        placeholder={t("settings.models.cloud.apiKey.placeholder")}
        showLabel={t("settings.models.cloud.apiKey.show")}
        hideLabel={t("settings.models.cloud.apiKey.hide")}
      />
      <p className="text-[11px] leading-4 text-text/50">
        {t("settings.models.cloud.balance.account.managementKeyHint")}
      </p>
    </div>
  );

  return (
    <div className="rounded-lg bg-mid-gray/10 px-3 py-2.5 space-y-2">
      <p className="text-xs text-text/60">
        {t("settings.models.cloud.balance.account.title")}
      </p>

      {account.status === "available" && (
        <div>
          <p className="text-2xl font-semibold tabular-nums leading-8">
            {money(account.total_credits - account.total_usage)}
          </p>
          <p className="text-xs text-text/60">
            {t("settings.models.cloud.balance.account.summary", {
              credits: money(account.total_credits),
              spent: money(account.total_usage),
            })}
          </p>
        </div>
      )}

      {account.status === "needs_management_key" && !showKeyInput && (
        <div className="flex items-center justify-between gap-3">
          <p className="text-xs text-text/60">
            {t("settings.models.cloud.balance.account.needsManagementKey")}
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setEditingKey(true)}
            className="shrink-0"
          >
            {t("settings.models.cloud.balance.account.addManagementKey")}
          </Button>
        </div>
      )}

      {account.status === "management_key_rejected" && (
        <div className="text-xs">
          <p className="flex items-center gap-1.5 text-error">
            <XCircle className="w-3.5 h-3.5 shrink-0" />
            {t("settings.models.cloud.balance.account.managementKeyRejected")}
          </p>
          <p className="ps-5 text-text/50 break-words">{account.message}</p>
        </div>
      )}

      {account.status === "unavailable" && (
        <p className="flex items-start gap-1.5 text-xs text-warning">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span className="break-words">
            {t("settings.models.cloud.balance.account.unavailable", {
              error: account.message,
            })}
          </span>
        </p>
      )}

      {showKeyInput
        ? keyInput
        : managementKey !== "" && (
            <button
              type="button"
              onClick={() => setEditingKey(true)}
              className="text-[11px] text-text/50 underline underline-offset-2 hover:text-text transition-colors cursor-pointer"
            >
              {t("settings.models.cloud.balance.account.managementKey")}
            </button>
          )}
    </div>
  );
};

const KeyBalance: React.FC<{
  info: CloudKeyInfo;
  account: CloudAccountStatus | null;
  managementKey: string;
  isSavingManagementKey: boolean;
  onManagementKeyChange: (key: string) => void;
  isRefreshing: boolean;
  onRefresh: () => void;
}> = ({
  info,
  account,
  managementKey,
  isSavingManagementKey,
  onManagementKeyChange,
  isRefreshing,
  onRefresh,
}) => {
  const { t, i18n } = useTranslation();
  const money = (value: number) => formatUsd(value, i18n.language);

  const limit = info.limit;
  const remaining = Math.max(info.limit_remaining ?? 0, 0);
  const ratio = limit && limit > 0 ? Math.min(remaining / limit, 1) : 0;
  const isLow = limit !== null && ratio < LOW_BALANCE_RATIO;
  const resets = info.limit_reset
    ? t(`settings.models.cloud.balance.resets.${info.limit_reset}`, {
        defaultValue: "",
      })
    : "";

  const spending = [
    { key: "today", value: info.usage_daily },
    { key: "week", value: info.usage_weekly },
    { key: "month", value: info.usage_monthly },
    { key: "total", value: info.usage },
  ];

  return (
    <div className="px-4 py-3 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">
          {t("settings.models.cloud.balance.title")}
        </h3>
        <ResetButton
          onClick={onRefresh}
          disabled={isRefreshing}
          ariaLabel={t("settings.models.cloud.balance.refresh")}
          className="flex h-7 w-7 items-center justify-center"
        >
          <RefreshCcw
            className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`}
          />
        </ResetButton>
      </div>

      {account && (
        <AccountBalance
          // Remount when the verdict changes so a rejected key opens the field.
          key={account.status}
          account={account}
          managementKey={managementKey}
          isSavingManagementKey={isSavingManagementKey}
          onManagementKeyChange={onManagementKeyChange}
        />
      )}

      <div className="space-y-1.5">
        <p className="text-xs text-text/60">
          {t("settings.models.cloud.balance.key.title")}
        </p>
        {limit !== null ? (
          <>
            <div className="flex items-baseline justify-between gap-3">
              <p className="text-sm text-text/70">
                <Trans
                  i18nKey="settings.models.cloud.balance.remaining"
                  values={{ remaining: money(remaining), limit: money(limit) }}
                  components={{
                    b: <span className="text-base font-semibold text-text" />,
                  }}
                />
              </p>
              {resets && (
                <p className="text-xs text-text/50 shrink-0">{resets}</p>
              )}
            </div>
            <div className="h-1.5 w-full rounded-full bg-mid-gray/20 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  isLow ? "bg-warning" : "bg-background-ui"
                }`}
                style={{ width: `${ratio * 100}%` }}
              />
            </div>
          </>
        ) : (
          <p className="text-xs text-text/60">
            {t("settings.models.cloud.balance.noLimit")}{" "}
            <ExternalLinkButton href={OPENROUTER_KEYS_URL}>
              {t("settings.models.cloud.balance.setLimit")}
            </ExternalLinkButton>
          </p>
        )}
      </div>

      <dl className="grid grid-cols-4 gap-2">
        {spending.map(({ key, value }) => (
          <div key={key} className="rounded-md bg-mid-gray/10 px-2.5 py-1.5">
            <dt className="text-[11px] leading-4 text-text/50">
              {t(`settings.models.cloud.balance.${key}`)}
            </dt>
            <dd className="text-sm font-medium tabular-nums">{money(value)}</dd>
          </div>
        ))}
      </dl>

      {info.is_free_tier && (
        <p className="flex items-start gap-1.5 text-xs text-warning">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span>
            {t("settings.models.cloud.balance.freeTier")}{" "}
            <ExternalLinkButton href={OPENROUTER_CREDITS_URL}>
              {t("settings.models.cloud.balance.addCredits")}
            </ExternalLinkButton>
          </span>
        </p>
      )}
    </div>
  );
};

interface CloudTranscriptionSettingsProps {
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
}

/**
 * The OpenRouter "account" card: status, API key and the key's balance. The
 * cloud models themselves are picked from the model list, like local ones.
 */
export const CloudTranscriptionSettings: React.FC<
  CloudTranscriptionSettingsProps
> = ({ expanded, onExpandedChange }) => {
  const { t, i18n } = useTranslation();
  const { settings, isUpdating, updateSetting, updatePostProcessApiKey } =
    useSettings();
  const { currentModel } = useModelStore();
  const userToggled = useRef(false);

  const apiKey =
    settings?.post_process_api_keys?.[OPENROUTER_PROVIDER_ID] ?? "";
  const managementKey = settings?.openrouter_management_key ?? "";
  const model = settings?.cloud_transcription_model || DEFAULT_CLOUD_MODEL;
  const hasApiKey = apiKey.trim() !== "";
  const isActive = currentModel === CLOUD_MODEL_ID;
  const { state: keyState, refresh: refreshKey } = useCloudKeyCheck(
    apiKey,
    managementKey,
  );

  // Open by default once the user has opted in; stay compact otherwise.
  const optedIn = hasApiKey || isActive;
  useEffect(() => {
    if (optedIn && !userToggled.current) {
      onExpandedChange(true);
    }
  }, [optedIn, onExpandedChange]);

  const keyInfo =
    keyState.status === "valid" || keyState.status === "checking"
      ? keyState.info
      : null;
  const keyAccount =
    keyState.status === "valid" || keyState.status === "checking"
      ? keyState.account
      : null;
  const balance = accountRemaining(keyAccount);

  const status: { tone: StatusTone; label: string } =
    keyState.status === "rejected"
      ? { tone: "error", label: t("settings.models.cloud.status.rejected") }
      : isActive
        ? { tone: "active", label: t("settings.models.cloud.status.active") }
        : !hasApiKey
          ? { tone: "neutral", label: t("settings.models.cloud.status.noKey") }
          : { tone: "neutral", label: t("settings.models.cloud.status.off") };
  const statusStyle = STATUS_STYLES[status.tone];

  const handleApiKeyChange = (value: string) => {
    const trimmed = value.trim();
    if (trimmed !== apiKey) {
      void updatePostProcessApiKey(OPENROUTER_PROVIDER_ID, trimmed);
    }
  };

  // The setting is updated optimistically, so re-check only once the backend
  // has stored the key it reads.
  const handleManagementKeyChange = async (key: string) => {
    await updateSetting("openrouter_management_key", key);
    void refreshKey();
  };

  return (
    <div className="bg-background border border-mid-gray/20 rounded-lg">
      <button
        type="button"
        onClick={() => {
          userToggled.current = true;
          onExpandedChange(!expanded);
        }}
        aria-expanded={expanded}
        className={`w-full flex items-center gap-3 px-4 py-3 text-start transition-colors hover:bg-mid-gray/5 cursor-pointer ${
          expanded ? "rounded-t-lg" : "rounded-lg"
        }`}
      >
        <span className="flex items-center justify-center w-9 h-9 shrink-0 rounded-lg bg-logo-primary/15 text-background-ui">
          <Cloud className="w-5 h-5" />
        </span>
        <span className="flex-1 min-w-0">
          <span className="block text-sm font-semibold">
            {t("settings.models.cloud.title")}
          </span>
          <span className="block text-xs text-text/60 truncate">
            {t("settings.models.cloud.subtitle", { model })}
          </span>
        </span>
        {balance !== null && (
          <span
            className="shrink-0 text-sm font-semibold tabular-nums"
            title={t("settings.models.cloud.balance.account.title")}
          >
            {formatUsd(balance, i18n.language)}
          </span>
        )}
        <span
          className={`inline-flex items-center gap-1.5 shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${statusStyle.pill}`}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${statusStyle.dot}`} />
          {status.label}
        </span>
        <ChevronDown
          className={`w-4 h-4 shrink-0 text-text/50 transition-transform ${
            expanded ? "rotate-180" : ""
          }`}
        />
      </button>

      {expanded && (
        <div className="border-t border-mid-gray/20 divide-y divide-mid-gray/20">
          <p className="px-4 py-2.5 text-xs leading-relaxed text-text/60">
            {t("settings.models.cloud.description")}
          </p>

          <div className="px-4 py-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <label
                htmlFor="openrouter-api-key"
                className="text-sm font-medium"
                title={t("settings.models.cloud.apiKey.description")}
              >
                {t("settings.models.cloud.apiKey.title")}
              </label>
              <ExternalLinkButton href={OPENROUTER_KEYS_URL}>
                {t("settings.models.cloud.getKey")}
              </ExternalLinkButton>
            </div>
            <SecretInput
              id="openrouter-api-key"
              value={apiKey}
              onCommit={handleApiKeyChange}
              disabled={isUpdating(
                `post_process_api_key:${OPENROUTER_PROVIDER_ID}`,
              )}
              placeholder={t("settings.models.cloud.apiKey.placeholder")}
              showLabel={t("settings.models.cloud.apiKey.show")}
              hideLabel={t("settings.models.cloud.apiKey.hide")}
            />
            <KeyStatus state={keyState} onRecheck={() => void refreshKey()} />
          </div>

          {keyInfo && (
            <KeyBalance
              info={keyInfo}
              account={keyAccount}
              managementKey={managementKey}
              isSavingManagementKey={isUpdating("openrouter_management_key")}
              onManagementKeyChange={(key) =>
                void handleManagementKeyChange(key)
              }
              isRefreshing={keyState.status === "checking"}
              onRefresh={() => void refreshKey()}
            />
          )}

          {hasApiKey && !isActive && keyState.status !== "rejected" && (
            <p className="flex items-start gap-1.5 px-4 py-2.5 text-xs text-text/60">
              <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              {t("settings.models.cloud.pickModel")}
            </p>
          )}
        </div>
      )}
    </div>
  );
};
