import { useCallback, useEffect, useRef, useState } from "react";
import {
  commands,
  type CloudAccountStatus,
  type CloudKeyInfo,
} from "@/bindings";

export type CloudKeyState =
  | { status: "none" }
  /** Keeps the last result so a refresh doesn't blank the balance. */
  | {
      status: "checking";
      info: CloudKeyInfo | null;
      account: CloudAccountStatus | null;
    }
  | { status: "valid"; info: CloudKeyInfo; account: CloudAccountStatus }
  | { status: "rejected"; message: string }
  | { status: "error"; message: string };

/**
 * Checks the saved OpenRouter key and loads its spending plus the account
 * balance whenever either key changes; `refresh` re-runs the check, e.g. to
 * update the balance after dictating. Answers for older keys are dropped.
 */
export const useCloudKeyCheck = (apiKey: string, managementKey: string) => {
  const [state, setState] = useState<CloudKeyState>({ status: "none" });
  const latestRequest = useRef(0);

  const refresh = useCallback(async () => {
    const request = ++latestRequest.current;
    if (apiKey.trim() === "") {
      setState({ status: "none" });
      return;
    }

    setState((previous) =>
      previous.status === "valid" || previous.status === "checking"
        ? { status: "checking", info: previous.info, account: previous.account }
        : { status: "checking", info: null, account: null },
    );
    try {
      const result = await commands.checkCloudTranscriptionKey();
      if (request !== latestRequest.current) return;
      if (result.status === "error") {
        setState({ status: "error", message: result.error });
      } else if (result.data.info) {
        setState({
          status: "valid",
          info: result.data.info,
          account: result.data.account,
        });
      } else {
        setState({ status: "rejected", message: result.data.rejected ?? "" });
      }
    } catch (error) {
      if (request === latestRequest.current) {
        setState({ status: "error", message: String(error) });
      }
    }
    // The management key only matters to the backend, which reads it from
    // settings; it is a dependency so changing it triggers a new check.
  }, [apiKey, managementKey]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { state, refresh };
};
