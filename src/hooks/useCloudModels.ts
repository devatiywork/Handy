import { useCallback, useEffect, useState } from "react";
import { commands, type CloudModelOption } from "@/bindings";

/**
 * OpenRouter's speech-to-text models. The catalog is public, but it is only
 * requested while `enabled` (the user opted into cloud transcription), so
 * local-only users never contact OpenRouter.
 */
export const useCloudModels = (enabled: boolean) => {
  const [models, setModels] = useState<CloudModelOption[]>([]);
  const [isFetching, setIsFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsFetching(true);
    setError(null);
    try {
      const result = await commands.fetchCloudTranscriptionModels();
      if (result.status === "ok") {
        setModels(result.data);
      } else {
        setError(result.error);
      }
    } catch (fetchError) {
      setError(String(fetchError));
    } finally {
      setIsFetching(false);
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      void refresh();
    }
  }, [enabled, refresh]);

  return { models, isFetching, error, refresh };
};
