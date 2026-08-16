/// <reference types="vite/client" />

/**
 * Build-time configuration.
 *
 * Declared explicitly rather than relying on Vite's catch-all index signature,
 * so a typo in the variable name is a compile error rather than a silently
 * undefined value that quietly sends every API call to the wrong origin.
 */
interface ImportMetaEnv {
  /**
   * Absolute origin of the backend, e.g. `https://my-api.up.railway.app`.
   *
   * Leave unset for a single-container deployment, where the backend serves
   * this bundle and same-origin relative paths are correct.
   */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
