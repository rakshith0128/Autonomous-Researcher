import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Mirrors the `paths` entry in tsconfig.json; both are needed because
    // tsconfig drives the typechecker and this drives the bundler.
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    // In development the API runs as a separate uvicorn process. In production
    // FastAPI serves this bundle itself from the same origin, so the proxy
    // only exists here and no CORS configuration is needed in either mode.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        // SSE breaks if the proxy buffers, so keep the connection raw.
        ws: false,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        // Plotly is ~3MB and only needed once results exist. Splitting it out
        // keeps the initial load -- the part a reviewer waits through -- small.
        manualChunks: {
          plotly: ["plotly.js-dist-min"],
          flow: ["@xyflow/react"],
        },
      },
    },
  },
});
