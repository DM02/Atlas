import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Allows access via nginx and tunnels (e.g. Cloudflare quick tunnels)
    // whose hostname isn't known in advance.
    allowedHosts: true,
  },
});
