import { defineConfig } from 'vite';

// Served from a subpath of the monorepo Pages site, so use relative asset URLs.
// Local `vite dev` and `vite preview` handle the base path transparently.
export default defineConfig({
  base: './',
});
