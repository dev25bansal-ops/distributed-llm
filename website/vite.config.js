import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  // Multi-page app configuration
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        docs: resolve(__dirname, 'docs.html'),
        blog: resolve(__dirname, 'blog.html'),
        benchmarks: resolve(__dirname, 'benchmarks.html'),
        tutorials: resolve(__dirname, 'tutorials.html'),
        security: resolve(__dirname, 'security.html'),
        community: resolve(__dirname, 'community.html'),
        changelog: resolve(__dirname, 'changelog.html'),
        'pricing-compare': resolve(__dirname, 'pricing-compare.html'),
        'use-cases': resolve(__dirname, 'use-cases.html'),
        showcase: resolve(__dirname, 'showcase.html'),
        status: resolve(__dirname, 'status.html'),
        playground: resolve(__dirname, 'playground.html'),
        glossary: resolve(__dirname, 'glossary.html'),
        '404': resolve(__dirname, '404.html'),
      },
      output: {
        // Asset hashing for cache busting
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]',
        // Manual chunks for code splitting
        manualChunks: {
          'vendor-theme': ['./js/theme.js'],
          'vendor-utils': ['./js/utils.js'],
          dashboard: ['./js/cluster-viz.js', './js/monitoring.js', './js/live-dashboard.js'],
          models: ['./js/model-explorer.js', './js/model-rec.js', './js/model-optimizer.js'],
        },
      },
    },
    // Minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
      format: {
        comments: false,
      },
    },
    // Source maps for debugging
    sourcemap: true,
    // Output directory
    outDir: 'dist',
    // Clean output directory before build
    emptyOutDir: true,
  },

  // Development server
  server: {
    port: 3000,
    open: false,
    headers: {
      'Content-Security-Policy': "default-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com https://plausible.io; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://plausible.io; img-src 'self' data: https:; font-src 'self' https://fonts.gstatic.com; connect-src 'self' ws: wss: https://api.github.com https://plausible.io https://pypistats.org https://hub.docker.com",
      'X-Frame-Options': 'DENY',
      'X-Content-Type-Options': 'nosniff',
    },
  },

  // CSS configuration
  css: {
    // CSS code splitting
    codeSplit: true,
  },

  // Resolve aliases for cleaner imports
  resolve: {
    alias: {
      '@': resolve(__dirname, '.'),
      '@js': resolve(__dirname, 'js'),
      '@css': resolve(__dirname, 'css'),
    },
  },
});
