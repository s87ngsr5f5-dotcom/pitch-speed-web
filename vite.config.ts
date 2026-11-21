import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  base: './',
  publicDir: false,
  plugins: [
    react(),
    tailwindcss(),
    viteStaticCopy({
      targets: [
        { src: ['public/manifest.json', 'public/sw.js'], dest: '.' },
        { src: ['public/icons/apple-touch-icon.png', 'public/icons/icon-{192,512}.png'], dest: 'icons' },
        { src: 'public/models/ball_yolo11n.onnx', dest: 'models' },
        {
          src: 'node_modules/onnxruntime-web/dist/*.{wasm,mjs}',
          dest: 'wasm',
        },
      ],
    }),
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web', 'onnxruntime-web/webgpu'],
  },
  worker: {
    format: 'es',
  },
});
