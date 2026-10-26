import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // Rutas relativas: el build funciona en la raíz del dominio o en una subcarpeta
  base: './',
  plugins: [tailwindcss()],
  build: {
    chunkSizeWarningLimit: 2000,
  },
})
