export default defineNuxtConfig({
  css: ['vuetify/styles'],
  build: {
    transpile: ['vuetify']
  },
  modules: [
    "@pinia/nuxt"
  ],
  plugins: ['~/plugins/vuetify'],
  vite: {
    define: {
      'process.env.DEBUG': false
    }
  },

  imports: {
    dirs: ["stores/*.ts"],
  },
  compatibilityDate: '2025-03-07'
})