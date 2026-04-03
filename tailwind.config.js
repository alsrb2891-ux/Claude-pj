/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0a0a1a',
          800: '#0f1029',
          700: '#151638',
          600: '#1c1e47',
        },
        accent: {
          gold: '#f0b90b',
          green: '#00c087',
          red: '#f6465d',
          blue: '#1e90ff',
          cyan: '#00d4ff',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
