/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        display: ['"Outfit"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        medical: {
          bg: '#070b14',
          card: 'rgba(15, 23, 42, 0.85)',
          border: 'rgba(255, 255, 255, 0.08)',
          lad: '#ef4444',
          lcx: '#f59e0b',
          rca: '#06b6d4',
        }
      }
    },
  },
  plugins: [],
}
