/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Backgrounds - Deep Void
        "bg-dark": "hsl(0, 0%, 2%)",           // #050505
        "surface": "hsl(225, 15%, 7%)",        // #0F1115
        "surface-hover": "hsl(225, 15%, 10%)", // Hover state

        // Borders
        "border": "hsl(220, 15%, 20%)",        // #2A2F3A
        "border-light": "hsl(220, 15%, 15%)",

        // Accent Colors
        "signal-green": "hsl(152, 100%, 47%)", // #00F090
        "crimson": "hsl(352, 90%, 58%)",       // #FF2E50
        "purple": "hsl(270, 70%, 60%)",        // Deep reasoning
        "amber": "hsl(45, 100%, 50%)",         // Warning/neutral

        // Text
        "text-primary": "rgba(255, 255, 255, 0.95)",
        "text-secondary": "rgba(255, 255, 255, 0.7)",
        "text-muted": "hsl(220, 10%, 50%)",
      },
      fontFamily: {
        "sans": ["Inter", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        "mono": ["JetBrains Mono", "Menlo", "monospace"],
      },
      animation: {
        "pulse-green": "pulse-green 2s ease-in-out infinite",
        "live-pulse": "live-pulse 2s ease-in-out infinite",
        "slide-up": "slide-up 0.3s ease-out",
        "fade-in": "fade-in 0.2s ease-out",
      },
      keyframes: {
        "pulse-green": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.6" },
        },
        "live-pulse": {
          "0%, 100%": {
            opacity: "1",
            transform: "scale(1)",
          },
          "50%": {
            opacity: "0.5",
            transform: "scale(1.2)",
          },
        },
        "slide-up": {
          "0%": {
            opacity: "0",
            transform: "translateY(10px)",
          },
          "100%": {
            opacity: "1",
            transform: "translateY(0)",
          },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
}
