#!/bin/bash
# Build 19Labs desktop app for all platforms
# Prerequisites: npm install in this directory

set -e
cd "$(dirname "$0")"

echo "📦 Installing dependencies..."
npm install

echo ""
echo "🔨 Building for current platform..."
npm run build

echo ""
echo "✅ Build complete! Check the 'dist' folder for installers."
echo ""
echo "To build for specific platforms:"
echo "  npm run build:mac    — macOS (.dmg)"
echo "  npm run build:win    — Windows (.exe)"
echo "  npm run build:linux  — Linux (.AppImage, .deb)"
