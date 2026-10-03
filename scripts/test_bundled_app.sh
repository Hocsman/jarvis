#!/bin/bash
# Test script to build and run the bundled app locally.
#
# macOS builds dist/Jarvis.app, Linux builds the onedir folder dist/Jarvis/.
# scripts/check_bundle_layout.py derives the data files the app reads at
# runtime from the source tree and names any that the build is missing,
# before the app is launched.

set -e

echo "🔨 Building Jarvis Desktop App with PyInstaller..."
echo ""

# Get to project root
cd "$(dirname "$0")/.." || exit

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build dist
echo ""

# Build with PyInstaller
echo "📦 Building app bundle..."
python -m PyInstaller jarvis_desktop.spec
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
    BUNDLE="dist/Jarvis.app"
    EXECUTABLE="dist/Jarvis.app/Contents/MacOS/Jarvis"
else
    BUNDLE="dist/Jarvis"
    EXECUTABLE="dist/Jarvis/Jarvis"
fi

# Check the build produced the executable
if [ ! -f "$EXECUTABLE" ]; then
    echo "❌ Build failed! $EXECUTABLE was not produced. Check the output above for errors."
    exit 1
fi

# Check the bundle carries every data file and licence text where it belongs
echo "🔍 Checking the build layout..."
if ! python scripts/check_bundle_layout.py --dist dist; then
    echo ""
    echo "❌ Build is incomplete: the files listed above are missing from $BUNDLE."
    echo "   Add them to the datas list in jarvis_desktop.spec."
    exit 1
fi
echo ""

echo "✅ Build successful!"
echo ""
echo "📍 App location: $(pwd)/$BUNDLE"
echo ""

chmod +x "$EXECUTABLE"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Show app contents for debugging
    echo "📂 App structure:"
    ls -lh dist/Jarvis.app/Contents/MacOS/
    echo ""

    # Run the app in terminal to see output
    echo "🚀 Launching app (console mode enabled for debugging)..."
    echo "   This should open a Terminal window showing the app's output"
    echo "   If successful, you'll see the Jarvis icon in the menu bar"
    echo ""

    open -a Terminal dist/Jarvis.app

    echo ""
    echo "📝 If the app crashes or fails:"
    echo "   1. Check the Terminal window that opened for error messages"
    echo "   2. Check ~/Library/Logs/jarvis_desktop_crash.log"
    echo "   3. Run manually: ./dist/Jarvis.app/Contents/MacOS/Jarvis"
    echo ""
else
    echo "🚀 Launching app..."
    echo "   Press Ctrl+C in this terminal to stop the app"
    echo ""

    "./$EXECUTABLE"
fi
