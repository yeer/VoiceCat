#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin
xcrun clang -fobjc-arc -O2 -Wall -Wextra -mmacosx-version-min=15.0 \
  -framework Foundation \
  -framework CoreAudio \
  -framework AudioToolbox \
  -framework CoreMedia \
  -framework CoreGraphics \
  -framework ScreenCaptureKit \
  -framework AVFoundation \
  -o bin/sysaudio-capture \
  native/capture.m
echo "built $(pwd)/bin/sysaudio-capture"
