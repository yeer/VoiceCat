#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
APP="dist/VoiceCat.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp packaging/Info.plist "$APP/Contents/Info.plist"
cp packaging/macos-launch "$APP/Contents/MacOS/launch"
cp desktop/icons/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
if [[ -x desktop/target/debug/sysaudio-stt-desktop ]]; then
  cp desktop/target/debug/sysaudio-stt-desktop "$APP/Contents/MacOS/sysaudio-stt-desktop"
  chmod +x "$APP/Contents/MacOS/sysaudio-stt-desktop"
fi
chmod +x "$APP/Contents/MacOS/launch"
printf '%s\n' "$(pwd)" > "$APP/Contents/Resources/engine-root"
SUPPORT="$HOME/Library/Application Support/VoiceCat"
mkdir -p "$SUPPORT" "$HOME/Applications"
printf '%s\n' "$(pwd)" > "$SUPPORT/engine-root"
rm -rf "$HOME/Applications/VoiceCat.app"
cp -R "$APP" "$HOME/Applications/VoiceCat.app"
echo "已安装 $HOME/Applications/VoiceCat.app"
echo "配置保存在 $SUPPORT，不再写入项目目录。"
