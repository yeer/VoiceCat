$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
if ($env:OS -ne "Windows_NT") {
  Write-Error "Windows 安装包必须在 Windows 上构建。当前机器不能交叉编译。"
}
python -m pip install pyaudiowpatch
Set-Location desktop
cargo build
Write-Host "已生成 desktop/target/debug/sysaudio-stt-desktop.exe"
